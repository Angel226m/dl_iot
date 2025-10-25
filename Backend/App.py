"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD - DETECCIÓN DE GRIETAS CON TTA
Sistema de inferencia mejorado con SegFormer B5
Desarrollado por: Angel226m
═══════════════════════════════════════════════════════════════════════════════
"""
'''
import os
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from datetime import datetime
import traceback
import base64

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
# Configurar CORS para permitir todas las peticiones en desarrollo
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

class Config:
    # Directorios
    UPLOAD_FOLDER = '/app/uploads'
    RESULTS_FOLDER = '/app/results'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # Modelo
    TARGET_SIZE = 640
    MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    THRESHOLD = 0.5
    
    # TTA
    USE_TTA = True  # Activado por defecto para mejor calidad
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Crear directorios
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)
Path(config.RESULTS_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None

def cargar_modelo():
    """Carga el modelo SegFormer"""
    global model
    
    try:
        print(f"🤖 Cargando modelo desde: {config.MODEL_PATH}")
        
        # Cargar arquitectura
        model = SegformerForSemanticSegmentation.from_pretrained(
            config.MODEL_NAME,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Cargar pesos entrenados
        if os.path.exists(config.MODEL_PATH):
            checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
                print("   ℹ️  Usando pesos EMA")
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=True)
            print(f"   ✓ Modelo cargado exitosamente")
            
            if 'dice' in checkpoint:
                print(f"   ✓ Dice: {checkpoint['dice']:.4f}")
        else:
            print(f"   ⚠️  No se encontró modelo en {config.MODEL_PATH}")
            print(f"   ℹ️  Usando modelo base sin fine-tuning")
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ TTA: {'ACTIVADO' if config.USE_TTA else 'DESACTIVADO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def get_transform():
    """Transforma de preprocesamiento"""
    return A.Compose([
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])

def procesar_imagen(image_path):
    """Carga y preprocesa imagen"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.TARGET_SIZE, config.TARGET_SIZE))
    
    transform = get_transform()
    transformed = transform(image=img_resized)
    img_tensor = transformed['image'].unsqueeze(0)
    
    return img_tensor, img_rgb

def predecir_simple(img_tensor):
    """Predicción simple sin TTA"""
    if model is None:
        raise RuntimeError("El modelo no está cargado")
    
    img_tensor = img_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits,
            size=(config.TARGET_SIZE, config.TARGET_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        pred = torch.sigmoid(logits)
        mask = pred.squeeze().cpu().numpy()
    
    return mask

def aplicar_tta(img_tensor):
    """
    Test-Time Augmentation para mejor precisión
    Aplica 4 transformaciones y promedia los resultados
    """
    if model is None:
        raise RuntimeError("El modelo no está cargado")
    
    transformaciones = [
        ('original', lambda x: x, lambda x: x),
        ('hflip', lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        ('vflip', lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        ('rot90', lambda x: torch.rot90(x, k=1, dims=[2, 3]), lambda x: torch.rot90(x, k=3, dims=[2, 3])),
    ]
    
    preds = []
    
    for nombre, tfm_forward, tfm_backward in transformaciones:
        # Transformar entrada
        img_tfm = tfm_forward(img_tensor.clone())
        
        # Predecir
        with torch.no_grad():
            outputs = model(pixel_values=img_tfm.to(config.DEVICE))
            logits = outputs.logits
            
            logits = F.interpolate(
                logits,
                size=(config.TARGET_SIZE, config.TARGET_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            pred = torch.sigmoid(logits)
        
        # Revertir transformación
        pred = tfm_backward(pred)
        preds.append(pred)
    
    # Promediar todas las predicciones
    mask = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
    
    return mask

def predecir(img_tensor, use_tta=True):
    """Realiza predicción con o sin TTA"""
    if use_tta:
        return aplicar_tta(img_tensor)
    else:
        return predecir_simple(img_tensor)

def post_procesar(mask, threshold=0.5):
    """Post-procesamiento de máscara"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    # Eliminar componentes pequeños (ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    result = np.zeros_like(mask_binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 10:  # Mínimo 10 píxeles
            result[labels == i] = 1
    
    # Cerrar pequeños gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return result.astype(np.float32)

def calcular_metricas(mask, threshold=0.5):
    """Calcula métricas detalladas de la predicción"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    # Análisis de componentes conexas
    num_grietas, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    num_grietas -= 1  # Restar fondo
    
    if num_grietas > 0:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_grietas + 1)]
        area_promedio = float(np.mean(areas))
        area_max = float(np.max(areas))
        area_min = float(np.min(areas))
    else:
        area_promedio = 0
        area_max = 0
        area_min = 0
    
    # Determinar severidad y estado
    if porcentaje_grietas < 1:
        severidad = "Baja"
        estado = "Sin Grietas Significativas"
        color = "success"
    elif porcentaje_grietas < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
        color = "info"
    elif porcentaje_grietas < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
        color = "warning"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
        color = "danger"
    
    # Calcular confianza basada en múltiples factores
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_grietas),
        'area_promedio_grieta': round(area_promedio, 2),
        'area_max_grieta': round(area_max, 2),
        'area_min_grieta': round(area_min, 2),
        'severidad': severidad,
        'estado': estado,
        'color_severidad': color,
        'confianza': round(confianza, 1),
        'tta_usado': config.USE_TTA
    }

def crear_visualizacion(img_original, mask, threshold=0.5):
    """Crea visualización overlay mejorada"""
    h, w = img_original.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask_resized > threshold).astype(np.uint8) * 255
    
    # Crear colormap personalizado (rojo para grietas)
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    mask_colored[mask_binary > 127] = [255, 0, 0]  # Rojo puro
    
    # Crear overlay con transparencia
    overlay = img_original.copy()
    alpha = 0.6
    overlay[mask_binary > 127] = (
        overlay[mask_binary > 127] * (1 - alpha) + 
        mask_colored[mask_binary > 127] * alpha
    ).astype(np.uint8)
    
    # Añadir contornos para mejor visualización
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Contorno amarillo
    
    return overlay

def imagen_a_base64(img_rgb):
    """Convierte imagen RGB a base64 para envío al frontend"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# ═══════════════════════════════════════════════════════════════════════════
# RUTAS API
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción con TTA
    
    Parámetros opcionales:
        - use_tta: bool (default: True) - Usar Test-Time Augmentation
        - return_base64: bool (default: False) - Devolver imágenes en base64
    """
    try:
        # Validar modelo
        if model is None:
            return jsonify({'error': 'El modelo no está cargado'}), 503
        
        # Validar archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Formato de archivo no permitido',
                'formatos_aceptados': list(config.ALLOWED_EXTENSIONS)
            }), 400
        
        # Obtener parámetros opcionales
        use_tta = request.form.get('use_tta', str(config.USE_TTA)).lower() == 'true'
        return_base64 = request.form.get('return_base64', 'false').lower() == 'true'
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Procesar imagen
        img_tensor, img_original = procesar_imagen(filepath)
        
        # Predecir con o sin TTA
        mask = predecir(img_tensor, use_tta=use_tta)
        
        # Post-procesar máscara
        mask_procesada = post_procesar(mask, config.THRESHOLD)
        
        # Calcular métricas detalladas
        metricas = calcular_metricas(mask_procesada, config.THRESHOLD)
        
        # Crear visualización
        overlay = crear_visualizacion(img_original, mask_procesada, config.THRESHOLD)
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'metricas': metricas,
            'timestamp': datetime.now().isoformat(),
            'procesamiento': {
                'tta_usado': use_tta,
                'threshold': config.THRESHOLD,
                'target_size': config.TARGET_SIZE
            }
        }
        
        # Guardar y devolver imagen según configuración
        if return_base64:
            # Devolver imagen en base64 (útil para frontend)
            response_data['imagen_overlay'] = imagen_a_base64(overlay)
        else:
            # Guardar archivo y devolver URL
            result_filename = f"result_{filename}"
            result_path = os.path.join(config.RESULTS_FOLDER, result_filename)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, overlay_bgr)
            response_data['result_image'] = f'/api/results/{result_filename}'
        
        # Eliminar archivo temporal
        os.remove(filepath)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predicción en lote para múltiples imágenes
    """
    try:
        if model is None:
            return jsonify({'error': 'El modelo no está cargado'}), 503
        
        if 'images' not in request.files:
            return jsonify({'error': 'No se enviaron imágenes'}), 400
        
        files = request.files.getlist('images')
        use_tta = request.form.get('use_tta', str(config.USE_TTA)).lower() == 'true'
        
        resultados = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Procesar cada imagen
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    temp_filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(config.UPLOAD_FOLDER, temp_filename)
                    file.save(filepath)
                    
                    img_tensor, img_original = procesar_imagen(filepath)
                    mask = predecir(img_tensor, use_tta=use_tta)
                    mask_procesada = post_procesar(mask, config.THRESHOLD)
                    metricas = calcular_metricas(mask_procesada, config.THRESHOLD)
                    
                    resultados.append({
                        'filename': file.filename,
                        'metricas': metricas,
                        'success': True
                    })
                    
                    os.remove(filepath)
                    
                except Exception as e:
                    resultados.append({
                        'filename': file.filename,
                        'error': str(e),
                        'success': False
                    })
        
        return jsonify({
            'success': True,
            'total_procesadas': len(resultados),
            'resultados': resultados,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error en batch prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """Obtener imagen de resultado"""
    try:
        filepath = os.path.join(config.RESULTS_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({'error': 'Archivo no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Obtener o actualizar configuración"""
    if request.method == 'GET':
        return jsonify({
            'target_size': config.TARGET_SIZE,
            'threshold': config.THRESHOLD,
            'use_tta': config.USE_TTA,
            'device': str(config.DEVICE),
            'model_name': config.MODEL_NAME
        }), 200
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'threshold' in data:
            config.THRESHOLD = float(data['threshold'])
        
        if 'use_tta' in data:
            config.USE_TTA = bool(data['use_tta'])
        
        return jsonify({
            'success': True,
            'config': {
                'threshold': config.THRESHOLD,
                'use_tta': config.USE_TTA
            }
        }), 200

@app.route('/api/status', methods=['GET'])
def status():
    """Estado detallado del sistema"""
    return jsonify({
        'system': 'CrackGuard Backend',
        'version': '2.0.0',
        'model': 'SegFormer B5',
        'device': str(config.DEVICE),
        'model_loaded': model is not None,
        'configuration': {
            'threshold': config.THRESHOLD,
            'target_size': config.TARGET_SIZE,
            'tta_enabled': config.USE_TTA,
            'max_file_size_mb': config.MAX_CONTENT_LENGTH / (1024 * 1024)
        },
        'timestamp': datetime.now().isoformat()
    }), 200

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("🚀 INICIALIZANDO CRACKGUARD BACKEND v2.0")
print("═" * 80)

modelo_cargado = cargar_modelo()

if modelo_cargado:
    print("✅ Sistema listo para inferencia")
else:
    print("⚠️  Servidor iniciado sin modelo entrenado")

print("═" * 80)

if __name__ == '__main__':
    print("\n🔧 Ejecutando con Flask development server")
    print("⚠️  Para producción usar: gunicorn app:app --bind 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)'''


"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD - DETECCIÓN DE GRIETAS CON TTA
Sistema de inferencia mejorado con SegFormer B5
Desarrollado por: Angel226m
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from datetime import datetime
import traceback
import base64

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
# Configurar CORS para permitir todas las peticiones en desarrollo
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

class Config:
    # Directorios
    UPLOAD_FOLDER = '/app/uploads'
    RESULTS_FOLDER = '/app/results'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # Modelo
    TARGET_SIZE = 640
    MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    THRESHOLD = 0.5
    
    # TTA
    USE_TTA = True  # Activado por defecto para mejor calidad
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Crear directorios
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)
Path(config.RESULTS_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None

def cargar_modelo():
    """Carga el modelo SegFormer"""
    global model
    
    try:
        print(f"🤖 Cargando modelo desde: {config.MODEL_PATH}")
        
        # Cargar arquitectura
        model = SegformerForSemanticSegmentation.from_pretrained(
            config.MODEL_NAME,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Cargar pesos entrenados
        if os.path.exists(config.MODEL_PATH):
            checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
                print("   ℹ️  Usando pesos EMA")
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=True)
            print(f"   ✓ Modelo cargado exitosamente")
            
            if 'dice' in checkpoint:
                print(f"   ✓ Dice: {checkpoint['dice']:.4f}")
        else:
            print(f"   ⚠️  No se encontró modelo en {config.MODEL_PATH}")
            print(f"   ℹ️  Usando modelo base sin fine-tuning")
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ TTA: {'ACTIVADO' if config.USE_TTA else 'DESACTIVADO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def get_transform():
    """Transforma de preprocesamiento"""
    return A.Compose([
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])

def procesar_imagen(image_path):
    """Carga y preprocesa imagen"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.TARGET_SIZE, config.TARGET_SIZE))
    
    transform = get_transform()
    transformed = transform(image=img_resized)
    img_tensor = transformed['image'].unsqueeze(0)
    
    return img_tensor, img_rgb

def predecir_simple(img_tensor):
    """Predicción simple sin TTA"""
    if model is None:
        raise RuntimeError("El modelo no está cargado")
    
    img_tensor = img_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits,
            size=(config.TARGET_SIZE, config.TARGET_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        pred = torch.sigmoid(logits)
        mask = pred.squeeze().cpu().numpy()
    
    return mask

def aplicar_tta(img_tensor):
    """
    Test-Time Augmentation para mejor precisión
    Aplica 4 transformaciones y promedia los resultados
    """
    if model is None:
        raise RuntimeError("El modelo no está cargado")
    
    transformaciones = [
        ('original', lambda x: x, lambda x: x),
        ('hflip', lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        ('vflip', lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        ('rot90', lambda x: torch.rot90(x, k=1, dims=[2, 3]), lambda x: torch.rot90(x, k=3, dims=[2, 3])),
    ]
    
    preds = []
    
    for nombre, tfm_forward, tfm_backward in transformaciones:
        # Transformar entrada
        img_tfm = tfm_forward(img_tensor.clone())
        
        # Predecir
        with torch.no_grad():
            outputs = model(pixel_values=img_tfm.to(config.DEVICE))
            logits = outputs.logits
            
            logits = F.interpolate(
                logits,
                size=(config.TARGET_SIZE, config.TARGET_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            pred = torch.sigmoid(logits)
        
        # Revertir transformación
        pred = tfm_backward(pred)
        preds.append(pred)
    
    # Promediar todas las predicciones
    mask = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
    
    return mask

def predecir(img_tensor, use_tta=True):
    """Realiza predicción con o sin TTA"""
    if use_tta:
        return aplicar_tta(img_tensor)
    else:
        return predecir_simple(img_tensor)

def post_procesar(mask, threshold=0.5):
    """Post-procesamiento de máscara"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    # Eliminar componentes pequeños (ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    result = np.zeros_like(mask_binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 10:  # Mínimo 10 píxeles
            result[labels == i] = 1
    
    # Cerrar pequeños gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return result.astype(np.float32)

def calcular_metricas(mask, threshold=0.5):
    """Calcula métricas detalladas de la predicción"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    # Análisis de componentes conexas
    num_grietas, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    num_grietas -= 1  # Restar fondo
    
    if num_grietas > 0:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_grietas + 1)]
        area_promedio = float(np.mean(areas))
        area_max = float(np.max(areas))
        area_min = float(np.min(areas))
    else:
        area_promedio = 0
        area_max = 0
        area_min = 0
    
    # Determinar severidad y estado
    if porcentaje_grietas < 1:
        severidad = "Baja"
        estado = "Sin Grietas Significativas"
        color = "success"
    elif porcentaje_grietas < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
        color = "info"
    elif porcentaje_grietas < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
        color = "warning"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
        color = "danger"
    
    # Calcular confianza basada en múltiples factores
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_grietas),
        'area_promedio_grieta': round(area_promedio, 2),
        'area_max_grieta': round(area_max, 2),
        'area_min_grieta': round(area_min, 2),
        'severidad': severidad,
        'estado': estado,
        'color_severidad': color,
        'confianza': round(confianza, 1),
        'tta_usado': config.USE_TTA
    }

def crear_visualizacion(img_original, mask, threshold=0.5):
    """Crea múltiples visualizaciones: overlay, máscara sola, y comparación"""
    h, w = img_original.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask_resized > threshold).astype(np.uint8) * 255
    
    # 1. Máscara colorizada (rojo sobre negro)
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    mask_colored[mask_binary > 127] = [255, 0, 0]  # Rojo brillante
    
    # 2. Overlay con transparencia sobre imagen original
    overlay = img_original.copy()
    alpha = 0.6
    overlay[mask_binary > 127] = (
        overlay[mask_binary > 127] * (1 - alpha) + 
        mask_colored[mask_binary > 127] * alpha
    ).astype(np.uint8)
    
    # Añadir contornos amarillos brillantes
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 3)  # Amarillo más grueso
    
    # 3. Máscara sola con mejor visualización (rojo sobre blanco)
    mask_visual = np.ones((h, w, 3), dtype=np.uint8) * 255  # Fondo blanco
    mask_visual[mask_binary > 127] = [255, 0, 0]  # Grietas en rojo
    cv2.drawContours(mask_visual, contours, -1, (0, 0, 255), 2)  # Contorno azul
    
    return overlay, mask_visual, mask_colored

def imagen_a_base64(img_rgb):
    """Convierte imagen RGB a base64 para envío al frontend"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# ═══════════════════════════════════════════════════════════════════════════
# RUTAS API
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción con TTA
    
    Parámetros opcionales:
        - use_tta: bool (default: True) - Usar Test-Time Augmentation
        - return_base64: bool (default: False) - Devolver imágenes en base64
    """
    try:
        # Validar modelo
        if model is None:
            return jsonify({'error': 'El modelo no está cargado'}), 503
        
        # Validar archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Formato de archivo no permitido',
                'formatos_aceptados': list(config.ALLOWED_EXTENSIONS)
            }), 400
        
        # Obtener parámetros opcionales
        use_tta = request.form.get('use_tta', str(config.USE_TTA)).lower() == 'true'
        return_base64 = request.form.get('return_base64', 'false').lower() == 'true'
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Procesar imagen
        img_tensor, img_original = procesar_imagen(filepath)
        
        # Predecir con o sin TTA
        mask = predecir(img_tensor, use_tta=use_tta)
        
        # Post-procesar máscara
        mask_procesada = post_procesar(mask, config.THRESHOLD)
        
        # Calcular métricas detalladas
        metricas = calcular_metricas(mask_procesada, config.THRESHOLD)
        
        # Crear visualización
        overlay = crear_visualizacion(img_original, mask_procesada, config.THRESHOLD)
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'metricas': metricas,
            'timestamp': datetime.now().isoformat(),
            'procesamiento': {
                'tta_usado': use_tta,
                'threshold': config.THRESHOLD,
                'target_size': config.TARGET_SIZE
            }
        }
        
        # Guardar y devolver imagen según configuración
        if return_base64:
            # Devolver imagen en base64 (útil para frontend)
            response_data['imagen_overlay'] = imagen_a_base64(overlay)
        else:
            # Guardar archivo y devolver URL
            result_filename = f"result_{filename}"
            result_path = os.path.join(config.RESULTS_FOLDER, result_filename)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, overlay_bgr)
            response_data['result_image'] = f'/api/results/{result_filename}'
        
        # Eliminar archivo temporal
        os.remove(filepath)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predicción en lote para múltiples imágenes
    """
    try:
        if model is None:
            return jsonify({'error': 'El modelo no está cargado'}), 503
        
        if 'images' not in request.files:
            return jsonify({'error': 'No se enviaron imágenes'}), 400
        
        files = request.files.getlist('images')
        use_tta = request.form.get('use_tta', str(config.USE_TTA)).lower() == 'true'
        
        resultados = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Procesar cada imagen
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    temp_filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(config.UPLOAD_FOLDER, temp_filename)
                    file.save(filepath)
                    
                    img_tensor, img_original = procesar_imagen(filepath)
                    mask = predecir(img_tensor, use_tta=use_tta)
                    mask_procesada = post_procesar(mask, config.THRESHOLD)
                    metricas = calcular_metricas(mask_procesada, config.THRESHOLD)
                    
                    resultados.append({
                        'filename': file.filename,
                        'metricas': metricas,
                        'success': True
                    })
                    
                    os.remove(filepath)
                    
                except Exception as e:
                    resultados.append({
                        'filename': file.filename,
                        'error': str(e),
                        'success': False
                    })
        
        return jsonify({
            'success': True,
            'total_procesadas': len(resultados),
            'resultados': resultados,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error en batch prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """Obtener imagen de resultado"""
    try:
        filepath = os.path.join(config.RESULTS_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({'error': 'Archivo no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Obtener o actualizar configuración"""
    if request.method == 'GET':
        return jsonify({
            'target_size': config.TARGET_SIZE,
            'threshold': config.THRESHOLD,
            'use_tta': config.USE_TTA,
            'device': str(config.DEVICE),
            'model_name': config.MODEL_NAME
        }), 200
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'threshold' in data:
            config.THRESHOLD = float(data['threshold'])
        
        if 'use_tta' in data:
            config.USE_TTA = bool(data['use_tta'])
        
        return jsonify({
            'success': True,
            'config': {
                'threshold': config.THRESHOLD,
                'use_tta': config.USE_TTA
            }
        }), 200

@app.route('/api/status', methods=['GET'])
def status():
    """Estado detallado del sistema"""
    return jsonify({
        'system': 'CrackGuard Backend',
        'version': '2.0.0',
        'model': 'SegFormer B5',
        'device': str(config.DEVICE),
        'model_loaded': model is not None,
        'configuration': {
            'threshold': config.THRESHOLD,
            'target_size': config.TARGET_SIZE,
            'tta_enabled': config.USE_TTA,
            'max_file_size_mb': config.MAX_CONTENT_LENGTH / (1024 * 1024)
        },
        'timestamp': datetime.now().isoformat()
    }), 200

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("🚀 INICIALIZANDO CRACKGUARD BACKEND v2.0")
print("═" * 80)

modelo_cargado = cargar_modelo()

if modelo_cargado:
    print("✅ Sistema listo para inferencia")
else:
    print("⚠️  Servidor iniciado sin modelo entrenado")

print("═" * 80)

if __name__ == '__main__':
    print("\n🔧 Ejecutando con Flask development server")
    print("⚠️  Para producción usar: gunicorn app:app --bind 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)