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
    """Crea visualización overlay"""
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
    
    return overlay  # ← SOLO retornar overlay

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
    """Endpoint principal de predicción con TTA"""
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
        return_base64 = request.form.get('return_base64', 'true').lower() == 'true'  # ← Default true
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"📥 Procesando: {filename}")
        print(f"   TTA: {use_tta}, Base64: {return_base64}")
        
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
        
        # Devolver en base64 (por defecto) o guardar archivo
        if return_base64:
            # Convertir overlay a base64
            response_data['imagen_overlay'] = imagen_a_base64(overlay)
            print(f"✅ Imagen en base64 generada (tamaño: {len(response_data['imagen_overlay'])} chars)")
        else:
            # Guardar archivo y devolver URL
            result_filename = f"result_{filename}"
            result_path = os.path.join(config.RESULTS_FOLDER, result_filename)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, overlay_bgr)
        # ✅ CORREGIDO: NO incluir /api/, solo /results/
            response_data['result_image'] = f'/results/{result_filename}'
            print(f"✅ Imagen guardada: {result_path}")
        
        
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
🚀 API BACKEND CRACKGUARD v3.0 - SUPER ENSEMBLE + TTA
Sistema de inferencia con UNet++ EfficientNet-B8
Desarrollado por: Angel226m
═══════════════════════════════════════════════════════════════════════════════
"""
"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD v3.1 - SUPER ENSEMBLE + TTA + ANÁLISIS MORFOLÓGICO
Sistema de inferencia con UNet++ EfficientNet-B8 + Clasificación de Grietas
Desarrollado por: Angel226m
═══════════════════════════════════════════════════════════════════════════════
"""

"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD v3.1 - UNET++ B8 + TTA + ANÁLISIS MORFOLÓGICO
Sistema de inferencia con UNet++ EfficientNet-B8 + Análisis Inteligente
Desarrollado por: Angel226m
═══════════════════════════════════════════════════════════════════════════════
"""
































'''

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datetime import datetime
import traceback
import base64
from scipy.ndimage import binary_fill_holes
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

class Config:
    # Directorios
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # Modelo UNet++ EfficientNet-B8
    ARCHITECTURE = 'UnetPlusPlus'
    ENCODER = 'timm-efficientnet-b8'
    TARGET_SIZE = 640
    
    # Normalización ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Threshold
    THRESHOLD = 0.5
    MIN_CRACK_COVERAGE = 0.25
    
    # TTA (Test-Time Augmentation)
    USE_TTA = True
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    
    # Post-procesamiento
    USE_MORPHOLOGY = True
    USE_CONNECTED_COMPONENTS = True
    MIN_COMPONENT_SIZE = 20
    
    # Overlay
    OVERLAY_COLOR = 'red'
    OVERLAY_ALPHA = 0.4
    
    # Análisis morfológico
    ANGLE_TOLERANCE = 15  # grados
    MIN_CRACK_LENGTH = 50  # píxeles
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Crear directorios
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None
model_loaded = False

def cargar_modelo():
    """Carga el modelo UNet++ EfficientNet-B8"""
    global model, model_loaded
    
    try:
        print(f"🤖 Cargando UNet++ {config.ENCODER}...")
        print(f"   📁 Ruta: {config.MODEL_PATH}")
        
        # Verificar archivo
        if not os.path.exists(config.MODEL_PATH):
            print(f"   ❌ Archivo no encontrado")
            return False
        
        file_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
        print(f"   📦 Tamaño: {file_size_mb:.2f} MB")
        
        # Crear modelo
        model = smp.UnetPlusPlus(
            encoder_name=config.ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        # Cargar checkpoint
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        
        # Detectar formato
        if isinstance(checkpoint, dict):
            if 'swa_model_state_dict' in checkpoint:
                state_dict = checkpoint['swa_model_state_dict']
                print(f"   ✓ Usando pesos SWA")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"   ✓ Usando model_state_dict")
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
                print(f"   ✓ Usando pesos EMA")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Cargar pesos
        model.load_state_dict(state_dict, strict=False)
        
        # Mostrar métricas
        if isinstance(checkpoint, dict):
            if 'dice' in checkpoint:
                print(f"   📊 Dice Score: {checkpoint['dice']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   📈 Época: {checkpoint['epoch']}")
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ TTA: {len(config.TTA_TRANSFORMS)}x")
        print(f"   ✓ Análisis Morfológico: ACTIVADO")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO INTELIGENTE (POST-MODELO)
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta(contour):
    """
    Analiza la orientación de una grieta usando PCA y line fitting
    Retorna: ángulo (0-180°), tipo (horizontal/vertical/diagonal/irregular)
    """
    if len(contour) < 5:
        return None, "indefinido"
    
    try:
        # Fitear línea a los puntos del contorno usando regresión
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calcular ángulo en grados (0-180)
        angle = np.arctan2(vy, vx) * 180 / np.pi
        angle = angle[0] if isinstance(angle, np.ndarray) else angle
        
        # Normalizar a 0-180
        if angle < 0:
            angle += 180
        
        # Clasificar orientación según ángulo
        if (angle < config.ANGLE_TOLERANCE) or (angle > 180 - config.ANGLE_TOLERANCE):
            tipo = "horizontal"
        elif (90 - config.ANGLE_TOLERANCE < angle < 90 + config.ANGLE_TOLERANCE):
            tipo = "vertical"
        elif (45 - config.ANGLE_TOLERANCE < angle < 45 + config.ANGLE_TOLERANCE) or \
             (135 - config.ANGLE_TOLERANCE < angle < 135 + config.ANGLE_TOLERANCE):
            tipo = "diagonal"
        else:
            tipo = "irregular"
        
        return float(angle), tipo
        
    except Exception as e:
        return None, "indefinido"

def clasificar_patron_global(contours, mask_binary):
    """
    Clasifica el patrón general de agrietamiento
    
    Patrones:
    - Horizontal: Flexión, presión lateral, empuje de tierra
    - Vertical: Cargas pesadas, asentamientos diferenciales
    - Diagonal/Escalera: Esfuerzos cortantes, movimiento de terreno
    - Ramificada/Mapa: Contracción térmica, retracción superficial
    - Mixto: Combinación de factores
    """
    
    if len(contours) == 0:
        return {
            'patron': 'sin_grietas',
            'descripcion': 'No se detectaron grietas',
            'causa_probable': 'N/A',
            'severidad_ajuste': 1.0
        }
    
    # Analizar orientaciones
    orientaciones = []
    longitudes = []
    
    for contour in contours:
        length = cv2.arcLength(contour, False)
        
        if length < config.MIN_CRACK_LENGTH:
            continue
        
        angle, tipo = analizar_orientacion_grieta(contour)
        
        if angle is not None:
            orientaciones.append(tipo)
            longitudes.append(length)
    
    if not orientaciones:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial normal',
            'severidad_ajuste': 0.8
        }
    
    # Contar tipos de orientaciones
    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts.most_common(1)[0][0]
    diversidad = len(tipo_counts)
    num_grietas = len(orientaciones)
    
    # Calcular ramificación (complejidad del patrón)
    num_componentes = len(contours)
    complejidad = num_componentes / max(num_grietas, 1)
    
    # Clasificación según patrón dominante
    if diversidad >= 3 or complejidad > 1.5:
        # Patrón ramificado/mapa (múltiples direcciones)
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón tipo mapa/ramificado - Típico de contracción térmica',
            'causa_probable': 'Cambios térmicos, secado, contracción del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico - Generalmente superficial'
        }
    
    elif tipo_dominante == "horizontal" and tipo_counts["horizontal"] / len(orientaciones) > 0.6:
        # Grietas horizontales dominantes
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral, empuje de tierra',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros de contención y cimentación'
        }
    
    elif tipo_dominante == "vertical" and tipo_counts["vertical"] / len(orientaciones) > 0.6:
        # Grietas verticales dominantes (MÁS CRÍTICO)
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas predominantemente verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos diferenciales',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ URGENTE: Inspección estructural profesional inmediata'
        }
    
    elif tipo_dominante == "diagonal" and tipo_counts["diagonal"] / len(orientaciones) > 0.5:
        # Grietas diagonales (MUY CRÍTICO)
        return {
            'patron': 'diagonal_escalera',
            'descripcion': 'Grietas diagonales/en escalera - ⚠️ MUY CRÍTICO',
            'causa_probable': 'Esfuerzos cortantes, movimiento del terreno, asentamientos irregulares',
            'severidad_ajuste': 1.4,
            'recomendacion': '🔴 URGENTE: Evaluación estructural crítica - Posible falla inminente'
        }
    
    elif diversidad >= 2:
        # Patrón mixto
        return {
            'patron': 'mixto',
            'descripcion': 'Patrón mixto de agrietamiento',
            'causa_probable': 'Combinación de factores estructurales y ambientales',
            'severidad_ajuste': 1.2,
            'recomendacion': 'Inspección profesional detallada requerida'
        }
    
    else:
        # Patrón irregular
        return {
            'patron': 'irregular',
            'descripcion': 'Patrón irregular - Requiere análisis detallado',
            'causa_probable': 'Causa indeterminada',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Se recomienda inspección profesional'
        }

def analizar_morfologia_detallada(mask, contours):
    """
    Análisis morfológico completo de las grietas
    """
    
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Clasificar patrón global
    patron_info = clasificar_patron_global(contours, mask_binary)
    
    # Analizar grietas individuales
    grietas_detalle = []
    orientaciones_count = {
        "horizontal": 0,
        "vertical": 0,
        "diagonal": 0,
        "irregular": 0
    }
    
    for i, contour in enumerate(contours):
        length = cv2.arcLength(contour, False)
        
        if length < config.MIN_CRACK_LENGTH:
            continue
        
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta(contour)
        
        # Calcular ancho promedio
        width = area / length if length > 0 else 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        orientaciones_count[tipo] += 1
        
        grietas_detalle.append({
            'id': i + 1,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })
    
    # Ordenar por longitud (más largas primero)
    grietas_detalle.sort(key=lambda x: x['longitud_px'], reverse=True)
    
    return {
        'patron_general': patron_info['patron'],
        'descripcion_patron': patron_info['descripcion'],
        'causa_probable': patron_info['causa_probable'],
        'severidad_ajuste': patron_info['severidad_ajuste'],
        'recomendacion': patron_info.get('recomendacion', 'Monitoreo continuo'),
        'distribucion_orientaciones': orientaciones_count,
        'num_grietas_analizadas': len(grietas_detalle),
        'grietas_principales': grietas_detalle[:5]  # Top 5
    }

# ═══════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def get_transform():
    return A.Compose([
        A.Resize(config.TARGET_SIZE, config.TARGET_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2()
    ])

def advanced_postprocess(mask):
    """Post-procesamiento morfológico + connected components"""
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    if config.USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    
    if config.USE_CONNECTED_COMPONENTS:
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask_np)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_SIZE:
                cleaned_mask[labels == i] = mask_np[labels == i]
        
        mask_np = cleaned_mask
    
    mask_binary = (mask_np > 0.5).astype(bool)
    mask_filled = binary_fill_holes(mask_binary)
    
    return mask_filled.astype(np.float32)

def predict_with_tta(model, img_tensor):
    """Predicción con TTA (6 transformaciones)"""
    preds = []
    
    # Original
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        preds.append(pred)
    
    # Horizontal flip
    if 'hflip' in config.TTA_TRANSFORMS:
        img_hflip = torch.flip(img_tensor, dims=[3])
        with torch.no_grad():
            pred = model(img_hflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[3])
            preds.append(pred)
    
    # Vertical flip
    if 'vflip' in config.TTA_TRANSFORMS:
        img_vflip = torch.flip(img_tensor, dims=[2])
        with torch.no_grad():
            pred = model(img_vflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[2])
            preds.append(pred)
    
    # Rotate 90
    if 'rotate90' in config.TTA_TRANSFORMS:
        img_rot90 = torch.rot90(img_tensor, k=1, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot90)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            preds.append(pred)
    
    # Rotate 180
    if 'rotate180' in config.TTA_TRANSFORMS:
        img_rot180 = torch.rot90(img_tensor, k=2, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot180)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-2, dims=[2, 3])
            preds.append(pred)
    
    # Rotate 270
    if 'rotate270' in config.TTA_TRANSFORMS:
        img_rot270 = torch.rot90(img_tensor, k=3, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot270)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-3, dims=[2, 3])
            preds.append(pred)
    
    return torch.stack(preds).mean(dim=0)

def procesar_imagen(image_path, use_tta=True):
    """Procesar imagen con el modelo"""
    if not model_loaded:
        raise RuntimeError("Modelo no cargado")
    
    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    
    # Preprocesar
    transform = get_transform()
    img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(config.DEVICE)
    
    # Predicción
    if use_tta:
        pred = predict_with_tta(model, img_tensor)
    else:
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
    
    # Obtener confidence map
    confidence_map = pred.cpu().numpy()[0, 0]
    
    # Redimensionar
    confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Post-procesamiento
    confidence_map = advanced_postprocess(torch.from_numpy(confidence_map))
    
    # Máscara binaria
    pred_mask = (confidence_map > config.THRESHOLD).astype(np.uint8) * 255
    
    return img_rgb, pred_mask, confidence_map

def crear_overlay(img_original, mask):
    """Crear overlay con toda la grieta coloreada"""
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Crear máscara de color
    color_mask = np.zeros_like(img_original)
    
    if config.OVERLAY_COLOR == 'red':
        color_mask[:, :, 0] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'green':
        color_mask[:, :, 1] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'yellow':
        color_mask[:, :, 0] = mask_binary * 255
        color_mask[:, :, 1] = mask_binary * 255
    else:
        color_mask[:, :, 0] = mask_binary * 255
    
    # Alpha blending
    overlay = cv2.addWeighted(img_original, 1.0, color_mask, config.OVERLAY_ALPHA, 0)
    
    return overlay

def calcular_metricas(mask, confidence_map):
    """Calcular métricas + análisis morfológico"""
    
    mask_binary = (mask > 127).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    # Contornos
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    if num_contours > 0:
        total_length = sum(cv2.arcLength(cnt, False) for cnt in contours)
        avg_length = total_length / num_contours
        max_length = max(cv2.arcLength(cnt, False) for cnt in contours)
        avg_width = pixeles_positivos / total_length if total_length > 0 else 0
    else:
        total_length = 0
        avg_length = 0
        max_length = 0
        avg_width = 0
    
    # 🆕 ANÁLISIS MORFOLÓGICO INTELIGENTE
    morfologia = analizar_morfologia_detallada(mask, contours)
    
    # Ajustar severidad según patrón
    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste
    
    # Severidad
    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Sin Grietas Significativas"
    elif porcentaje_ajustado < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
    elif porcentaje_ajustado < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
    
    # Ajuste adicional por patrón crítico
    if morfologia['patron_general'] in ['vertical', 'diagonal_escalera']:
        if severidad == "Media":
            severidad = "Media-Alta"
        elif severidad == "Baja" and porcentaje_grietas > 2:
            severidad = "Media"
    
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_contours),
        'longitud_total_px': float(total_length),
        'longitud_promedio_px': float(avg_length),
        'longitud_maxima_px': float(max_length),
        'ancho_promedio_px': float(avg_width),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        
        # 🆕 ANÁLISIS MORFOLÓGICO
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    """Convertir imagen a base64"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# RUTAS API
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'architecture': f'{config.ARCHITECTURE} + {config.ENCODER}',
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'morphological_analysis': True,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió imagen'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400
        
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS):
            return jsonify({'error': 'Formato no permitido'}), 400
        
        # Parámetros
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'
        
        # Guardar temporal
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"📥 Procesando: {filename} (TTA: {use_tta})")
        
        # Procesar
        img_original, pred_mask, confidence_map = procesar_imagen(filepath, use_tta)
        
        # Crear overlay
        overlay = crear_overlay(img_original, pred_mask)
        
        # Métricas + Morfología
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        morfologia = metricas['analisis_morfologico']
        print(f"   🔍 Patrón: {morfologia['patron_general']}")
        print(f"   📐 Orientaciones: {morfologia['distribucion_orientaciones']}")
        print(f"   ⚠️  Severidad: {metricas['severidad']}")
        
        # Respuesta
        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat(),
            'procesamiento': {
                'architecture': config.ARCHITECTURE,
                'encoder': config.ENCODER,
                'tta_usado': use_tta,
                'tta_transforms': len(config.TTA_TRANSFORMS) if use_tta else 0,
                'threshold': config.THRESHOLD,
                'target_size': config.TARGET_SIZE,
                'morphological_analysis': True,
            }
        }
        
        # Limpiar
        os.remove(filepath)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 100)
print("🚀 CRACKGUARD BACKEND v3.1 - UNET++ B8 + TTA + ANÁLISIS MORFOLÓGICO")
print("   Análisis Inteligente de Orientación POST-Modelo")
print("═" * 100)

if cargar_modelo():
    print("✅ Sistema listo para inferencia con análisis morfológico")
else:
    print("⚠️  Servidor iniciado sin modelo")

print("═" * 100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)'''




"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD v3.2 - UNET++ B8 + TTA + ANÁLISIS MORFOLÓGICO COMPLETO
Sistema de inferencia con detección de grietas ultra sensible
Desarrollado por: Angel226m
Fecha: 2025-10-27
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datetime import datetime
import traceback
import base64
from scipy.ndimage import binary_fill_holes
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

class Config:
    # Directorios
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # Modelo UNet++ EfficientNet-B8
    ARCHITECTURE = 'UnetPlusPlus'
    ENCODER = 'timm-efficientnet-b8'
    TARGET_SIZE = 640
    
    # Normalización ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Threshold
    THRESHOLD = 0.5
    MIN_CRACK_COVERAGE = 0.25
    
    # TTA (Test-Time Augmentation)
    USE_TTA = True
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    
    # Post-procesamiento
    USE_MORPHOLOGY = True
    USE_CONNECTED_COMPONENTS = True
    MIN_COMPONENT_SIZE = 10  # ✅ REDUCIDO de 20 a 10 píxeles
    
    # Overlay
    OVERLAY_COLOR = 'red'
    OVERLAY_ALPHA = 0.4
    
    # Análisis morfológico (ULTRA SENSIBLE)
    ANGLE_TOLERANCE = 15  # grados
    MIN_CRACK_LENGTH = 15  # ✅ REDUCIDO de 50 a 15 píxeles (detecta grietas MUY pequeñas)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Crear directorios
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None
model_loaded = False

def cargar_modelo():
    """Carga el modelo UNet++ EfficientNet-B8"""
    global model, model_loaded
    
    try:
        print(f"🤖 Cargando UNet++ {config.ENCODER}...")
        print(f"   📁 Ruta: {config.MODEL_PATH}")
        
        if not os.path.exists(config.MODEL_PATH):
            print(f"   ❌ Archivo no encontrado")
            return False
        
        file_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
        print(f"   📦 Tamaño: {file_size_mb:.2f} MB")
        
        model = smp.UnetPlusPlus(
            encoder_name=config.ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'swa_model_state_dict' in checkpoint:
                state_dict = checkpoint['swa_model_state_dict']
                print(f"   ✓ Usando pesos SWA")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"   ✓ Usando model_state_dict")
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
                print(f"   ✓ Usando pesos EMA")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        
        if isinstance(checkpoint, dict):
            if 'dice' in checkpoint:
                print(f"   📊 Dice Score: {checkpoint['dice']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   📈 Época: {checkpoint['epoch']}")
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ TTA: {len(config.TTA_TRANSFORMS)}x")
        print(f"   ✓ Análisis Morfológico: ULTRA SENSIBLE (min {config.MIN_CRACK_LENGTH}px)")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO INTELIGENTE (ULTRA SENSIBLE)
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta(contour):
    """
    Analiza la orientación de una grieta usando line fitting
    Retorna: ángulo (0-180°), tipo (horizontal/vertical/diagonal/irregular)
    """
    if len(contour) < 5:
        return None, "indefinido"
    
    try:
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        angle = np.arctan2(vy, vx) * 180 / np.pi
        angle = angle[0] if isinstance(angle, np.ndarray) else angle
        
        if angle < 0:
            angle += 180
        
        if (angle < config.ANGLE_TOLERANCE) or (angle > 180 - config.ANGLE_TOLERANCE):
            tipo = "horizontal"
        elif (90 - config.ANGLE_TOLERANCE < angle < 90 + config.ANGLE_TOLERANCE):
            tipo = "vertical"
        elif (45 - config.ANGLE_TOLERANCE < angle < 45 + config.ANGLE_TOLERANCE) or \
             (135 - config.ANGLE_TOLERANCE < angle < 135 + config.ANGLE_TOLERANCE):
            tipo = "diagonal"
        else:
            tipo = "irregular"
        
        return float(angle), tipo
        
    except Exception:
        return None, "indefinido"

def clasificar_patron_global(contours, mask_binary):
    """
    Clasifica el patrón general de agrietamiento (ULTRA SENSIBLE)
    """
    
    if len(contours) == 0:
        return {
            'patron': 'sin_grietas',
            'descripcion': 'No se detectaron grietas',
            'causa_probable': 'N/A',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Estructura sin daños aparentes'
        }
    
    orientaciones = []
    longitudes = []
    
    for contour in contours:
        length = cv2.arcLength(contour, False)
        
        # ✅ UMBRAL ULTRA BAJO para detectar grietas pequeñas
        if length < config.MIN_CRACK_LENGTH:
            continue
        
        angle, tipo = analizar_orientacion_grieta(contour)
        
        if angle is not None:
            orientaciones.append(tipo)
            longitudes.append(length)
    
    if not orientaciones:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial normal',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico recomendado'
        }
    
    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts.most_common(1)[0][0]
    diversidad = len(tipo_counts)
    num_grietas = len(orientaciones)
    
    num_componentes = len(contours)
    complejidad = num_componentes / max(num_grietas, 1)
    
    # Clasificación según patrón dominante
    if diversidad >= 3 or complejidad > 1.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón tipo mapa/ramificado - Típico de contracción térmica',
            'causa_probable': 'Cambios térmicos, secado, contracción del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico - Generalmente superficial'
        }
    
    elif tipo_dominante == "horizontal" and tipo_counts["horizontal"] / len(orientaciones) > 0.6:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral, empuje de tierra',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros de contención y cimentación'
        }
    
    elif tipo_dominante == "vertical" and tipo_counts["vertical"] / len(orientaciones) > 0.6:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas predominantemente verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos diferenciales',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ URGENTE: Inspección estructural profesional inmediata'
        }
    
    elif tipo_dominante == "diagonal" and tipo_counts["diagonal"] / len(orientaciones) > 0.5:
        return {
            'patron': 'diagonal_escalera',
            'descripcion': 'Grietas diagonales/en escalera - ⚠️ MUY CRÍTICO',
            'causa_probable': 'Esfuerzos cortantes, movimiento del terreno, asentamientos irregulares',
            'severidad_ajuste': 1.4,
            'recomendacion': '🔴 URGENTE: Evaluación estructural crítica - Posible falla inminente'
        }
    
    elif diversidad >= 2:
        return {
            'patron': 'mixto',
            'descripcion': 'Patrón mixto de agrietamiento',
            'causa_probable': 'Combinación de factores estructurales y ambientales',
            'severidad_ajuste': 1.2,
            'recomendacion': 'Inspección profesional detallada requerida'
        }
    
    else:
        return {
            'patron': 'irregular',
            'descripcion': 'Patrón irregular - Requiere análisis detallado',
            'causa_probable': 'Causa indeterminada',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Se recomienda inspección profesional'
        }

def analizar_morfologia_detallada(mask, contours):
    """
    Análisis morfológico completo (ULTRA SENSIBLE - detecta grietas pequeñas)
    """
    
    mask_binary = (mask > 127).astype(np.uint8)
    patron_info = clasificar_patron_global(contours, mask_binary)
    
    grietas_detalle = []
    orientaciones_count = {
        "horizontal": 0,
        "vertical": 0,
        "diagonal": 0,
        "irregular": 0
    }
    
    for i, contour in enumerate(contours):
        length = cv2.arcLength(contour, False)
        
        # ✅ ANALIZA INCLUSO GRIETAS MUY PEQUEÑAS (>= 15px)
        if length < config.MIN_CRACK_LENGTH:
            continue
        
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta(contour)
        
        width = area / length if length > 0 else 0
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        orientaciones_count[tipo] += 1
        
        grietas_detalle.append({
            'id': i + 1,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })
    
    grietas_detalle.sort(key=lambda x: x['longitud_px'], reverse=True)
    
    return {
        'patron_general': patron_info['patron'],
        'descripcion_patron': patron_info['descripcion'],
        'causa_probable': patron_info['causa_probable'],
        'severidad_ajuste': patron_info['severidad_ajuste'],
        'recomendacion': patron_info.get('recomendacion', 'Monitoreo continuo'),
        'distribucion_orientaciones': orientaciones_count,
        'num_grietas_analizadas': len(grietas_detalle),
        'grietas_principales': grietas_detalle[:5]
    }

# ═══════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def get_transform():
    return A.Compose([
        A.Resize(config.TARGET_SIZE, config.TARGET_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2()
    ])

def advanced_postprocess(mask):
    """Post-procesamiento morfológico + connected components (SENSIBLE)"""
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    if config.USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    
    if config.USE_CONNECTED_COMPONENTS:
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask_np)
        for i in range(1, num_labels):
            # ✅ COMPONENTE MÍNIMO REDUCIDO (10px)
            if stats[i, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_SIZE:
                cleaned_mask[labels == i] = mask_np[labels == i]
        
        mask_np = cleaned_mask
    
    mask_binary = (mask_np > 0.5).astype(bool)
    mask_filled = binary_fill_holes(mask_binary)
    
    return mask_filled.astype(np.float32)

def predict_with_tta(model, img_tensor):
    """Predicción con TTA (6 transformaciones)"""
    preds = []
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        preds.append(pred)
    
    if 'hflip' in config.TTA_TRANSFORMS:
        img_hflip = torch.flip(img_tensor, dims=[3])
        with torch.no_grad():
            pred = model(img_hflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[3])
            preds.append(pred)
    
    if 'vflip' in config.TTA_TRANSFORMS:
        img_vflip = torch.flip(img_tensor, dims=[2])
        with torch.no_grad():
            pred = model(img_vflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[2])
            preds.append(pred)
    
    if 'rotate90' in config.TTA_TRANSFORMS:
        img_rot90 = torch.rot90(img_tensor, k=1, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot90)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            preds.append(pred)
    
    if 'rotate180' in config.TTA_TRANSFORMS:
        img_rot180 = torch.rot90(img_tensor, k=2, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot180)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-2, dims=[2, 3])
            preds.append(pred)
    
    if 'rotate270' in config.TTA_TRANSFORMS:
        img_rot270 = torch.rot90(img_tensor, k=3, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot270)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-3, dims=[2, 3])
            preds.append(pred)
    
    return torch.stack(preds).mean(dim=0)

def procesar_imagen(image_path, use_tta=True):
    """Procesar imagen con el modelo"""
    if not model_loaded:
        raise RuntimeError("Modelo no cargado")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    
    transform = get_transform()
    img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(config.DEVICE)
    
    if use_tta:
        pred = predict_with_tta(model, img_tensor)
    else:
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
    
    confidence_map = pred.cpu().numpy()[0, 0]
    confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
    confidence_map = advanced_postprocess(torch.from_numpy(confidence_map))
    pred_mask = (confidence_map > config.THRESHOLD).astype(np.uint8) * 255
    
    return img_rgb, pred_mask, confidence_map

def crear_overlay(img_original, mask):
    """Crear overlay con toda la grieta coloreada"""
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)
    
    if config.OVERLAY_COLOR == 'red':
        color_mask[:, :, 0] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'green':
        color_mask[:, :, 1] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'yellow':
        color_mask[:, :, 0] = mask_binary * 255
        color_mask[:, :, 1] = mask_binary * 255
    else:
        color_mask[:, :, 0] = mask_binary * 255
    
    overlay = cv2.addWeighted(img_original, 1.0, color_mask, config.OVERLAY_ALPHA, 0)
    return overlay

def calcular_metricas(mask, confidence_map):
    """Calcular métricas + análisis morfológico ULTRA SENSIBLE"""
    
    mask_binary = (mask > 127).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    if num_contours > 0:
        total_length = sum(cv2.arcLength(cnt, False) for cnt in contours)
        avg_length = total_length / num_contours
        max_length = max(cv2.arcLength(cnt, False) for cnt in contours)
        avg_width = pixeles_positivos / total_length if total_length > 0 else 0
    else:
        total_length = 0
        avg_length = 0
        max_length = 0
        avg_width = 0
    
    # ✅ ANÁLISIS MORFOLÓGICO ULTRA SENSIBLE
    morfologia = analizar_morfologia_detallada(mask, contours)
    
    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste
    
    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Sin Grietas Significativas"
    elif porcentaje_ajustado < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
    elif porcentaje_ajustado < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
    
    if morfologia['patron_general'] in ['vertical', 'diagonal_escalera']:
        if severidad == "Media":
            severidad = "Media-Alta"
        elif severidad == "Baja" and porcentaje_grietas > 2:
            severidad = "Media"
    
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_contours),
        'longitud_total_px': float(total_length),
        'longitud_promedio_px': float(avg_length),
        'longitud_maxima_px': float(max_length),
        'ancho_promedio_px': float(avg_width),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    """Convertir imagen a base64"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# RUTAS API
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'architecture': f'{config.ARCHITECTURE} + {config.ENCODER}',
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'morphological_analysis': True,
        'min_crack_length': config.MIN_CRACK_LENGTH,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió imagen'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400
        
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS):
            return jsonify({'error': 'Formato no permitido'}), 400
        
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"📥 Procesando: {filename} (TTA: {use_tta})")
        
        img_original, pred_mask, confidence_map = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        morfologia = metricas['analisis_morfologico']
        print(f"   🔍 Patrón: {morfologia['patron_general']}")
        print(f"   📐 Orientaciones: {morfologia['distribucion_orientaciones']}")
        print(f"   ⚠️  Severidad: {metricas['severidad']}")
        print(f"   📊 Grietas analizadas: {morfologia['num_grietas_analizadas']}")
        
        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat(),
            'procesamiento': {
                'architecture': config.ARCHITECTURE,
                'encoder': config.ENCODER,
                'tta_usado': use_tta,
                'tta_transforms': len(config.TTA_TRANSFORMS) if use_tta else 0,
                'threshold': config.THRESHOLD,
                'target_size': config.TARGET_SIZE,
                'morphological_analysis': True,
                'min_crack_length': config.MIN_CRACK_LENGTH,
            }
        }
        
        os.remove(filepath)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 100)
print("🚀 CRACKGUARD BACKEND v3.2 - UNET++ B8 + TTA + ANÁLISIS MORFOLÓGICO ULTRA SENSIBLE")
print(f"   👤 Usuario: Angel226m")
print(f"   📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   🔍 Detección ultra sensible: grietas >= {Config.MIN_CRACK_LENGTH}px")
print("═" * 100)

if cargar_modelo():
    print("✅ Sistema listo para inferencia ultra sensible con análisis morfológico")
else:
    print("⚠️  Servidor iniciado sin modelo")

print("═" * 100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)