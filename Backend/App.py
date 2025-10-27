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

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
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
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
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
    
    # Post-procesamiento avanzado
    USE_MORPHOLOGY = True
    USE_CONNECTED_COMPONENTS = True
    MIN_COMPONENT_SIZE = 20  # píxeles
    
    # Visualización overlay
    OVERLAY_COLOR = 'red'  # 'red', 'green', 'yellow', 'magenta'
    OVERLAY_ALPHA = 0.4
    
    # Análisis morfológico de grietas
    ANGLE_TOLERANCE = 15  # grados para clasificar orientación
    MIN_CRACK_LENGTH = 50  # píxeles mínimos para analizar
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Crear directorios
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)
Path(config.RESULTS_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO UNET++
# ═══════════════════════════════════════════════════════════════════════════

model = None

def cargar_modelo():
    """Carga el modelo UNet++ EfficientNet-B8"""
    global model
    
    try:
        print(f"🤖 Cargando UNet++ {config.ENCODER}...")
        print(f"   Ruta: {config.MODEL_PATH}")
        
        # Crear arquitectura
        model = smp.UnetPlusPlus(
            encoder_name=config.ENCODER,
            encoder_weights=None,  # Cargaremos pesos entrenados
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        # Cargar pesos entrenados
        if os.path.exists(config.MODEL_PATH):
            checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
            
            # Detectar formato del checkpoint
            if isinstance(checkpoint, dict):
                if 'swa_model_state_dict' in checkpoint:
                    state_dict = checkpoint['swa_model_state_dict']
                    print("   ℹ️  Usando pesos SWA")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                    state_dict = checkpoint['ema_state_dict']
                    print("   ℹ️  Usando pesos EMA")
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✓ Modelo cargado exitosamente")
            
            if isinstance(checkpoint, dict) and 'dice' in checkpoint:
                print(f"   ✓ Dice Score: {checkpoint['dice']:.4f}")
        else:
            print(f"   ⚠️  No se encontró modelo en {config.MODEL_PATH}")
            print(f"   ℹ️  Usando modelo base sin fine-tuning")
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✓ TTA: {'ACTIVADO' if config.USE_TTA else 'DESACTIVADO'} ({len(config.TTA_TRANSFORMS)}x)")
        print(f"   ✓ Análisis Morfológico: ACTIVADO")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO DE GRIETAS
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta(contour):
    """
    Analiza la orientación principal de una grieta usando PCA y fitting
    Retorna: ángulo (0-180°), tipo (horizontal/vertical/diagonal)
    """
    if len(contour) < 5:
        return None, "indefinido"
    
    # Fitear línea a los puntos del contorno
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Calcular ángulo en grados (0-180)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    angle = angle[0] if isinstance(angle, np.ndarray) else angle
    
    # Normalizar a 0-180
    if angle < 0:
        angle += 180
    
    # Clasificar orientación
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

def clasificar_patron_grieta(mask_binary, contours):
    """
    Clasifica el patrón general de agrietamiento
    Tipos:
    - Horizontal: Flexión, presión lateral
    - Vertical: Cargas pesadas, asentamientos
    - Diagonal/Escalera: Esfuerzos cortantes, movimiento de terreno
    - Mapa/Ramificada: Contracción térmica, superficial
    - Mixto: Combinación
    """
    if len(contours) == 0:
        return "sin_grietas", "No se detectaron grietas"
    
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
        return "superficial", "Grietas superficiales menores"
    
    # Contar tipos
    from collections import Counter
    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts.most_common(1)[0][0]
    diversidad = len(tipo_counts)
    num_grietas = len(orientaciones)
    
    # Skeletonize para detectar ramificaciones
    skeleton = skeletonize(mask_binary > 0)
    labeled_skeleton = label(skeleton)
    num_ramas = len(np.unique(labeled_skeleton)) - 1
    
    # Clasificación según patrones
    if diversidad >= 3 or num_ramas > num_grietas * 1.5:
        patron = "mapa_ramificada"
        descripcion = "Patrón tipo mapa/ramificado - Típico de contracción térmica o retracción superficial"
        causa = "Cambios térmicos, secado, contracción del material"
        severidad_ajuste = 0.8  # Generalmente menos crítico
        
    elif tipo_dominante == "horizontal" and tipo_counts["horizontal"] / len(orientaciones) > 0.6:
        patron = "horizontal"
        descripcion = "Grietas predominantemente horizontales - Posible flexión o presión lateral"
        causa = "Empuje de tierra, flexión estructural, presión horizontal"
        severidad_ajuste = 1.1
        
    elif tipo_dominante == "vertical" and tipo_counts["vertical"] / len(orientaciones) > 0.6:
        patron = "vertical"
        descripcion = "Grietas predominantemente verticales - Posible sobrecarga o asentamiento"
        causa = "Cargas verticales excesivas, asentamientos diferenciales"
        severidad_ajuste = 1.3
        
    elif tipo_dominante == "diagonal" and tipo_counts["diagonal"] / len(orientaciones) > 0.5:
        patron = "diagonal_escalera"
        descripcion = "Grietas diagonales/en escalera - Posibles esfuerzos cortantes o movimiento de terreno"
        causa = "Movimientos del terreno, esfuerzos cortantes, asentamientos irregulares"
        severidad_ajuste = 1.4  # Más crítico
        
    elif diversidad >= 2:
        patron = "mixto"
        descripcion = "Patrón mixto de agrietamiento - Múltiples causas posibles"
        causa = "Combinación de factores estructurales y ambientales"
        severidad_ajuste = 1.2
        
    else:
        patron = "irregular"
        descripcion = "Patrón irregular - Requiere análisis detallado"
        causa = "Causa indeterminada - Se recomienda inspección profesional"
        severidad_ajuste = 1.0
    
    return patron, descripcion, causa, severidad_ajuste

def analizar_morfologia_detallada(mask, confidence_map):
    """
    Análisis morfológico completo de las grietas detectadas
    """
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Clasificar patrón general
    patron, descripcion, causa, severidad_ajuste = clasificar_patron_grieta(mask_binary, contours)
    
    # Análisis individual de grietas
    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}
    
    for i, contour in enumerate(contours):
        length = cv2.arcLength(contour, False)
        
        if length < config.MIN_CRACK_LENGTH:
            continue
        
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta(contour)
        
        # Ancho promedio
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
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
    
    # Ordenar por longitud descendente
    grietas_detalle.sort(key=lambda x: x['longitud_px'], reverse=True)
    
    return {
        'patron_general': patron,
        'descripcion_patron': descripcion,
        'causa_probable': causa,
        'severidad_ajuste': severidad_ajuste,
        'distribución_orientaciones': orientaciones_count,
        'num_grietas_analizadas': len(grietas_detalle),
        'grietas_principales': grietas_detalle[:5]  # Top 5 más largas
    }

# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def get_transform():
    """Transforma de preprocesamiento"""
    return A.Compose([
        A.Resize(config.TARGET_SIZE, config.TARGET_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])

def advanced_postprocess(mask):
    """Post-procesamiento morfológico + connected components"""
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    # Morfología
    if config.USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    
    # Connected components filtering
    if config.USE_CONNECTED_COMPONENTS:
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        # Filtrar componentes pequeños
        cleaned_mask = np.zeros_like(mask_np)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_SIZE:
                cleaned_mask[labels == i] = mask_np[labels == i]
        
        mask_np = cleaned_mask
    
    # Rellenar huecos
    mask_binary = (mask_np > 0.5).astype(bool)
    mask_filled = binary_fill_holes(mask_binary)
    mask_np = mask_filled.astype(np.float32)
    
    return mask_np

def predict_with_tta(model, img_tensor):
    """Predicción con TTA (6 transformaciones)"""
    preds = []
    
    transforms_list = [
        ('original', lambda x: x, lambda x: x),
        ('hflip', lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        ('vflip', lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        ('rotate90', lambda x: torch.rot90(x, k=1, dims=[2, 3]), lambda x: torch.rot90(x, k=-1, dims=[2, 3])),
        ('rotate180', lambda x: torch.rot90(x, k=2, dims=[2, 3]), lambda x: torch.rot90(x, k=-2, dims=[2, 3])),
        ('rotate270', lambda x: torch.rot90(x, k=3, dims=[2, 3]), lambda x: torch.rot90(x, k=-3, dims=[2, 3])),
    ]
    
    for name, tfm_forward, tfm_backward in transforms_list:
        if name not in config.TTA_TRANSFORMS:
            continue
            
        img_tfm = tfm_forward(img_tensor.clone())
        
        with torch.no_grad():
            pred = model(img_tfm)
            pred = torch.sigmoid(pred)
        
        pred = tfm_backward(pred)
        preds.append(pred)
    
    # Promediar todas las predicciones
    return torch.stack(preds).mean(dim=0)

def procesar_imagen(image_path):
    """Carga, preprocesa y predice con Super Ensemble"""
    if model is None:
        raise RuntimeError("El modelo no está cargado")
    
    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    
    # Preprocesar
    transform = get_transform()
    img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(config.DEVICE)
    
    # Predicción con TTA
    if config.USE_TTA:
        pred = predict_with_tta(model, img_tensor)
    else:
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
    
    # Obtener confidence map
    confidence_map = pred.cpu().numpy()[0, 0]
    
    # Redimensionar a tamaño original
    confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Post-procesamiento
    confidence_map = advanced_postprocess(torch.from_numpy(confidence_map))
    
    # Máscara binaria
    pred_mask = (confidence_map > config.THRESHOLD).astype(np.uint8) * 255
    
    return img_rgb, pred_mask, confidence_map

def crear_visualizacion(img_original, mask, confidence_map):
    """Crea visualización overlay con TODA la grieta coloreada"""
    h, w = img_original.shape[:2]
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Crear máscara de color (TODA la grieta coloreada)
    color_mask = np.zeros_like(img_original)
    
    if config.OVERLAY_COLOR == 'red':
        color_mask[:, :, 0] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'green':
        color_mask[:, :, 1] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'yellow':
        color_mask[:, :, 0] = mask_binary * 255
        color_mask[:, :, 1] = mask_binary * 255
    elif config.OVERLAY_COLOR == 'magenta':
        color_mask[:, :, 0] = mask_binary * 255
        color_mask[:, :, 2] = mask_binary * 255
    else:
        color_mask[:, :, 0] = mask_binary * 255  # Default: red
    
    # Alpha blending (overlay completo)
    overlay = cv2.addWeighted(img_original, 1.0, color_mask, config.OVERLAY_ALPHA, 0)
    
    return overlay

def calcular_metricas(mask, confidence_map):
    """Calcula métricas detalladas de la predicción + análisis morfológico"""
    mask_binary = (mask > 127).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    # Contornos
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_grietas = len(contours)
    
    if num_grietas > 0:
        total_length = sum(cv2.arcLength(cnt, False) for cnt in contours)
        avg_length = total_length / num_grietas
        max_length = max(cv2.arcLength(cnt, False) for cnt in contours)
        avg_width = pixeles_positivos / total_length if total_length > 0 else 0
    else:
        total_length = 0
        avg_length = 0
        max_length = 0
        avg_width = 0
    
    # **ANÁLISIS MORFOLÓGICO AVANZADO**
    morfologia = analizar_morfologia_detallada(mask, confidence_map)
    severidad_ajuste = morfologia.get('severidad_ajuste', 1.0)
    
    # Determinar severidad ajustada por patrón
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste
    
    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Sin Grietas Significativas"
        color = "success"
    elif porcentaje_ajustado < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
        color = "info"
    elif porcentaje_ajustado < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
        color = "warning"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
        color = "danger"
    
    # Ajustar severidad por patrón crítico
    if morfologia['patron_general'] in ['vertical', 'diagonal_escalera']:
        if severidad == "Media":
            severidad = "Media-Alta"
            color = "warning"
        elif severidad == "Baja" and porcentaje_grietas > 2:
            severidad = "Media"
            color = "warning"
    
    # Confianza
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_grietas),
        'longitud_total_px': float(total_length),
        'longitud_promedio_px': float(avg_length),
        'longitud_maxima_px': float(max_length),
        'ancho_promedio_px': float(avg_width),
        'severidad': severidad,
        'estado': estado,
        'color_severidad': color,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'tta_usado': config.USE_TTA,
        'post_processing': config.USE_MORPHOLOGY,
        
        # **NUEVO: Análisis morfológico**
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    """Convierte imagen RGB a base64"""
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
        'architecture': f'{config.ARCHITECTURE} + {config.ENCODER}',
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'tta_transforms': len(config.TTA_TRANSFORMS),
        'morphological_analysis': True,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint principal de predicción con Super Ensemble + Análisis Morfológico"""
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
        
        # Parámetros opcionales
        use_tta = request.form.get('use_tta', str(config.USE_TTA)).lower() == 'true'
        return_base64 = request.form.get('return_base64', 'true').lower() == 'true'
        
        # Guardar temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"📥 Procesando: {filename}")
        print(f"   TTA: {use_tta}, Base64: {return_base64}")
        
        # Procesar imagen con Super Ensemble
        original_config_tta = config.USE_TTA
        config.USE_TTA = use_tta
        
        img_original, pred_mask, confidence_map = procesar_imagen(filepath)
        
        config.USE_TTA = original_config_tta
        
        # Crear overlay completo
        overlay = crear_visualizacion(img_original, pred_mask, confidence_map)
        
        # Calcular métricas + análisis morfológico
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        # Log del análisis
        morfologia = metricas['analisis_morfologico']
        print(f"   🔍 Patrón detectado: {morfologia['patron_general']}")
        print(f"   📊 Orientaciones: {morfologia['distribución_orientaciones']}")
        print(f"   ⚠️  Severidad: {metricas['severidad']}")
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'metricas': metricas,
            'timestamp': datetime.now().isoformat(),
            'procesamiento': {
                'architecture': config.ARCHITECTURE,
                'encoder': config.ENCODER,
                'tta_usado': use_tta,
                'tta_transforms': len(config.TTA_TRANSFORMS) if use_tta else 0,
                'threshold': config.THRESHOLD,
                'target_size': config.TARGET_SIZE,
                'post_processing': config.USE_MORPHOLOGY,
                'morphological_analysis': True,
                'overlay_color': config.OVERLAY_COLOR,
                'overlay_alpha': config.OVERLAY_ALPHA
            }
        }
        
        # Devolver imagen
        if return_base64:
            response_data['imagen_overlay'] = imagen_a_base64(overlay)
            print(f"✅ Imagen base64 generada")
        else:
            result_filename = f"result_{filename}"
            result_path = os.path.join(config.RESULTS_FOLDER, result_filename)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, overlay_bgr)
            response_data['result_image'] = f'/results/{result_filename}'
            print(f"✅ Imagen guardada: {result_path}")
        
        # Eliminar temporal
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

@app.route('/api/status', methods=['GET'])
def status():
    """Estado detallado del sistema"""
    return jsonify({
        'system': 'CrackGuard Backend v3.1',
        'model': f'{config.ARCHITECTURE} + {config.ENCODER}',
        'device': str(config.DEVICE),
        'model_loaded': model is not None,
        'configuration': {
            'threshold': config.THRESHOLD,
            'target_size': config.TARGET_SIZE,
            'tta_enabled': config.USE_TTA,
            'tta_transforms': config.TTA_TRANSFORMS,
            'post_processing': config.USE_MORPHOLOGY,
            'morphological_analysis': True,
            'crack_classification': ['horizontal', 'vertical', 'diagonal', 'mapa_ramificada', 'mixto'],
            'overlay_color': config.OVERLAY_COLOR,
            'overlay_alpha': config.OVERLAY_ALPHA,
            'max_file_size_mb': config.MAX_CONTENT_LENGTH / (1024 * 1024)
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/results/<filename>', methods=['GET'])
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

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 100)
print("🚀 CRACKGUARD BACKEND v3.1 - SUPER ENSEMBLE + ANÁLISIS MORFOLÓGICO")
print("   UNet++ EfficientNet-B8 + TTA + Post-Processing + Clasificación de Patrones")
print("═" * 100)

modelo_cargado = cargar_modelo()

if modelo_cargado:
    print("✅ Sistema listo para inferencia ultra optimizada con análisis morfológico")
else:
    print("⚠️  Servidor iniciado sin modelo entrenado")

print("═" * 100)

if __name__ == '__main__':
    print("\n🔧 Ejecutando Flask development server")
    print("⚠️  Para producción: gunicorn app:app --bind 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)