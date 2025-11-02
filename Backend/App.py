 
"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD v3.3 - DETECCIÓN ULTRA ULTRA SENSIBLE
Desarrollado por: Angel226m
Fecha: 2025-10-27
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
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    ARCHITECTURE = 'UnetPlusPlus'
    ENCODER = 'timm-efficientnet-b8'
    TARGET_SIZE = 640
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    THRESHOLD = 0.5
    MIN_CRACK_COVERAGE = 0.25
    
    USE_TTA = True
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    
    USE_MORPHOLOGY = True
    USE_CONNECTED_COMPONENTS = True
    MIN_COMPONENT_SIZE = 5  # ✅ ULTRA SENSIBLE (5px²)
    
    OVERLAY_COLOR = 'red'
    OVERLAY_ALPHA = 0.4
    
    ANGLE_TOLERANCE = 15
    MIN_CRACK_LENGTH = 10  # ✅ ULTRA SENSIBLE (10px)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None
model_loaded = False

def cargar_modelo():
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
        
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ TTA: {len(config.TTA_TRANSFORMS)}x")
        print(f"   ✓ Umbral de grieta: {config.MIN_CRACK_LENGTH}px (ULTRA SENSIBLE)")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta(contour):
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
    except:
        return None, "indefinido"

def clasificar_patron_global(contours, mask_binary):
    if len(contours) == 0:
        return {
            'patron': 'sin_grietas',
            'descripcion': 'No se detectaron grietas',
            'causa_probable': 'N/A',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Estructura sin daños'
        }
    
    orientaciones = []
    longitudes = []
    grietas_filtradas = 0
    
    for contour in contours:
        length = cv2.arcLength(contour, False)
        
        if length < config.MIN_CRACK_LENGTH:
            grietas_filtradas += 1
            continue
        
        angle, tipo = analizar_orientacion_grieta(contour)
        
        if angle is not None:
            orientaciones.append(tipo)
            longitudes.append(length)
    
    # ✅ LOG DETALLADO
    print(f"   📏 Grietas totales: {len(contours)} | Filtradas (< {config.MIN_CRACK_LENGTH}px): {grietas_filtradas} | Analizadas: {len(orientaciones)}")
    
    if not orientaciones:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    
    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts.most_common(1)[0][0]
    diversidad = len(tipo_counts)
    num_grietas = len(orientaciones)
    
    # ✅ LOG DETALLADO
    print(f"   📐 Orientaciones detectadas: {dict(tipo_counts)}")
    
    if diversidad >= 3:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    elif tipo_dominante == "horizontal" and tipo_counts["horizontal"] / len(orientaciones) > 0.6:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    elif tipo_dominante == "vertical" and tipo_counts["vertical"] / len(orientaciones) > 0.6:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    elif tipo_dominante == "diagonal" and tipo_counts["diagonal"] / len(orientaciones) > 0.5:
        return {
            'patron': 'diagonal_escalera',
            'descripcion': 'Grietas diagonales - ⚠️ MUY CRÍTICO',
            'causa_probable': 'Esfuerzos cortantes, movimiento del terreno',
            'severidad_ajuste': 1.4,
            'recomendacion': '🔴 Evaluación estructural CRÍTICA'
        }
    elif diversidad >= 2:
        return {
            'patron': 'mixto',
            'descripcion': 'Patrón mixto de agrietamiento',
            'causa_probable': 'Combinación de factores',
            'severidad_ajuste': 1.2,
            'recomendacion': 'Inspección profesional detallada'
        }
    else:
        return {
            'patron': 'irregular',
            'descripcion': 'Patrón irregular',
            'causa_probable': 'Causa indeterminada',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Inspección profesional'
        }

def analizar_morfologia_detallada(mask, contours):
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
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
    
    grietas_detalle.sort(key=lambda x: x['longitud_px'], reverse=True)
    
    return {
        'patron_general': patron_info['patron'],
        'descripcion_patron': patron_info['descripcion'],
        'causa_probable': patron_info['causa_probable'],
        'severidad_ajuste': patron_info['severidad_ajuste'],
        'recomendacion': patron_info.get('recomendacion', 'Monitoreo'),
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
        print(f"   ⚠️  Severidad: {metricas['severidad']}")
        
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
print("🚀 CRACKGUARD BACKEND v3.3 - DETECCIÓN ULTRA ULTRA SENSIBLE")
print(f"   Umbral mínimo: {Config.MIN_CRACK_LENGTH}px | Componente mínimo: {Config.MIN_COMPONENT_SIZE}px²")
print("═" * 100)

if cargar_modelo():
    print("✅ Sistema listo para inferencia")
else:
    print("⚠️  Servidor iniciado sin modelo")

print("═" * 100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)'''


"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD v3.4 - ANÁLISIS MORFOLÓGICO ULTRA PRECISO
Desarrollado por: Angel226m
Optimizado para CPU + Detección de patrones mejorada
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
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    ARCHITECTURE = 'UnetPlusPlus'
    ENCODER = 'timm-efficientnet-b8'
    TARGET_SIZE = 640
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    THRESHOLD = 0.5
    MIN_CRACK_COVERAGE = 0.25
    
    USE_TTA = True
    TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    
    USE_MORPHOLOGY = True
    USE_CONNECTED_COMPONENTS = True
    MIN_COMPONENT_SIZE = 5
    
    OVERLAY_COLOR = 'red'
    OVERLAY_ALPHA = 0.4
    
    # ✅ NUEVOS PARÁMETROS MORFOLÓGICOS PRECISOS
    ANGLE_HORIZONTAL = 15      # ±15° = horizontal
    ANGLE_VERTICAL = 15        # 90±15° = vertical
    ANGLE_DIAGONAL_45 = 12     # 45±12° = diagonal
    ANGLE_DIAGONAL_135 = 12    # 135±12° = diagonal
    
    MIN_CRACK_LENGTH = 10
    MIN_CRACK_AREA = 15        # px² mínimo para analizar
    MIN_ASPECT_RATIO = 2.5     # Largo/ancho para ser grieta lineal
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

# ✅ TRANSFORM GLOBAL (no recrear cada vez)
transform = None

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

model = None
model_loaded = False

def cargar_modelo():
    global model, model_loaded, transform
    
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
        
        model = model.to(config.DEVICE)
        model.eval()
        
        # ✅ CREAR TRANSFORM UNA SOLA VEZ
        transform = A.Compose([
            A.Resize(config.TARGET_SIZE, config.TARGET_SIZE, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2()
        ])
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ TTA: {len(config.TTA_TRANSFORMS)}x")
        print(f"   ✓ Análisis morfológico: ULTRA PRECISO")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ✅ ANÁLISIS MORFOLÓGICO MEJORADO Y PRECISO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_precisa(contour):
    """
    Análisis de orientación MEJORADO con múltiples métodos
    """
    if len(contour) < 5:
        return None, "indefinido", 0.0
    
    try:
        # 🔹 MÉTODO 1: Ajuste de línea (PCA)
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy, vx) * 180 / np.pi
        angle_fit = angle_fit[0] if isinstance(angle_fit, np.ndarray) else angle_fit
        
        # 🔹 MÉTODO 2: Rectángulo mínimo rotado
        rect = cv2.minAreaRect(contour)
        angle_rect = rect[2]
        
        # Normalizar ángulos a [0, 180)
        if angle_fit < 0:
            angle_fit += 180
        
        # OpenCV devuelve ángulos en [-90, 0] → normalizar
        if angle_rect < 0:
            angle_rect += 90
        else:
            angle_rect = angle_rect if rect[1][0] < rect[1][1] else angle_rect + 90
        
        # 🔹 MÉTODO 3: Momentos de imagen
        M = cv2.moments(contour)
        if M['mu20'] != M['mu02']:
            angle_moments = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02']) * 180 / np.pi
            if angle_moments < 0:
                angle_moments += 180
        else:
            angle_moments = 0 if M['mu20'] > M['mu02'] else 90
        
        # ✅ PROMEDIO PONDERADO (fitLine tiene más peso)
        angle = (angle_fit * 0.5 + angle_rect * 0.3 + angle_moments * 0.2)
        
        # Calcular confianza basada en consistencia
        angles = [angle_fit, angle_rect, angle_moments]
        std_angles = np.std(angles)
        confianza = max(0.5, 1.0 - (std_angles / 45.0))  # Baja confianza si hay mucha variación
        
        # 📐 CLASIFICACIÓN MEJORADA
        if (angle < config.ANGLE_HORIZONTAL) or (angle > 180 - config.ANGLE_HORIZONTAL):
            tipo = "horizontal"
        elif (90 - config.ANGLE_VERTICAL < angle < 90 + config.ANGLE_VERTICAL):
            tipo = "vertical"
        elif (45 - config.ANGLE_DIAGONAL_45 < angle < 45 + config.ANGLE_DIAGONAL_45):
            tipo = "diagonal_45"
        elif (135 - config.ANGLE_DIAGONAL_135 < angle < 135 + config.ANGLE_DIAGONAL_135):
            tipo = "diagonal_135"
        else:
            tipo = "irregular"
        
        return float(angle), tipo, float(confianza)
        
    except Exception as e:
        print(f"   ⚠️ Error en análisis de orientación: {e}")
        return None, "indefinido", 0.0

def calcular_metricas_grieta(contour, mask_shape):
    """
    Calcula métricas detalladas de una grieta individual
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, False)
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    
    # Ancho promedio
    width_avg = area / max(perimeter, 1)
    
    # Compacidad (qué tan "recta" es)
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    compactness = area / max(rect_area, 1)
    
    # Extensión (qué % del bbox ocupa)
    bbox_area = w * h
    extent = area / max(bbox_area, 1)
    
    # Solidez (vs convex hull)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'width_avg': width_avg,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'extent': extent,
        'solidity': solidity,
        'bbox': (x, y, w, h)
    }

def clasificar_patron_global_mejorado(contours, mask_binary, mask_shape):
    """
    Clasificación de patrón con análisis estadístico robusto
    """
    if len(contours) == 0:
        return {
            'patron': 'sin_grietas',
            'descripcion': 'No se detectaron grietas',
            'causa_probable': 'N/A',
            'severidad_ajuste': 1.0,
            'recomendacion': 'Estructura sin daños',
            'confianza_patron': 1.0
        }
    
    orientaciones = []
    longitudes = []
    anchos = []
    confianzas = []
    grietas_validas = []
    
    for contour in contours:
        metricas = calcular_metricas_grieta(contour, mask_shape)
        
        # ✅ FILTROS DE CALIDAD
        if metricas['perimeter'] < config.MIN_CRACK_LENGTH:
            continue
        if metricas['area'] < config.MIN_CRACK_AREA:
            continue
        if metricas['aspect_ratio'] < config.MIN_ASPECT_RATIO:
            continue  # Descartar manchas redondas
        
        angle, tipo, conf = analizar_orientacion_grieta_precisa(contour)
        
        if angle is not None and conf > 0.5:
            orientaciones.append(tipo)
            longitudes.append(metricas['perimeter'])
            anchos.append(metricas['width_avg'])
            confianzas.append(conf)
            grietas_validas.append({
                'tipo': tipo,
                'angulo': angle,
                'longitud': metricas['perimeter'],
                'ancho': metricas['width_avg'],
                'confianza': conf,
                'metricas': metricas
            })
    
    # ✅ LOG DETALLADO
    print(f"   📏 Grietas totales: {len(contours)} | Válidas: {len(grietas_validas)}")
    
    if not orientaciones:
        return {
            'patron': 'superficial',
            'descripcion': 'Microgrietas o ruido',
            'causa_probable': 'Desgaste superficial menor',
            'severidad_ajuste': 0.7,
            'recomendacion': 'Monitoreo periódico',
            'confianza_patron': 0.6
        }
    
    # 📊 ANÁLISIS ESTADÍSTICO
    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts.most_common(1)[0][0]
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)
    confianza_promedio = np.mean(confianzas)
    
    # ✅ LOG DETALLADO
    print(f"   📐 Orientaciones: {dict(tipo_counts)}")
    print(f"   🎯 Dominante: {tipo_dominante} ({porcentaje_dominante*100:.1f}%)")
    print(f"   💯 Confianza promedio: {confianza_promedio:.2f}")
    
    # 🔍 CLASIFICACIÓN MEJORADA
    
    # PATRÓN VERTICAL (CRÍTICO)
    if tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales predominantes - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos diferenciales',
            'severidad_ajuste': 1.4,
            'recomendacion': '🔴 Inspección estructural URGENTE',
            'confianza_patron': confianza_promedio
        }
    
    # PATRÓN DIAGONAL (MUY CRÍTICO)
    if (tipo_dominante in ["diagonal_45", "diagonal_135"]) and porcentaje_dominante > 0.50:
        return {
            'patron': 'diagonal_escalera',
            'descripcion': 'Grietas diagonales - ⚠️ MUY CRÍTICO',
            'causa_probable': 'Esfuerzos cortantes, movimiento del terreno, falla de cimentación',
            'severidad_ajuste': 1.5,
            'recomendacion': '🔴 Evaluación estructural CRÍTICA INMEDIATA',
            'confianza_patron': confianza_promedio
        }
    
    # PATRÓN HORIZONTAL
    if tipo_dominante == "horizontal" and porcentaje_dominante > 0.60:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas horizontales predominantes',
            'causa_probable': 'Flexión estructural, presión lateral, asentamiento uniforme',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación',
            'confianza_patron': confianza_promedio
        }
    
    # PATRÓN RAMIFICADO (RETRACCIÓN)
    if diversidad >= 3 and porcentaje_dominante < 0.5:
        longitud_promedio = np.mean(longitudes)
        if longitud_promedio < 50:  # Grietas cortas y ramificadas
            return {
                'patron': 'ramificada_mapa',
                'descripcion': 'Patrón ramificado tipo mapa - Retracción térmica',
                'causa_probable': 'Cambios térmicos, secado rápido del material, contracción',
                'severidad_ajuste': 0.85,
                'recomendacion': 'Monitoreo periódico, evaluar impermeabilización',
                'confianza_patron': confianza_promedio
            }
    
    # PATRÓN MIXTO
    if diversidad >= 2 and porcentaje_dominante < 0.65:
        return {
            'patron': 'mixto',
            'descripcion': 'Patrón mixto de agrietamiento',
            'causa_probable': 'Combinación de factores estructurales y ambientales',
            'severidad_ajuste': 1.2,
            'recomendacion': 'Inspección profesional detallada requerida',
            'confianza_patron': confianza_promedio * 0.9
        }
    
    # PATRÓN IRREGULAR (POR DEFECTO)
    return {
        'patron': 'irregular',
        'descripcion': 'Patrón irregular o no clasificado',
        'causa_probable': 'Causa indeterminada - requiere análisis profesional',
        'severidad_ajuste': 1.0,
        'recomendacion': 'Inspección profesional recomendada',
        'confianza_patron': confianza_promedio * 0.8
    }

def analizar_morfologia_detallada(mask, contours):
    """
    Análisis morfológico completo con filtrado inteligente
    """
    mask_binary = (mask > 127).astype(np.uint8)
    patron_info = clasificar_patron_global_mejorado(contours, mask_binary, mask.shape)
    
    grietas_detalle = []
    orientaciones_count = {
        "horizontal": 0,
        "vertical": 0,
        "diagonal_45": 0,
        "diagonal_135": 0,
        "irregular": 0
    }
    
    for i, contour in enumerate(contours):
        metricas = calcular_metricas_grieta(contour, mask.shape)
        
        # ✅ FILTROS DE CALIDAD
        if metricas['perimeter'] < config.MIN_CRACK_LENGTH:
            continue
        if metricas['area'] < config.MIN_CRACK_AREA:
            continue
        if metricas['aspect_ratio'] < config.MIN_ASPECT_RATIO:
            continue
        
        angle, tipo, conf = analizar_orientacion_grieta_precisa(contour)
        
        if angle is None or conf < 0.5:
            continue
        
        orientaciones_count[tipo] += 1
        
        x, y, w, h = metricas['bbox']
        
        grietas_detalle.append({
            'id': i + 1,
            'longitud_px': round(float(metricas['perimeter']), 2),
            'area_px': int(metricas['area']),
            'ancho_promedio_px': round(float(metricas['width_avg']), 2),
            'angulo_grados': round(angle, 1),
            'orientacion': tipo,
            'confianza': round(conf, 2),
            'aspect_ratio': round(float(metricas['aspect_ratio']), 2),
            'compacidad': round(float(metricas['compactness']), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
    
    grietas_detalle.sort(key=lambda x: x['longitud_px'], reverse=True)
    
    return {
        'patron_general': patron_info['patron'],
        'descripcion_patron': patron_info['descripcion'],
        'causa_probable': patron_info['causa_probable'],
        'severidad_ajuste': patron_info['severidad_ajuste'],
        'recomendacion': patron_info.get('recomendacion', 'Monitoreo'),
        'confianza_patron': patron_info.get('confianza_patron', 0.8),
        'distribucion_orientaciones': orientaciones_count,
        'num_grietas_analizadas': len(grietas_detalle),
        'grietas_principales': grietas_detalle[:5]
    }

# ═══════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def advanced_postprocess(mask):
    """
    Postprocesado condicional optimizado
    """
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    # ✅ EARLY EXIT si no hay señal
    if mask_np.max() < 0.3:
        return (mask_np > 0.5).astype(np.float32)
    
    mask_np = (mask_np > 0.5).astype(np.uint8)
    
    # ✅ MORFOLOGÍA solo si hay suficiente área
    if config.USE_MORPHOLOGY and mask_np.sum() > 500:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    
    # ✅ COMPONENTES solo si hay muchos
    if config.USE_CONNECTED_COMPONENTS:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        
        if num_labels > 100:  # Solo si hay muchos componentes
            cleaned_mask = np.zeros_like(mask_np)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_SIZE:
                    cleaned_mask[labels == i] = 1
            mask_np = cleaned_mask
    
    return mask_np.astype(np.float32)

def predict_with_tta(model, img_tensor):
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
    if not model_loaded:
        raise RuntimeError("Modelo no cargado")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # ✅ USAR TRANSFORM GLOBAL
    img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(config.DEVICE)
    
    if use_tta:
        pred = predict_with_tta(model, img_tensor)
    else:
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
    
    # ✅ RESIZE MÁS RÁPIDO con torch.nn.functional
    import torch.nn.functional as F
    confidence_map = F.interpolate(
        pred, size=(h, w), mode='bilinear', align_corners=False
    ).cpu().numpy()[0, 0]
    
    confidence_map = advanced_postprocess(torch.from_numpy(confidence_map))
    pred_mask = (confidence_map > config.THRESHOLD).astype(np.uint8) * 255
    
    return img_rgb, pred_mask, confidence_map

def crear_overlay(img_original, mask):
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
    mask_binary = (mask > 127).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    # ✅ EARLY EXIT si no hay grietas reales
    if pixeles_positivos < 200:
        morfologia = {
            'patron_general': 'sin_grietas',
            'descripcion_patron': 'Sin daños detectables',
            'causa_probable': 'N/A',
            'num_grietas_analizadas': 0,
            'grietas_principales': [],
            'distribucion_orientaciones': {},
            'severidad_ajuste': 1.0,
            'confianza_patron': 1.0
        }
        num_contours = 0
        total_length = 0
        avg_length = 0
        max_length = 0
        avg_width = 0
    else:
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
    
    confianza = min(95.0, 80.0 + (morfologia.get('confianza_patron', 0.8) * 15.0))
    
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
        'morphological_analysis': 'ULTRA_PRECISO',
        'min_crack_length': config.MIN_CRACK_LENGTH,
        'angle_precision': {
            'horizontal': f'±{config.ANGLE_HORIZONTAL}°',
            'vertical': f'±{config.ANGLE_VERTICAL}°',
            'diagonal': f'±{config.ANGLE_DIAGONAL_45}°'
        },
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
        
        # ✅ MEDIR TIEMPO
        start_time = time.time()
        
        img_original, pred_mask, confidence_map = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        proc_time = time.time() - start_time
        
        morfologia = metricas['analisis_morfologico']
        print(f"   🔍 Patrón: {morfologia['patron_general']} (confianza: {morfologia.get('confianza_patron', 0):.2f})")
        print(f"   ⚠️  Severidad: {metricas['severidad']}")
        print(f"   ⏱️  Tiempo: {proc_time:.3f}s")
        
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
                'morphological_analysis': 'ULTRA_PRECISO',
                'min_crack_length': config.MIN_CRACK_LENGTH,
                'tiempo_ms': round(proc_time * 1000, 1),
                'angle_precision': {
                    'horizontal': config.ANGLE_HORIZONTAL,
                    'vertical': config.ANGLE_VERTICAL,
                    'diagonal_45': config.ANGLE_DIAGONAL_45,
                    'diagonal_135': config.ANGLE_DIAGONAL_135
                }
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
print("🚀 CRACKGUARD BACKEND v3.4 - ANÁLISIS MORFOLÓGICO ULTRA PRECISO")
print(f"   📐 Precisión angular: ±{Config.ANGLE_HORIZONTAL}° (H), ±{Config.ANGLE_VERTICAL}° (V)")
print(f"   🔍 Métodos: FitLine + MinAreaRect + Momentos (ponderados)")
print(f"   ✅ Filtros: Longitud ≥{Config.MIN_CRACK_LENGTH}px | Área ≥{Config.MIN_CRACK_AREA}px² | Aspect ≥{Config.MIN_ASPECT_RATIO}")
print("═" * 100)

if cargar_modelo():
    print("✅ Sistema listo para inferencia")
else:
    print("⚠️  Servidor iniciado sin modelo")

print("═" * 100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)