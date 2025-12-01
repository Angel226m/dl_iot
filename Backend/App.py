#!/usr/bin/env python3

'''import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
from functools import lru_cache
import warnings
import time
import eventlet
import requests

warnings.filterwarnings('ignore')
eventlet.monkey_patch()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ✅ CORS CONFIGURADO CORRECTAMENTE
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ✅ SOCKETIO PARA WEBSOCKET
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    logger=True,
    engineio_logger=True
)

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
    
    ANGLE_TOLERANCE = 12
    MIN_CRACK_LENGTH = 10
    
    MAX_IMAGE_DIMENSION = 2048
    MAX_GRIETAS_ANALIZAR = 10
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TORCH_THREADS = 4
    CV2_THREADS = 4
    
    PNG_COMPRESSION = 6
    JPEG_QUALITY = 92
    USE_JPEG_OUTPUT = False

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

torch.set_num_threads(config.TORCH_THREADS)
torch.set_num_interop_threads(config.TORCH_THREADS)
cv2.setNumThreads(config.CV2_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# VARIABLES GLOBALES - DISPOSITIVOS CONECTADOS
# ═══════════════════════════════════════════════════════════════════════════

connected_devices = {}  # { device_id: {...info} }
latest_photos = {}  # { device_id: base64_image }

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
        
        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)
            print(f"   ⚡ Optimizado para CPU ({config.TORCH_THREADS} threads)")
        
        print(f"   ✓ Device: {config.DEVICE}")
        print(f"   ✓ TTA: {len(config.TTA_TRANSFORMS)}x")
        print(f"   ✓ Max resolución: {config.MAX_IMAGE_DIMENSION}px")
        print(f"   ✓ Análisis morfológico: CONDICIONAL")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_mejorada(contour):
    if len(contour) < 5:
        return None, "indefinido"
    
    try:
        moments = cv2.moments(contour)
        if moments['mu20'] - moments['mu02'] != 0:
            angle_moments = 0.5 * np.arctan2(2 * moments['mu11'], 
                                            moments['mu20'] - moments['mu02'])
            angle_moments = np.degrees(angle_moments)
        else:
            angle_moments = None
        
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy[0], vx[0]) * 180 / np.pi
        
        angle = angle_moments if angle_moments is not None else angle_fit
        angle = angle % 180
        if angle < 0:
            angle += 180
        
        tol = config.ANGLE_TOLERANCE
        
        if angle < tol or angle > (180 - tol):
            tipo = "horizontal"
        elif abs(angle - 90) < tol:
            tipo = "vertical"
        elif abs(angle - 45) < tol:
            tipo = "diagonal"
        elif abs(angle - 135) < tol:
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
    
    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]
    
    if len(indices_validos) == 0:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    
    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]]
    
    orientaciones = []
    grietas_filtradas = len(contours) - len(top_indices)
    
    for idx in top_indices:
        angle, tipo = analizar_orientacion_grieta_mejorada(contours[idx])
        if angle is not None:
            orientaciones.append(tipo)
    
    print(f"   📏 Grietas totales: {len(contours)} | Filtradas: {grietas_filtradas} | Analizadas: {len(orientaciones)}")
    
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
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)
    
    print(f"   📐 Orientaciones: {dict(tipo_counts)} | Dominante: {tipo_dominante} ({porcentaje_dominante:.1%})")
    
    if diversidad >= 3 and porcentaje_dominante < 0.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    
    elif tipo_dominante == "horizontal" and porcentaje_dominante > 0.55:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    
    elif tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    
    elif tipo_dominante == "diagonal" and porcentaje_dominante > 0.45:
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
    
    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]
    
    if len(indices_validos) == 0:
        return {
            'patron_general': 'superficial',
            'descripcion_patron': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }
    
    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][::-1]
    
    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}
    
    for rank, idx in enumerate(top_indices, 1):
        contour = contours[idx]
        length = longitudes[idx]
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta_mejorada(contour)
        
        width = area / length if length > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        orientaciones_count[tipo] += 1
        
        grietas_detalle.append({
            'id': rank,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
    
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
# PROCESAMIENTO DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
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
    
    h_orig, w_orig = img.shape[:2]
    original_dimensions = (w_orig, h_orig)
    
    if max(h_orig, w_orig) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"   📐 Imagen optimizada: {w_orig}x{h_orig} → {new_w}x{new_h} ({scale:.2%})")
    
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
    
    return img_rgb, pred_mask, confidence_map, original_dimensions

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)
    
    if config.OVERLAY_COLOR == 'red':
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
    
    if num_contours == 0 or porcentaje_grietas < 0.1:
        return {
            'total_pixeles': int(total_pixeles),
            'pixeles_con_grietas': 0,
            'porcentaje_grietas': 0.0,
            'num_grietas_detectadas': 0,
            'longitud_total_px': 0.0,
            'longitud_promedio_px': 0.0,
            'longitud_maxima_px': 0.0,
            'ancho_promedio_px': 0.0,
            'severidad': "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map.max()),
            'confidence_mean': float(confidence_map.mean()),
            'analisis_morfologico': None
        }
    
    print(f"   ✅ Detectadas {num_contours} grietas → Iniciando análisis morfológico...")
    
    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes.mean()
    max_length = longitudes.max()
    avg_width = pixeles_positivos / total_length if total_length > 0 else 0
    
    morfologia = analizar_morfologia_detallada(mask, contours)
    
    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste
    
    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Grietas Menores"
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
        'longitud_total_px': round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if config.USE_JPEG_OUTPUT:
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        _, buffer = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION])
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# ✅ RUTAS API REST
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'architecture': f'{config.ARCHITECTURE} + {config.ENCODER}',
        'device': str(config.DEVICE),
        'tta_enabled': config.USE_TTA,
        'cpu_threads': config.TORCH_THREADS,
        'max_resolution': f'{config.MAX_IMAGE_DIMENSION}px',
        'max_grietas_analizar': config.MAX_GRIETAS_ANALIZAR,
        'morphological_analysis': 'condicional (solo si hay grietas)',
        'output_format': 'JPEG' if config.USE_JPEG_OUTPUT else 'PNG',
        'websocket_enabled': True,
        'connected_devices': len(connected_devices),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
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
        
        img_original, pred_mask, confidence_map, original_dims = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        if metricas['analisis_morfologico']:
            morfologia = metricas['analisis_morfologico']
            print(f"   🔍 Patrón: {morfologia['patron_general']}")
            print(f"   ⚠️  Severidad: {metricas['severidad']}")
            print(f"   📊 {morfologia['num_grietas_analizadas']} grietas analizadas (de {metricas['num_grietas_detectadas']} totales)")
        else:
            print(f"   ✅ Sin grietas detectadas")
        
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
                'cpu_optimized': True,
                'cpu_threads': config.TORCH_THREADS,
                'max_resolution': config.MAX_IMAGE_DIMENSION,
                'original_dimensions': {'width': original_dims[0], 'height': original_dims[1]},
                'output_format': 'JPEG' if config.USE_JPEG_OUTPUT else 'PNG',
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
# ✅ RUTAS PARA RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/devices', methods=['GET', 'OPTIONS'])
def list_devices():
    """Lista todos los dispositivos Raspberry Pi conectados"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        devices_list = []
        for dev_id, info in connected_devices.items():
            device_data = {
                'device_id': dev_id,
                'type': info['type'],
                'ip_local': info['ip_local'],
                'capabilities': info['capabilities'],
                'connected_at': info['connected_at']
            }
            
            # ✅ AGREGAR ÚLTIMA FOTO SI EXISTE
            if dev_id in latest_photos:
                device_data['has_photo'] = True
                device_data['last_photo_time'] = latest_photos[dev_id].get('timestamp', '')
            else:
                device_data['has_photo'] = False
            
            devices_list.append(device_data)
        
        print(f"📱 GET /api/devices - {len(devices_list)} dispositivos conectados")
        
        return jsonify({
            'devices': devices_list,
            'total': len(devices_list),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error en /api/devices: {e}")
        traceback.print_exc()
        return jsonify({
            'devices': [],
            'total': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/send_command/<device_id>', methods=['POST', 'OPTIONS'])
def send_command(device_id):
    """Envía comando a un Raspberry Pi específico"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in connected_devices:
            print(f"⚠️  Dispositivo {device_id} no encontrado")
            print(f"   Dispositivos conectados: {list(connected_devices.keys())}")
            return jsonify({
                'error': 'Dispositivo no conectado',
                'device_id': device_id,
                'connected_devices': list(connected_devices.keys())
            }), 404
        
        data = request.get_json() if request.is_json else {}
        action = data.get('action', 'unknown')
        params = data.get('params', {})
        
        device_info = connected_devices[device_id]
        sid = device_info['sid']
        
        print(f"📤 Enviando comando a {device_id}:")
        print(f"   Action: {action}")
        print(f"   Params: {params}")
        print(f"   SID: {sid}")
        
        # Enviar comando al dispositivo específico vía Socket.IO
        socketio.emit('command', {
            'action': action,
            'params': params,
            'timestamp': time.time()
        }, room=sid, namespace='/ws/raspberry')
        
        return jsonify({
            'status': 'sent',
            'device_id': device_id,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error en send_command: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'device_id': device_id,
            'timestamp': datetime.now().isoformat()
        }), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ PROXY PARA STREAMING DE RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/stream/<device_id>', methods=['GET'])
def stream_proxy(device_id):
    """Proxy del stream del Raspberry Pi para evitar Mixed Content"""
    if device_id not in connected_devices:
        return jsonify({'error': 'Dispositivo no conectado'}), 404
    
    device_info = connected_devices[device_id]
    ip_local = device_info['ip_local']
    stream_url = f"http://{ip_local}:8080/video_feed"
    
    print(f"📹 Iniciando proxy de stream para {device_id}")
    print(f"   URL: {stream_url}")
    
    try:
        # Hacer request al Raspberry y hacer streaming de la respuesta
        resp = requests.get(stream_url, stream=True, timeout=10)
        
        def generate():
            try:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            except Exception as e:
                print(f"❌ Error en streaming: {e}")
        
        return Response(
            generate(),
            content_type=resp.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame'),
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        print(f"❌ Error en stream proxy: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ OBTENER ÚLTIMA FOTO DE RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """Obtiene la última foto capturada por el Raspberry Pi"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in latest_photos:
            return jsonify({
                'error': 'No hay foto disponible para este dispositivo',
                'device_id': device_id
            }), 404
        
        photo_data = latest_photos[device_id]
        
        return jsonify({
            'success': True,
            'device_id': device_id,
            'image': photo_data['image'],
            'timestamp': photo_data['timestamp'],
            'metadata': photo_data.get('metadata', {})
        }), 200
        
    except Exception as e:
        print(f"❌ Error al obtener foto: {e}")
        return jsonify({
            'error': str(e),
            'device_id': device_id
        }), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ WEBSOCKET - RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@socketio.on('connect', namespace='/ws/raspberry')
def on_rpi_connect():
    print(f"\n{'='*70}")
    print(f"📡 Raspberry Pi conectado")
    print(f"   SID: {request.sid}")
    print(f"   Namespace: /ws/raspberry")
    print(f"{'='*70}\n")
    emit('server_hello', {
        'message': 'Bienvenido a CrackGuard Backend',
        'timestamp': time.time(),
        'server_version': '3.5'
    })

@socketio.on('register', namespace='/ws/raspberry')
def register_device(data):
    device_id = data.get('device_id', 'unknown')
    device_type = data.get('type', 'raspberry_pi')
    capabilities = data.get('capabilities', [])
    ip_local = data.get('ip_local', 'N/A')
    
    connected_devices[device_id] = {
        'sid': request.sid,
        'type': device_type,
        'capabilities': capabilities,
        'ip_local': ip_local,
        'connected_at': datetime.now().isoformat()
    }
    
    print(f"\n{'='*70}")
    print(f"✅ Dispositivo registrado exitosamente")
    print(f"   Device ID: {device_id}")
    print(f"   Tipo: {device_type}")
    print(f"   IP Local: {ip_local}")
    print(f"   Capacidades: {', '.join(capabilities)}")
    print(f"   SID: {request.sid}")
    print(f"   Total dispositivos: {len(connected_devices)}")
    print(f"{'='*70}\n")
    
    emit('registered', {
        'status': 'ok',
        'device_id': device_id,
        'timestamp': time.time(),
        'message': 'Registro exitoso en CrackGuard Backend'
    })
    
    # ✅ BROADCAST A TODOS LOS CLIENTES QUE HAY UN NUEVO DISPOSITIVO
    socketio.emit('device_connected', {
        'device_id': device_id,
        'type': device_type,
        'ip_local': ip_local,
        'capabilities': capabilities,
        'timestamp': time.time()
    }, namespace='/ws/raspberry', broadcast=True)

@socketio.on('photo_result', namespace='/ws/raspberry')
def receive_photo(data):
    device_id = data.get('device_id', 'unknown')
    image_b64 = data.get('image')
    timestamp_capture = data.get('timestamp', time.time())
    metadata = data.get('metadata', {})
    
    if not image_b64:
        print(f"❌ Foto vacía recibida de {device_id}")
        emit('error', {'message': 'Imagen vacía', 'timestamp': time.time()})
        return
    
    try:
        # ✅ GUARDAR FOTO EN MEMORIA (ÚLTIMA FOTO)
        latest_photos[device_id] = {
            'image': image_b64,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        # Guardar foto en disco también
        filename = f"{device_id}_{int(timestamp_capture)}.jpg"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_b64))
        
        file_size = os.path.getsize(filepath)
        
        print(f"\n{'='*70}")
        print(f"📸 Foto recibida desde Raspberry Pi")
        print(f"   Device ID: {device_id}")
        print(f"   Archivo: {filename}")
        print(f"   Tamaño: {file_size / 1024:.2f} KB")
        print(f"   Resolución: {metadata.get('resolution', 'N/A')}")
        print(f"   Formato: {metadata.get('format', 'N/A')}")
        print(f"{'='*70}\n")
        
        # Confirmar recepción al dispositivo
        emit('photo_saved', {
            'status': 'ok',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size / 1024, 2),
            'timestamp': time.time()
        })
        
        # ✅ BROADCAST A TODOS LOS CLIENTES CON LA FOTO
        socketio.emit('new_photo_available', {
            'device_id': device_id,
            'filename': filename,
            'image': f"data:image/jpeg;base64,{image_b64}",
            'size_kb': round(file_size / 1024, 2),
            'metadata': metadata,
            'timestamp': time.time()
        }, namespace='/ws/raspberry', broadcast=True)
        
    except Exception as e:
        print(f"❌ Error al guardar foto de {device_id}: {e}")
        traceback.print_exc()
        emit('error', {
            'message': f'Error al guardar foto: {str(e)}',
            'timestamp': time.time()
        })

@socketio.on('disconnect', namespace='/ws/raspberry')
def on_rpi_disconnect():
    disconnected_device = None
    
    for dev_id, info in list(connected_devices.items()):
        if info['sid'] == request.sid:
            disconnected_device = dev_id
            del connected_devices[dev_id]
            break
    
    if disconnected_device:
        print(f"\n{'='*70}")
        print(f"📴 Dispositivo desconectado")
        print(f"   Device ID: {disconnected_device}")
        print(f"   SID: {request.sid}")
        print(f"   Dispositivos restantes: {len(connected_devices)}")
        print(f"{'='*70}\n")
        
        # ✅ BROADCAST DESCONEXIÓN
        socketio.emit('device_disconnected', {
            'device_id': disconnected_device,
            'timestamp': time.time()
        }, namespace='/ws/raspberry', broadcast=True)

@socketio.on('pong', namespace='/ws/raspberry')
def handle_pong(data):
    device_id = data.get('device_id', 'unknown')
    print(f"🏓 Pong recibido de {device_id}")

@socketio.on('stream_url', namespace='/ws/raspberry')
def handle_stream_url(data):
    device_id = data.get('device_id', 'unknown')
    url = data.get('url', '')
    status = data.get('status', 'unknown')
    
    print(f"\n📹 Stream URL recibida de {device_id}")
    print(f"   URL: {url}")
    print(f"   Status: {status}\n")

@socketio.on('error', namespace='/ws/raspberry')
def handle_error(data):
    device_id = data.get('device_id', 'unknown')
    error_msg = data.get('error', 'Unknown error')
    
    print(f"\n❌ Error reportado por {device_id}")
    print(f"   Error: {error_msg}\n")

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v3.5 - ULTRA OPTIMIZADO CPU + WEBSOCKET + RASPBERRY PI")
print("═" * 100)
print(f"⚡ PyTorch: {config.TORCH_THREADS} threads | OpenCV: {config.CV2_THREADS} threads")
print(f"📐 Max resolución: {config.MAX_IMAGE_DIMENSION}px (optimización automática)")
print(f"🔍 Análisis morfológico: CONDICIONAL (solo si hay grietas)")
print(f"📊 Max grietas a analizar: {config.MAX_GRIETAS_ANALIZAR}")
print(f"🖼️  Formato salida: {'JPEG (rápido)' if config.USE_JPEG_OUTPUT else 'PNG (calidad)'}")
print(f"🔌 WebSocket: ACTIVADO (namespace: /ws/raspberry)")
print(f"🌐 CORS: ACTIVADO (origins: *)")
print(f"📡 Rutas disponibles:")
print(f"   • GET  /health")
print(f"   • POST /api/predict")
print(f"   • GET  /api/devices")
print(f"   • POST /api/send_command/<device_id>")
print(f"   • GET  /api/stream/<device_id>")
print(f"   • GET  /api/photo/<device_id>")
print(f"   • WS   /socket.io/?EIO=4&transport=websocket (namespace: /ws/raspberry)")
print("═" * 100 + "\n")

if cargar_modelo():
    print("✅ Sistema listo para inferencia (OPTIMIZADO CPU)")
    print("✅ Backend escuchando en 0.0.0.0:5000")
    print("✅ Esperando conexiones de Raspberry Pi...\n")
else:
    print("⚠️  Servidor iniciado sin modelo de IA")
    print("⚠️  Solo funcionarán las rutas de Raspberry Pi\n")

print("═" * 100 + "\n")

if __name__ == '__main__':
    # ✅ USAR SOCKETIO.RUN EN LUGAR DE APP.RUN
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        use_reloader=False,
        log_output=True
    )'''
#!/usr/bin/env python3








'''
#junior
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CRACKGUARD BACKEND v4.2 - CONTROL TOTAL STREAMING + FOTOS + NGINX
Backend optimizado para Nginx proxy reverso
URL: https://crackguard.angelproyect.com
═══════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response
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
from functools import lru_cache
import warnings
import time
import requests
import threading

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Length"]
    }
})

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB para fotos grandes
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
    
    ANGLE_TOLERANCE = 12
    MIN_CRACK_LENGTH = 10
    
    MAX_IMAGE_DIMENSION = 2048
    MAX_GRIETAS_ANALIZAR = 10
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TORCH_THREADS = 4
    CV2_THREADS = 4
    
    PNG_COMPRESSION = 6
    JPEG_QUALITY = 92
    USE_JPEG_OUTPUT = False
    
    DEVICE_HEARTBEAT_TIMEOUT = 30
    DEVICE_CHECK_INTERVAL = 10

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

torch.set_num_threads(config.TORCH_THREADS)
torch.set_num_interop_threads(config.TORCH_THREADS)
cv2.setNumThreads(config.CV2_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# VARIABLES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

connected_devices = {}
latest_photos = {}
device_commands = {}
device_lock = threading.Lock()

model = None
model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO IA
# ═══════════════════════════════════════════════════════════════════════════

def cargar_modelo():
    global model, model_loaded
    
    try:
        print(f"🤖 Cargando modelo UNet++ {config.ENCODER}...")
        
        if not os.path.exists(config.MODEL_PATH):
            print(f"   ⚠️  Modelo no encontrado: {config.MODEL_PATH}")
            return False
        
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
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(config.DEVICE)
        model.eval()
        
        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)
        
        print(f"   ✅ Modelo cargado en {config.DEVICE}")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_mejorada(contour):
    if len(contour) < 5:
        return None, "indefinido"
    
    try:
        moments = cv2.moments(contour)
        if moments['mu20'] - moments['mu02'] != 0:
            angle_moments = 0.5 * np.arctan2(2 * moments['mu11'], 
                                            moments['mu20'] - moments['mu02'])
            angle_moments = np.degrees(angle_moments)
        else:
            angle_moments = None
        
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy[0], vx[0]) * 180 / np.pi
        
        angle = angle_moments if angle_moments is not None else angle_fit
        angle = angle % 180
        if angle < 0:
            angle += 180
        
        tol = config.ANGLE_TOLERANCE
        
        if angle < tol or angle > (180 - tol):
            tipo = "horizontal"
        elif abs(angle - 90) < tol:
            tipo = "vertical"
        elif abs(angle - 45) < tol:
            tipo = "diagonal"
        elif abs(angle - 135) < tol:
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
    
    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]
    
    if len(indices_validos) == 0:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    
    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]]
    
    orientaciones = []
    
    for idx in top_indices:
        angle, tipo = analizar_orientacion_grieta_mejorada(contours[idx])
        if angle is not None:
            orientaciones.append(tipo)
    
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
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)
    
    if diversidad >= 3 and porcentaje_dominante < 0.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    elif tipo_dominante == "horizontal" and porcentaje_dominante > 0.55:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    elif tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    elif tipo_dominante == "diagonal" and porcentaje_dominante > 0.45:
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
    
    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]
    
    if len(indices_validos) == 0:
        return {
            'patron_general': 'superficial',
            'descripcion_patron': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }
    
    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][::-1]
    
    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}
    
    for rank, idx in enumerate(top_indices, 1):
        contour = contours[idx]
        length = longitudes[idx]
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta_mejorada(contour)
        
        width = area / length if length > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        orientaciones_count[tipo] += 1
        
        grietas_detalle.append({
            'id': rank,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
    
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
# PROCESAMIENTO DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
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
    
    h_orig, w_orig = img.shape[:2]
    original_dimensions = (w_orig, h_orig)
    
    if max(h_orig, w_orig) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
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
    
    return img_rgb, pred_mask, confidence_map, original_dimensions

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)
    
    if config.OVERLAY_COLOR == 'red':
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
    
    if num_contours == 0 or porcentaje_grietas < 0.1:
        return {
            'total_pixeles': int(total_pixeles),
            'pixeles_con_grietas': 0,
            'porcentaje_grietas': 0.0,
            'num_grietas_detectadas': 0,
            'longitud_total_px': 0.0,
            'longitud_promedio_px': 0.0,
            'longitud_maxima_px': 0.0,
            'ancho_promedio_px': 0.0,
            'severidad': "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map.max()),
            'confidence_mean': float(confidence_map.mean()),
            'analisis_morfologico': None
        }
    
    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes.mean()
    max_length = longitudes.max()
    avg_width = pixeles_positivos / total_length if total_length > 0 else 0
    
    morfologia = analizar_morfologia_detallada(mask, contours)
    
    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste
    
    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Grietas Menores"
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
        'longitud_total_px': round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if config.USE_JPEG_OUTPUT:
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        _, buffer = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION])
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# LIMPIEZA AUTOMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_offline_devices():
    while True:
        time.sleep(config.DEVICE_CHECK_INTERVAL)
        
        with device_lock:
            now = time.time()
            offline_devices = []
            
            for device_id, info in list(connected_devices.items()):
                last_seen = info.get('last_seen', 0)
                if now - last_seen > config.DEVICE_HEARTBEAT_TIMEOUT:
                    offline_devices.append(device_id)
                    del connected_devices[device_id]
                    if device_id in device_commands:
                        del device_commands[device_id]
            
            if offline_devices:
                print(f"🧹 Dispositivos offline: {offline_devices}")

cleanup_thread = threading.Thread(target=cleanup_offline_devices, daemon=True)
cleanup_thread.start()

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/rpi/register', methods=['POST', 'OPTIONS'])
def rpi_register():
    """Raspberry Pi se registra"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        device_type = data.get('type', 'raspberry_pi')
        capabilities = data.get('capabilities', [])
        ip_local = data.get('ip_local', request.remote_addr)
        stream_port = data.get('stream_port', 8889)
        
        if not device_id:
            return jsonify({'error': 'device_id requerido'}), 400
        
        with device_lock:
            connected_devices[device_id] = {
                'type': device_type,
                'capabilities': capabilities,
                'ip_local': ip_local,
                'stream_port': stream_port,
                'streaming_active': False,
                'connected_at': datetime.now().isoformat(),
                'last_seen': time.time()
            }
            
            if device_id not in device_commands:
                device_commands[device_id] = []
        
        print(f"\n{'='*70}")
        print(f"✅ Raspberry Pi registrado: {device_id}")
        print(f"   IP: {ip_local}")
        print(f"   Stream port: {stream_port}")
        print(f"{'='*70}\n")
        
        return jsonify({
            'status': 'registered',
            'device_id': device_id,
            'backend_version': '4.2',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error register: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/heartbeat', methods=['POST', 'OPTIONS'])
def rpi_heartbeat():
    """Heartbeat + comandos pendientes"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        streaming_status = data.get('streaming_active', False)
        
        if not device_id or device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404
        
        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()
            connected_devices[device_id]['streaming_active'] = streaming_status
        
        commands = []
        with device_lock:
            if device_id in device_commands and device_commands[device_id]:
                commands = device_commands[device_id].copy()
                device_commands[device_id] = []
        
        return jsonify({
            'status': 'ok',
            'commands': commands,
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        print(f"❌ Error heartbeat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/photo', methods=['POST', 'OPTIONS'])
def rpi_upload_photo():
    """Recibir foto desde Raspberry Pi"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        image_b64 = data.get('image')
        metadata = data.get('metadata', {})
        
        if not device_id or not image_b64:
            return jsonify({'error': 'device_id e image requeridos'}), 400
        
        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404
        
        with device_lock:
            latest_photos[device_id] = {
                'image': image_b64,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            connected_devices[device_id]['last_seen'] = time.time()
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device_id}_{timestamp_str}.jpg"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]
        
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_b64))
        
        file_size_kb = os.path.getsize(filepath) / 1024
        
        print(f"\n📸 Foto de {device_id}: {filename} ({file_size_kb:.1f}KB)")
        
        return jsonify({
            'status': 'saved',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size_kb, 2),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error upload_photo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/devices', methods=['GET', 'OPTIONS'])
def list_devices():
    """Lista dispositivos conectados"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        devices_list = []
        
        with device_lock:
            for dev_id, info in connected_devices.items():
                device_data = {
                    'device_id': dev_id,
                    'type': info['type'],
                    'ip_local': info['ip_local'],
                    'stream_port': info.get('stream_port', 8889),
                    'streaming_active': info.get('streaming_active', False),
                    'stream_url_direct': f"http://{info['ip_local']}:{info.get('stream_port', 8889)}/cam",
                    'stream_url_proxy': f"/api/stream/{dev_id}",
                    'capabilities': info['capabilities'],
                    'connected_at': info['connected_at'],
                    'last_seen_ago': round(time.time() - info.get('last_seen', 0), 1)
                }
                
                if dev_id in latest_photos:
                    device_data['has_photo'] = True
                    device_data['last_photo_time'] = latest_photos[dev_id]['timestamp']
                else:
                    device_data['has_photo'] = False
                
                devices_list.append(device_data)
        
        return jsonify({
            'devices': devices_list,
            'total': len(devices_list),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error list_devices: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/start/<device_id>', methods=['POST', 'OPTIONS'])
def start_streaming(device_id):
    """🎬 Iniciar streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404
        
        command = {
            'action': 'start_streaming',
            'params': {},
            'timestamp': time.time()
        }
        
        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()
        
        print(f"🎬 START STREAMING → {device_id}")
        
        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'start_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error start_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/stop/<device_id>', methods=['POST', 'OPTIONS'])
def stop_streaming(device_id):
    """🛑 Detener streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404
        
        command = {
            'action': 'stop_streaming',
            'params': {},
            'timestamp': time.time()
        }
        
        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()
        
        print(f"🛑 STOP STREAMING → {device_id}")
        
        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'stop_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error stop_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/capture/<device_id>', methods=['POST', 'OPTIONS'])
def send_capture_command(device_id):
    """📸 Capturar foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404
        
        data = request.get_json() if request.is_json else {}
        resolution = data.get('resolution', '1920x1080')
        
        command = {
            'action': 'capture',
            'params': {
                'resolution': resolution,
                'format': 'jpg'
            },
            'timestamp': time.time()
        }
        
        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()
        
        print(f"📸 CAPTURE → {device_id} ({resolution})")
        
        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'capture',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """Ver última foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if device_id not in latest_photos:
            return jsonify({'error': 'No hay foto'}), 404
        
        photo_data = latest_photos[device_id]
        
        return jsonify({
            'success': True,
            'device_id': device_id,
            'image': photo_data['image'],
            'timestamp': photo_data['timestamp'],
            'metadata': photo_data.get('metadata', {})
        }), 200
        
    except Exception as e:
        print(f"❌ Error latest_photo: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ PROXY STREAMING WebRTC
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/stream/<device_id>', methods=['GET'])
def stream_proxy(device_id):
    """Proxy HTML del MediaMTX"""
    if device_id not in connected_devices:
        return jsonify({'error': 'Dispositivo no conectado'}), 404
    
    device_info = connected_devices[device_id]
    ip_local = device_info['ip_local']
    stream_port = device_info.get('stream_port', 8889)
    stream_url = f"http://{ip_local}:{stream_port}/cam"
    
    print(f"📹 Proxy stream: {device_id} → {stream_url}")
    
    try:
        resp = requests.get(stream_url, timeout=10)
        html_content = resp.text
        
        html_content = html_content.replace(
            f"http://{ip_local}:{stream_port}",
            f"/api/stream-proxy/{device_id}"
        )
        
        return Response(
            html_content,
            content_type='text/html',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        print(f"❌ Error proxy: {e}")
        return jsonify({'error': str(e), 'url': stream_url}), 500

@app.route('/api/stream-proxy/<device_id>/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def stream_proxy_assets(device_id, subpath):
    """Proxy assets WebRTC"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if device_id not in connected_devices:
        return jsonify({'error': 'No conectado'}), 404
    
    device_info = connected_devices[device_id]
    ip_local = device_info['ip_local']
    stream_port = device_info.get('stream_port', 8889)
    target_url = f"http://{ip_local}:{stream_port}/{subpath}"
    
    try:
        if request.method == 'GET':
            resp = requests.get(target_url, timeout=10)
        elif request.method == 'POST':
            resp = requests.post(
                target_url,
                data=request.get_data(),
                headers={'Content-Type': request.content_type},
                timeout=10
            )
        
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type'),
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    except Exception as e:
        print(f"❌ Error proxy assets: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - DETECCIÓN IA
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'connected_devices': len(connected_devices),
        'backend_version': '4.2 - Nginx Ready',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Análisis IA de grietas"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No imagen'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400
        
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"📥 Procesando: {filename}")
        
        img_original, pred_mask, confidence_map, original_dims = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)
        
        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat()
        }
        
        os.remove(filepath)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN ddd
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v4.2 - NGINX READY")
print("═" * 100)
print(f"📡 URL: https://crackguard.angelproyect.com")
print(f"🤖 IA: UNet++ + {config.ENCODER}")
print(f"⚡ Device: {config.DEVICE}")
print(f"\n📍 Endpoints:")
print(f"   • POST /api/rpi/register")
print(f"   • POST /api/rpi/heartbeat")
print(f"   • POST /api/rpi/photo")
print(f"   • GET  /api/rpi/devices")
print(f"   • POST /api/rpi/streaming/start/<id>")
print(f"   • POST /api/rpi/streaming/stop/<id>")
print(f"   • POST /api/rpi/capture/<id>")
print(f"   • GET  /api/rpi/latest-photo/<id>")
print(f"   • GET  /api/stream/<id>")
print(f"   • POST /api/predict")
print(f"   • GET  /health")
print("═" * 100 + "\n")

if cargar_modelo():
    print("✅ Modelo IA cargado")
else:
    print("⚠️  Backend sin IA")

print("✅ Backend en 0.0.0.0:5000\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)'''





'''

#junior
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CRACKGUARD BACKEND v4.2 - CONTROL TOTAL STREAMING + FOTOS + NGINX
Backend optimizado para Nginx proxy reverso
URL: https://crackguard.angelproyect.com
═══════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response
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
from functools import lru_cache
import warnings
import time
import requests
import threading

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Length"]
    }
})

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB para fotos grandes
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

    ANGLE_TOLERANCE = 12
    MIN_CRACK_LENGTH = 10

    MAX_IMAGE_DIMENSION = 2048
    MAX_GRIETAS_ANALIZAR = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TORCH_THREADS = 4
    CV2_THREADS = 4

    PNG_COMPRESSION = 6
    JPEG_QUALITY = 92
    USE_JPEG_OUTPUT = False

    DEVICE_HEARTBEAT_TIMEOUT = 30
    DEVICE_CHECK_INTERVAL = 10

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

torch.set_num_threads(config.TORCH_THREADS)
torch.set_num_interop_threads(config.TORCH_THREADS)
cv2.setNumThreads(config.CV2_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# VARIABLES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

connected_devices = {}
latest_photos = {}
device_commands = {}
device_lock = threading.Lock()

model = None
model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO IA
# ═══════════════════════════════════════════════════════════════════════════

def cargar_modelo():
    global model, model_loaded

    try:
        print(f"🤖 Cargando modelo UNet++ {config.ENCODER}...")

        if not os.path.exists(config.MODEL_PATH):
            print(f"   ⚠️  Modelo no encontrado: {config.MODEL_PATH}")
            return False

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
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(config.DEVICE)
        model.eval()

        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)

        print(f"   ✅ Modelo cargado en {config.DEVICE}")

        model_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_mejorada(contour):
    if len(contour) < 5:
        return None, "indefinido"

    try:
        moments = cv2.moments(contour)
        if moments['mu20'] - moments['mu02'] != 0:
            angle_moments = 0.5 * np.arctan2(2 * moments['mu11'],
                                            moments['mu20'] - moments['mu02'])
            angle_moments = np.degrees(angle_moments)
        else:
            angle_moments = None

        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy[0], vx[0]) * 180 / np.pi

        angle = angle_moments if angle_moments is not None else angle_fit
        angle = angle % 180
        if angle < 0:
            angle += 180

        tol = config.ANGLE_TOLERANCE

        if angle < tol or angle > (180 - tol):
            tipo = "horizontal"
        elif abs(angle - 90) < tol:
            tipo = "vertical"
        elif abs(angle - 45) < tol:
            tipo = "diagonal"
        elif abs(angle - 135) < tol:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]]

    orientaciones = []

    for idx in top_indices:
        angle, tipo = analizar_orientacion_grieta_mejorada(contours[idx])
        if angle is not None:
            orientaciones.append(tipo)

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
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)

    if diversidad >= 3 and porcentaje_dominante < 0.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    elif tipo_dominante == "horizontal" and porcentaje_dominante > 0.55:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    elif tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    elif tipo_dominante == "diagonal" and porcentaje_dominante > 0.45:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron_general': 'superficial',
            'descripcion_patron': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][::-1]

    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}

    for rank, idx in enumerate(top_indices, 1):
        contour = contours[idx]
        length = longitudes[idx]
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta_mejorada(contour)

        width = area / length if length > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        orientaciones_count[tipo] += 1

        grietas_detalle.append({
            'id': rank,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })

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
# PROCESAMIENTO DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
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

    h_orig, w_orig = img.shape[:2]
    original_dimensions = (w_orig, h_orig)

    if max(h_orig, w_orig) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

    return img_rgb, pred_mask, confidence_map, original_dimensions

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)

    if config.OVERLAY_COLOR == 'red':
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

    if num_contours == 0 or porcentaje_grietas < 0.1:
        return {
            'total_pixeles': int(total_pixeles),
            'pixeles_con_grietas': 0,
            'porcentaje_grietas': 0.0,
            'num_grietas_detectadas': 0,
            'longitud_total_px': 0.0,
            'longitud_promedio_px': 0.0,
            'longitud_maxima_px': 0.0,
            'ancho_promedio_px': 0.0,
            'severidad': "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map.max()),
            'confidence_mean': float(confidence_map.mean()),
            'analisis_morfologico': None
        }

    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes.mean()
    max_length = longitudes.max()
    avg_width = pixeles_positivos / total_length if total_length > 0 else 0

    morfologia = analizar_morfologia_detallada(mask, contours)

    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste

    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Grietas Menores"
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
        'longitud_total_px': round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if config.USE_JPEG_OUTPUT:
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        _, buffer = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION])
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# LIMPIEZA AUTOMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_offline_devices():
    while True:
        time.sleep(config.DEVICE_CHECK_INTERVAL)

        with device_lock:
            now = time.time()
            offline_devices = []

            for device_id, info in list(connected_devices.items()):
                last_seen = info.get('last_seen', 0)
                if now - last_seen > config.DEVICE_HEARTBEAT_TIMEOUT:
                    offline_devices.append(device_id)
                    del connected_devices[device_id]
                    if device_id in device_commands:
                        del device_commands[device_id]

            if offline_devices:
                print(f"🧹 Dispositivos offline: {offline_devices}")

cleanup_thread = threading.Thread(target=cleanup_offline_devices, daemon=True)
cleanup_thread.start()

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/rpi/register', methods=['POST', 'OPTIONS'])
def rpi_register():
    """Raspberry Pi se registra"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        device_type = data.get('type', 'raspberry_pi')
        capabilities = data.get('capabilities', [])
        ip_local = data.get('ip_local', request.remote_addr)
        stream_port = data.get('stream_port', 8889)

        if not device_id:
            return jsonify({'error': 'device_id requerido'}), 400

        with device_lock:
            connected_devices[device_id] = {
                'type': device_type,
                'capabilities': capabilities,
                'ip_local': ip_local,
                'stream_port': stream_port,
                'streaming_active': False,
                'connected_at': datetime.now().isoformat(),
                'last_seen': time.time()
            }

            if device_id not in device_commands:
                device_commands[device_id] = []

        print(f"\n{'='*70}")
        print(f"✅ Raspberry Pi registrado: {device_id}")
        print(f"   IP: {ip_local}")
        print(f"   Stream port: {stream_port}")
        print(f"{'='*70}\n")

        return jsonify({
            'status': 'registered',
            'device_id': device_id,
            'backend_version': '4.2',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error register: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/heartbeat', methods=['POST', 'OPTIONS'])
def rpi_heartbeat():
    """Heartbeat + comandos pendientes"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        streaming_status = data.get('streaming_active', False)

        if not device_id or device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()
            connected_devices[device_id]['streaming_active'] = streaming_status

        commands = []
        with device_lock:
            if device_id in device_commands and device_commands[device_id]:
                commands = device_commands[device_id].copy()
                device_commands[device_id] = []

        return jsonify({
            'status': 'ok',
            'commands': commands,
            'timestamp': time.time()
        }), 200

    except Exception as e:
        print(f"❌ Error heartbeat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/photo', methods=['POST', 'OPTIONS'])
def rpi_upload_photo():
    """Recibir foto desde Raspberry Pi"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        image_b64 = data.get('image')
        metadata = data.get('metadata', {})

        if not device_id or not image_b64:
            return jsonify({'error': 'device_id e image requeridos'}), 400

        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            latest_photos[device_id] = {
                'image': image_b64,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            connected_devices[device_id]['last_seen'] = time.time()

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device_id}_{timestamp_str}.jpg"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)

        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_b64))

        file_size_kb = os.path.getsize(filepath) / 1024

        print(f"\n📸 Foto de {device_id}: {filename} ({file_size_kb:.1f}KB)")

        return jsonify({
            'status': 'saved',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size_kb, 2),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error upload_photo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/devices', methods=['GET', 'OPTIONS'])
def list_devices():
    """Lista dispositivos conectados"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        devices_list = []

        with device_lock:
            for dev_id, info in connected_devices.items():
                device_data = {
                    'device_id': dev_id,
                    'type': info['type'],
                    'ip_local': info['ip_local'],
                    'stream_port': info.get('stream_port', 8889),
                    'streaming_active': info.get('streaming_active', False),
                    'stream_url_direct': f"http://{info['ip_local']}:{info.get('stream_port', 8889)}/cam",
                    'stream_url_proxy': f"/api/stream/{dev_id}",
                    'capabilities': info['capabilities'],
                    'connected_at': info['connected_at'],
                    'last_seen_ago': round(time.time() - info.get('last_seen', 0), 1)
                }

                if dev_id in latest_photos:
                    device_data['has_photo'] = True
                    device_data['last_photo_time'] = latest_photos[dev_id]['timestamp']
                else:
                    device_data['has_photo'] = False

                devices_list.append(device_data)

        return jsonify({
            'devices': devices_list,
            'total': len(devices_list),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error list_devices: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/start/<device_id>', methods=['POST', 'OPTIONS'])
def start_streaming(device_id):
    """🎬 Iniciar streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'start_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🎬 START STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'start_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error start_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/stop/<device_id>', methods=['POST', 'OPTIONS'])
def stop_streaming(device_id):
    """🛑 Detener streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'stop_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🛑 STOP STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'stop_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error stop_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/capture/<device_id>', methods=['POST', 'OPTIONS'])
def send_capture_command(device_id):
    """📸 Capturar foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        data = request.get_json() if request.is_json else {}
        resolution = data.get('resolution', '1920x1080')

        command = {
            'action': 'capture',
            'params': {
                'resolution': resolution,
                'format': 'jpg'
            },
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"📸 CAPTURE → {device_id} ({resolution})")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'capture',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """Ver última foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in latest_photos:
            return jsonify({'error': 'No hay foto'}), 404

        photo_data = latest_photos[device_id]

        return jsonify({
            'success': True,
            'device_id': device_id,
            'image': photo_data['image'],
            'timestamp': photo_data['timestamp'],
            'metadata': photo_data.get('metadata', {})
        }), 200

    except Exception as e:
        print(f"❌ Error latest_photo: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ PROXY STREAMING WebRTC
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/stream/<device_id>', methods=['GET'])
def stream_proxy(device_id):
    """Proxy HTML del MediaMTX"""
    if device_id not in connected_devices:
        return jsonify({'error': 'Dispositivo no conectado'}), 404

    device_info = connected_devices[device_id]
    ip_local = device_info['ip_local']
    stream_port = device_info.get('stream_port', 8889)
    stream_url = f"http://{ip_local}:{stream_port}/cam"

    print(f"📹 Proxy stream: {device_id} → {stream_url}")

    try:
        resp = requests.get(stream_url, timeout=10)
        html_content = resp.text

        html_content = html_content.replace(
            f"http://{ip_local}:{stream_port}",
            f"/api/stream-proxy/{device_id}"
        )

        return Response(
            html_content,
            content_type='text/html',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        print(f"❌ Error proxy: {e}")
        return jsonify({'error': str(e), 'url': stream_url}), 500

@app.route('/api/stream-proxy/<device_id>/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def stream_proxy_assets(device_id, subpath):
    """Proxy assets WebRTC"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if device_id not in connected_devices:
        return jsonify({'error': 'No conectado'}), 404

    device_info = connected_devices[device_id]
    ip_local = device_info['ip_local']
    stream_port = device_info.get('stream_port', 8889)
    target_url = f"http://{ip_local}:{stream_port}/{subpath}"

    try:
        if request.method == 'GET':
            resp = requests.get(target_url, timeout=10)
        elif request.method == 'POST':
            resp = requests.post(
                target_url,
                data=request.get_data(),
                headers={'Content-Type': request.content_type},
                timeout=10
            )

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type'),
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    except Exception as e:
        print(f"❌ Error proxy assets: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - DETECCIÓN IA
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'connected_devices': len(connected_devices),
        'backend_version': '4.2 - Nginx Ready',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Análisis IA de grietas"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400

        use_tta = request.form.get('use_tta', 'true').lower() == 'true'

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"📥 Procesando: {filename}")

        img_original, pred_mask, confidence_map, original_dims = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)

        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat()
        }

        os.remove(filepath)

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN ddd
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v4.2 - NGINX READY")
print("═" * 100)
print(f"📡 URL: https://crackguard.angelproyect.com")
print(f"🤖 IA: UNet++ + {config.ENCODER}")
print(f"⚡ Device: {config.DEVICE}")
print(f"\n📍 Endpoints:")
print(f"   • POST /api/rpi/register")
print(f"   • POST /api/rpi/heartbeat")
print(f"   • POST /api/rpi/photo")
print(f"   • GET  /api/rpi/devices")
print(f"   • POST /api/rpi/streaming/start/<id>")
print(f"   • POST /api/rpi/streaming/stop/<id>")
print(f"   • POST /api/rpi/capture/<id>")
print(f"   • GET  /api/rpi/latest-photo/<id>")
print(f"   • GET  /api/stream/<id>")
print(f"   • POST /api/predict")
print(f"   • GET  /health")
print("═" * 100 + "\n")

if cargar_modelo():
    print("✅ Modelo IA cargado")
else:
    print("⚠️  Backend sin IA")

print("✅ Backend en 0.0.0.0:5000\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
















'''













'''


#jAngel1
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CRACKGUARD BACKEND v4.2 - CONTROL TOTAL STREAMING + FOTOS + NGINX
Backend optimizado para Nginx proxy reverso
URL: https://crackguard.angelproyect.com
═══════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response
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
from functools import lru_cache
import warnings
import time
import requests
import threading

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Length"]
    }
})

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB para fotos grandes
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

    ANGLE_TOLERANCE = 12
    MIN_CRACK_LENGTH = 10

    MAX_IMAGE_DIMENSION = 2048
    MAX_GRIETAS_ANALIZAR = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TORCH_THREADS = 4
    CV2_THREADS = 4

    PNG_COMPRESSION = 6
    JPEG_QUALITY = 92
    USE_JPEG_OUTPUT = False

    DEVICE_HEARTBEAT_TIMEOUT = 30
    DEVICE_CHECK_INTERVAL = 10

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

torch.set_num_threads(config.TORCH_THREADS)
torch.set_num_interop_threads(config.TORCH_THREADS)
cv2.setNumThreads(config.CV2_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# VARIABLES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

connected_devices = {}
latest_photos = {}
device_commands = {}
device_lock = threading.Lock()

model = None
model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO IA
# ═══════════════════════════════════════════════════════════════════════════

def cargar_modelo():
    global model, model_loaded

    try:
        print(f"🤖 Cargando modelo UNet++ {config.ENCODER}...")

        if not os.path.exists(config.MODEL_PATH):
            print(f"   ⚠️  Modelo no encontrado: {config.MODEL_PATH}")
            return False

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
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(config.DEVICE)
        model.eval()

        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)

        print(f"   ✅ Modelo cargado en {config.DEVICE}")

        model_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MORFOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_mejorada(contour):
    if len(contour) < 5:
        return None, "indefinido"

    try:
        moments = cv2.moments(contour)
        if moments['mu20'] - moments['mu02'] != 0:
            angle_moments = 0.5 * np.arctan2(2 * moments['mu11'],
                                            moments['mu20'] - moments['mu02'])
            angle_moments = np.degrees(angle_moments)
        else:
            angle_moments = None

        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy[0], vx[0]) * 180 / np.pi

        angle = angle_moments if angle_moments is not None else angle_fit
        angle = angle % 180
        if angle < 0:
            angle += 180

        tol = config.ANGLE_TOLERANCE

        if angle < tol or angle > (180 - tol):
            tipo = "horizontal"
        elif abs(angle - 90) < tol:
            tipo = "vertical"
        elif abs(angle - 45) < tol:
            tipo = "diagonal"
        elif abs(angle - 135) < tol:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]]

    orientaciones = []

    for idx in top_indices:
        angle, tipo = analizar_orientacion_grieta_mejorada(contours[idx])
        if angle is not None:
            orientaciones.append(tipo)

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
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)

    if diversidad >= 3 and porcentaje_dominante < 0.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    elif tipo_dominante == "horizontal" and porcentaje_dominante > 0.55:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    elif tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    elif tipo_dominante == "diagonal" and porcentaje_dominante > 0.45:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron_general': 'superficial',
            'descripcion_patron': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][::-1]

    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}

    for rank, idx in enumerate(top_indices, 1):
        contour = contours[idx]
        length = longitudes[idx]
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta_mejorada(contour)

        width = area / length if length > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        orientaciones_count[tipo] += 1

        grietas_detalle.append({
            'id': rank,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })

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
# PROCESAMIENTO DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
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

    h_orig, w_orig = img.shape[:2]
    original_dimensions = (w_orig, h_orig)

    if max(h_orig, w_orig) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

    return img_rgb, pred_mask, confidence_map, original_dimensions

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)

    if config.OVERLAY_COLOR == 'red':
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

    if num_contours == 0 or porcentaje_grietas < 0.1:
        return {
            'total_pixeles': int(total_pixeles),
            'pixeles_con_grietas': 0,
            'porcentaje_grietas': 0.0,
            'num_grietas_detectadas': 0,
            'longitud_total_px': 0.0,
            'longitud_promedio_px': 0.0,
            'longitud_maxima_px': 0.0,
            'ancho_promedio_px': 0.0,
            'severidad': "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map.max()),
            'confidence_mean': float(confidence_map.mean()),
            'analisis_morfologico': None
        }

    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes.mean()
    max_length = longitudes.max()
    avg_width = pixeles_positivos / total_length if total_length > 0 else 0

    morfologia = analizar_morfologia_detallada(mask, contours)

    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste

    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Grietas Menores"
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
        'longitud_total_px': round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if config.USE_JPEG_OUTPUT:
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        _, buffer = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION])
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# LIMPIEZA AUTOMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_offline_devices():
    while True:
        time.sleep(config.DEVICE_CHECK_INTERVAL)

        with device_lock:
            now = time.time()
            offline_devices = []

            for device_id, info in list(connected_devices.items()):
                last_seen = info.get('last_seen', 0)
                if now - last_seen > config.DEVICE_HEARTBEAT_TIMEOUT:
                    offline_devices.append(device_id)
                    del connected_devices[device_id]
                    if device_id in device_commands:
                        del device_commands[device_id]

            if offline_devices:
                print(f"🧹 Dispositivos offline: {offline_devices}")

cleanup_thread = threading.Thread(target=cleanup_offline_devices, daemon=True)
cleanup_thread.start()

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/rpi/register', methods=['POST', 'OPTIONS'])
def rpi_register():
    """Raspberry Pi se registra"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        device_type = data.get('type', 'raspberry_pi')
        capabilities = data.get('capabilities', [])
        ip_local = data.get('ip_local', request.remote_addr)
        stream_port = data.get('stream_port', 8889)

        if not device_id:
            return jsonify({'error': 'device_id requerido'}), 400

        with device_lock:
            connected_devices[device_id] = {
                'type': device_type,
                'capabilities': capabilities,
                'ip_local': ip_local,
                'stream_port': stream_port,
                'streaming_active': False,
                'connected_at': datetime.now().isoformat(),
                'last_seen': time.time()
            }

            if device_id not in device_commands:
                device_commands[device_id] = []

        print(f"\n{'='*70}")
        print(f"✅ Raspberry Pi registrado: {device_id}")
        print(f"   IP: {ip_local}")
        print(f"   Stream port: {stream_port}")
        print(f"{'='*70}\n")

        return jsonify({
            'status': 'registered',
            'device_id': device_id,
            'backend_version': '4.2',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error register: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/heartbeat', methods=['POST', 'OPTIONS'])
def rpi_heartbeat():
    """Heartbeat + comandos pendientes"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        streaming_status = data.get('streaming_active', False)

        if not device_id or device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()
            connected_devices[device_id]['streaming_active'] = streaming_status

        commands = []
        with device_lock:
            if device_id in device_commands and device_commands[device_id]:
                commands = device_commands[device_id].copy()
                device_commands[device_id] = []

        return jsonify({
            'status': 'ok',
            'commands': commands,
            'timestamp': time.time()
        }), 200

    except Exception as e:
        print(f"❌ Error heartbeat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/photo', methods=['POST', 'OPTIONS'])
def rpi_upload_photo():
    """Recibir foto desde Raspberry Pi"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        image_b64 = data.get('image')
        metadata = data.get('metadata', {})

        if not device_id or not image_b64:
            return jsonify({'error': 'device_id e image requeridos'}), 400

        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            latest_photos[device_id] = {
                'image': image_b64,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            connected_devices[device_id]['last_seen'] = time.time()

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device_id}_{timestamp_str}.jpg"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)

        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_b64))

        file_size_kb = os.path.getsize(filepath) / 1024

        print(f"\n📸 Foto de {device_id}: {filename} ({file_size_kb:.1f}KB)")

        return jsonify({
            'status': 'saved',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size_kb, 2),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error upload_photo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

 

@app.route('/api/rpi/devices', methods=['GET', 'OPTIONS'])
def list_devices():
    """Lista dispositivos conectados"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        devices_list = []

        with device_lock:
            for dev_id, info in connected_devices.items():
                device_data = {
                    'device_id': dev_id,
                    'type': info['type'],
                    'ip_local': info['ip_local'],
                    'stream_port': info.get('stream_port', 8889),
                    'streaming_active': info.get('streaming_active', False),
                    'stream_url_local': f"http://{info['ip_local']}:{info.get('stream_port', 8889)}/cam",
                    'stream_url_public': info.get('stream_url_public', None),  # ✅ NUEVO
                    'tunnel_type': info.get('tunnel_type', None),  # ✅ NUEVO
                    'stream_url_proxy': f"/api/stream/{dev_id}",
                    'capabilities': info['capabilities'],
                    'connected_at': info['connected_at'],
                    'last_seen_ago': round(time.time() - info.get('last_seen', 0), 1)
                }

                if dev_id in latest_photos:
                    device_data['has_photo'] = True
                    device_data['last_photo_time'] = latest_photos[dev_id]['timestamp']
                else:
                    device_data['has_photo'] = False

                devices_list.append(device_data)

        return jsonify({
            'devices': devices_list,
            'total': len(devices_list),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error list_devices: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500 
    





@app.route('/api/rpi/streaming/start/<device_id>', methods=['POST', 'OPTIONS'])
def start_streaming(device_id):
    """🎬 Iniciar streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'start_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🎬 START STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'start_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error start_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/stop/<device_id>', methods=['POST', 'OPTIONS'])
def stop_streaming(device_id):
    """🛑 Detener streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'stop_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🛑 STOP STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'stop_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error stop_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/capture/<device_id>', methods=['POST', 'OPTIONS'])
def send_capture_command(device_id):
    """📸 Capturar foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        data = request.get_json() if request.is_json else {}
        resolution = data.get('resolution', '1920x1080')

        command = {
            'action': 'capture',
            'params': {
                'resolution': resolution,
                'format': 'jpg'
            },
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"📸 CAPTURE → {device_id} ({resolution})")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'capture',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """Ver última foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in latest_photos:
            return jsonify({'error': 'No hay foto'}), 404

        photo_data = latest_photos[device_id]

        return jsonify({
            'success': True,
            'device_id': device_id,
            'image': photo_data['image'],
            'timestamp': photo_data['timestamp'],
            'metadata': photo_data.get('metadata', {})
        }), 200

    except Exception as e:
        print(f"❌ Error latest_photo: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ PROXY STREAMING WebRTC
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/api/rpi/stream-url', methods=['POST', 'OPTIONS'])
def rpi_stream_url():
    """Recibir URL del Cloudflare Tunnel"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        stream_url = data.get('stream_url')
        tunnel_type = data.get('tunnel_type', 'unknown')

        if not device_id or not stream_url:
            return jsonify({'error': 'device_id y stream_url requeridos'}), 400

        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            connected_devices[device_id]['stream_url_public'] = stream_url
            connected_devices[device_id]['tunnel_type'] = tunnel_type
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"\n🌐 URL pública actualizada: {device_id}")
        print(f"   {stream_url}\n")

        return jsonify({
            'status': 'ok',
            'device_id': device_id,
            'stream_url': stream_url
        }), 200

    except Exception as e:
        print(f"❌ Error stream_url: {e}")
        return jsonify({'error': str(e)}), 500
    



@app.route('/api/stream/<device_id>', methods=['GET'])
def stream_proxy(device_id):
    """Proxy del stream usando URL PÚBLICA del túnel"""
    if device_id not in connected_devices:
        return jsonify({'error': 'Dispositivo no conectado'}), 404

    device_info = connected_devices[device_id]
    stream_url_public = device_info.get('stream_url_public')

    if not stream_url_public:
        return jsonify({
            'error': 'Stream no disponible',
            'hint': 'El RPi aún no envió la URL del túnel',
            'device_id': device_id
        }), 503

    print(f"Proxy stream: {device_id} → {stream_url_public}")

    try:
        # Proxy directo a la URL pública
        resp = requests.get(stream_url_public, timeout=15, stream=True)
        resp.raise_for_status()

        # Reemplazar URLs internas por proxy relativo
        html_content = resp.text
        base_url = stream_url_public.rsplit('/', 1)[0]  # e.g. https://abc.trycloudflare.com

        html_content = html_content.replace(
            base_url,
            f"/api/stream-proxy/{device_id}"
        )

        return Response(
            html_content,
            content_type='text/html',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        print(f"Error proxy stream: {error_msg}")
        return jsonify({
            'error': 'Stream no responde',
            'url': stream_url_public,
            'detail': error_msg
        }), 502
    except Exception as e:
        print(f"Error inesperado: {e}")
        return jsonify({'error': str(e)}), 500
    

    

@app.route('/api/stream-proxy/<device_id>/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def stream_proxy_assets(device_id, subpath):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if device_id not in connected_devices:
        return jsonify({'error': 'No conectado'}), 404

    stream_url_public = connected_devices[device_id].get('stream_url_public')
    if not stream_url_public:
        return jsonify({'error': 'URL pública no disponible'}), 503

    target_url = f"{stream_url_public.rsplit('/', 1)[0]}/{subpath}"

    try:
        if request.method == 'GET':
            resp = requests.get(target_url, timeout=15)
        elif request.method == 'POST':
            resp = requests.post(
                target_url,
                data=request.get_data(),
                headers={'Content-Type': request.content_type},
                timeout=15
            )

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type'),
            headers={
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        print(f"Error proxy asset: {e}")
        return jsonify({'error': str(e)}), 500
# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - DETECCIÓN IA
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'connected_devices': len(connected_devices),
        'backend_version': '4.2 - Nginx Ready',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Análisis IA de grietas"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400

        use_tta = request.form.get('use_tta', 'true').lower() == 'true'

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"📥 Procesando: {filename}")

        img_original, pred_mask, confidence_map, original_dims = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)

        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat()
        }

        os.remove(filepath)

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN ddd
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v4.2 - NGINX READY")
print("═" * 100)
print(f"📡 URL: https://crackguard.angelproyect.com")
print(f"🤖 IA: UNet++ + {config.ENCODER}")
print(f"⚡ Device: {config.DEVICE}")
print(f"\n📍 Endpoints:")
print(f"   • POST /api/rpi/register")
print(f"   • POST /api/rpi/heartbeat")
print(f"   • POST /api/rpi/photo")
print(f"   • GET  /api/rpi/devices")
print(f"   • POST /api/rpi/streaming/start/<id>")
print(f"   • POST /api/rpi/streaming/stop/<id>")
print(f"   • POST /api/rpi/capture/<id>")
print(f"   • GET  /api/rpi/latest-photo/<id>")
print(f"   • GET  /api/stream/<id>")
print(f"   • POST /api/predict")
print(f"   • GET  /health")
print("═" * 100 + "\n")

if cargar_modelo():
    print("✅ Modelo IA cargado")
else:
    print("⚠️  Backend sin IA")

print("✅ Backend en 0.0.0.0:5000\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)'''




#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CRACKGUARD BACKEND v4.3 - FOTOS OPTIMIZADO
Backend optimizado para Nginx proxy reverso
URL: https://crackguard.angelproyect.com
═══════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datetime import datetime, timedelta
import traceback
import base64
from scipy.ndimage import binary_fill_holes
from collections import Counter
from functools import lru_cache
import warnings
import time
import requests
import threading
import glob

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Length"]
    }
})

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model.pth')
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB
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

    ANGLE_TOLERANCE = 12
    MIN_CRACK_LENGTH = 10

    MAX_IMAGE_DIMENSION = 2048
    MAX_GRIETAS_ANALIZAR = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TORCH_THREADS = 4
    CV2_THREADS = 4

    PNG_COMPRESSION = 6
    JPEG_QUALITY = 92
    USE_JPEG_OUTPUT = False

    # 🆕 NUEVOS PARÁMETROS
    DEVICE_HEARTBEAT_TIMEOUT = 30
    DEVICE_CHECK_INTERVAL = 10
    MAX_PHOTOS_PER_DEVICE = 5  # Máximo de fotos por dispositivo
    PHOTO_RETENTION_HOURS = 24  # Borrar fotos más antiguas de 24h
    CLEANUP_INTERVAL = 3600  # Limpiar cada 1 hora

config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
Path(config.UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

torch.set_num_threads(config.TORCH_THREADS)
torch.set_num_interop_threads(config.TORCH_THREADS)
cv2.setNumThreads(config.CV2_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# VARIABLES GLOBALES - 🆕 OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════════════

connected_devices = {}
device_commands = {}
device_lock = threading.Lock()

# 🆕 Solo guardamos metadata, NO la imagen completa
latest_photos_metadata = {}  # {device_id: {'filename': ..., 'timestamp': ..., 'size_kb': ...}}

model = None
model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════
# 🧹 LIMPIEZA AUTOMÁTICA DE FOTOS
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_old_photos():
    """Borrar fotos antiguas y mantener solo las últimas N por dispositivo"""
    while True:
        time.sleep(config.CLEANUP_INTERVAL)
        
        try:
            now = datetime.now()
            retention_time = timedelta(hours=config.PHOTO_RETENTION_HOURS)
            
            # Obtener todas las fotos
            all_photos = glob.glob(os.path.join(config.UPLOAD_FOLDER, '*.jpg'))
            
            # Agrupar por device_id
            device_photos = {}
            for photo in all_photos:
                basename = os.path.basename(photo)
                device_id = basename.split('_')[0]
                
                if device_id not in device_photos:
                    device_photos[device_id] = []
                
                # Obtener timestamp del archivo
                file_time = datetime.fromtimestamp(os.path.getmtime(photo))
                device_photos[device_id].append({
                    'path': photo,
                    'time': file_time
                })
            
            deleted_count = 0
            
            # Procesar cada dispositivo
            for device_id, photos in device_photos.items():
                # Ordenar por fecha (más reciente primero)
                photos.sort(key=lambda x: x['time'], reverse=True)
                
                for idx, photo_info in enumerate(photos):
                    photo_age = now - photo_info['time']
                    
                    # Borrar si:
                    # 1. Es más antigua que PHOTO_RETENTION_HOURS
                    # 2. O si supera MAX_PHOTOS_PER_DEVICE (mantener solo las N más recientes)
                    if photo_age > retention_time or idx >= config.MAX_PHOTOS_PER_DEVICE:
                        try:
                            os.remove(photo_info['path'])
                            deleted_count += 1
                        except Exception as e:
                            print(f"⚠️  Error borrando {photo_info['path']}: {e}")
            
            if deleted_count > 0:
                print(f"🧹 Limpieza: {deleted_count} fotos antiguas eliminadas")
                
        except Exception as e:
            print(f"❌ Error en cleanup_old_photos: {e}")

# Iniciar thread de limpieza
cleanup_photos_thread = threading.Thread(target=cleanup_old_photos, daemon=True)
cleanup_photos_thread.start()

# ═══════════════════════════════════════════════════════════════════════════
# 🧹 LIMPIEZA DE DISPOSITIVOS OFFLINE
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_offline_devices():
    while True:
        time.sleep(config.DEVICE_CHECK_INTERVAL)

        with device_lock:
            now = time.time()
            offline_devices = []

            for device_id, info in list(connected_devices.items()):
                last_seen = info.get('last_seen', 0)
                if now - last_seen > config.DEVICE_HEARTBEAT_TIMEOUT:
                    offline_devices.append(device_id)
                    del connected_devices[device_id]
                    if device_id in device_commands:
                        del device_commands[device_id]

            if offline_devices:
                print(f"🧹 Dispositivos offline: {offline_devices}")

cleanup_thread = threading.Thread(target=cleanup_offline_devices, daemon=True)
cleanup_thread.start()

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR MODELO IA (sin cambios, omitido por brevedad)
# ═══════════════════════════════════════════════════════════════════════════

def cargar_modelo():
    global model, model_loaded
    try:
        print(f"🤖 Cargando modelo UNet++ {config.ENCODER}...")
        if not os.path.exists(config.MODEL_PATH):
            print(f"   ⚠️  Modelo no encontrado: {config.MODEL_PATH}")
            return False

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
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                state_dict = checkpoint['ema_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(config.DEVICE)
        model.eval()

        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)

        print(f"   ✅ Modelo cargado en {config.DEVICE}")
        model_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - RASPBERRY PI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/rpi/register', methods=['POST', 'OPTIONS'])
def rpi_register():
    """Raspberry Pi se registra"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        device_type = data.get('type', 'raspberry_pi')
        capabilities = data.get('capabilities', [])
        ip_local = data.get('ip_local', request.remote_addr)
        stream_port = data.get('stream_port', 8889)

        if not device_id:
            return jsonify({'error': 'device_id requerido'}), 400

        with device_lock:
            connected_devices[device_id] = {
                'type': device_type,
                'capabilities': capabilities,
                'ip_local': ip_local,
                'stream_port': stream_port,
                'streaming_active': False,
                'connected_at': datetime.now().isoformat(),
                'last_seen': time.time()
            }

            if device_id not in device_commands:
                device_commands[device_id] = []

        print(f"\n{'='*70}")
        print(f"✅ Raspberry Pi registrado: {device_id}")
        print(f"   IP: {ip_local}")
        print(f"   Stream port: {stream_port}")
        print(f"{'='*70}\n")

        return jsonify({
            'status': 'registered',
            'device_id': device_id,
            'backend_version': '4.3',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error register: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/heartbeat', methods=['POST', 'OPTIONS'])
def rpi_heartbeat():
    """Heartbeat + comandos pendientes"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        streaming_status = data.get('streaming_active', False)

        if not device_id or device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()
            connected_devices[device_id]['streaming_active'] = streaming_status

        commands = []
        with device_lock:
            if device_id in device_commands and device_commands[device_id]:
                commands = device_commands[device_id].copy()
                device_commands[device_id] = []

        return jsonify({
            'status': 'ok',
            'commands': commands,
            'timestamp': time.time()
        }), 200

    except Exception as e:
        print(f"❌ Error heartbeat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/photo', methods=['POST', 'OPTIONS'])
def rpi_upload_photo():
    """🆕 Recibir foto desde Raspberry Pi - OPTIMIZADO"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        image_b64 = data.get('image')
        metadata = data.get('metadata', {})

        if not device_id or not image_b64:
            return jsonify({'error': 'device_id e image requeridos'}), 400

        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        # Actualizar last_seen
        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()

        # Generar nombre de archivo
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device_id}_{timestamp_str}.jpg"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)

        # Guardar a disco
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_b64))

        file_size_kb = os.path.getsize(filepath) / 1024

        # 🆕 Guardar solo metadata (NO la imagen en RAM)
        latest_photos_metadata[device_id] = {
            'filename': filename,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            'size_kb': round(file_size_kb, 2),
            'metadata': metadata
        }

        print(f"\n📸 Foto de {device_id}: {filename} ({file_size_kb:.1f}KB)")

        return jsonify({
            'status': 'saved',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size_kb, 2),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error upload_photo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """🆕 Ver última foto - devuelve archivo directamente CON HEADERS ANTI-CACHÉ"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in latest_photos_metadata:
            return jsonify({'error': 'No hay foto'}), 404

        photo_info = latest_photos_metadata[device_id]
        filepath = photo_info['filepath']

        if not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404

        # 🆕 Agregar timestamp al ETag para forzar nueva descarga
        file_mtime = os.path.getmtime(filepath)
        etag = f'"{device_id}-{file_mtime}"'

        # Devolver archivo con headers anti-caché
        response = send_file(
            filepath,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=photo_info['filename']
        )
        
        # 🆕 CRÍTICO: Headers para evitar caché del navegador
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['ETag'] = etag
        
        return response

    except Exception as e:
        print(f"❌ Error latest_photo: {e}")
        return jsonify({'error': str(e)}), 500





@app.route('/api/rpi/latest-photo-info/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo_info(device_id):
    """🆕 Obtener solo metadata de la última foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in latest_photos_metadata:
            return jsonify({'error': 'No hay foto'}), 404

        photo_info = latest_photos_metadata[device_id].copy()
        photo_info.pop('filepath', None)  # No exponer ruta interna

        return jsonify({
            'success': True,
            'device_id': device_id,
            **photo_info
        }), 200

    except Exception as e:
        print(f"❌ Error latest_photo_info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/devices', methods=['GET', 'OPTIONS'])
def list_devices():
    """Lista dispositivos conectados"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        devices_list = []

        with device_lock:
            for dev_id, info in connected_devices.items():
                device_data = {
                    'device_id': dev_id,
                    'type': info['type'],
                    'ip_local': info['ip_local'],
                    'stream_port': info.get('stream_port', 8889),
                    'streaming_active': info.get('streaming_active', False),
                    'stream_url_local': f"http://{info['ip_local']}:{info.get('stream_port', 8889)}/cam",
                    'stream_url_public': info.get('stream_url_public', None),
                    'tunnel_type': info.get('tunnel_type', None),
                    'stream_url_proxy': f"/api/stream/{dev_id}",
                    'capabilities': info['capabilities'],
                    'connected_at': info['connected_at'],
                    'last_seen_ago': round(time.time() - info.get('last_seen', 0), 1)
                }

                # 🆕 Verificar si hay foto (desde metadata)
                if dev_id in latest_photos_metadata:
                    device_data['has_photo'] = True
                    device_data['last_photo_time'] = latest_photos_metadata[dev_id]['timestamp']
                    device_data['last_photo_url'] = f"/api/rpi/latest-photo/{dev_id}"
                else:
                    device_data['has_photo'] = False

                devices_list.append(device_data)

        return jsonify({
            'devices': devices_list,
            'total': len(devices_list),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error list_devices: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/start/<device_id>', methods=['POST', 'OPTIONS'])
def start_streaming(device_id):
    """🎬 Iniciar streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'start_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🎬 START STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'start_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error start_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/streaming/stop/<device_id>', methods=['POST', 'OPTIONS'])
def stop_streaming(device_id):
    """🛑 Detener streaming"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        command = {
            'action': 'stop_streaming',
            'params': {},
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"🛑 STOP STREAMING → {device_id}")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'stop_streaming',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error stop_streaming: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/capture/<device_id>', methods=['POST', 'OPTIONS'])
def send_capture_command(device_id):
    """📸 Capturar foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        data = request.get_json() if request.is_json else {}
        resolution = data.get('resolution', '1920x1080')

        command = {
            'action': 'capture',
            'params': {
                'resolution': resolution,
                'format': 'jpg'
            },
            'timestamp': time.time()
        }

        with device_lock:
            device_commands[device_id].append(command)
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"📸 CAPTURE → {device_id} ({resolution})")

        return jsonify({
            'status': 'command_queued',
            'device_id': device_id,
            'action': 'capture',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/stream-url', methods=['POST', 'OPTIONS'])
def rpi_stream_url():
    """Recibir URL del Cloudflare Tunnel"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        stream_url = data.get('stream_url')
        tunnel_type = data.get('tunnel_type', 'unknown')

        if not device_id or not stream_url:
            return jsonify({'error': 'device_id y stream_url requeridos'}), 400

        if device_id not in connected_devices:
            return jsonify({'error': 'No registrado'}), 404

        with device_lock:
            connected_devices[device_id]['stream_url_public'] = stream_url
            connected_devices[device_id]['tunnel_type'] = tunnel_type
            connected_devices[device_id]['last_seen'] = time.time()

        print(f"\n🌐 URL pública actualizada: {device_id}")
        print(f"   {stream_url}\n")

        return jsonify({
            'status': 'ok',
            'device_id': device_id,
            'stream_url': stream_url
        }), 200

    except Exception as e:
        print(f"❌ Error stream_url: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/<device_id>', methods=['GET'])
def stream_proxy(device_id):
    """Proxy del stream usando URL PÚBLICA del túnel"""
    if device_id not in connected_devices:
        return jsonify({'error': 'Dispositivo no conectado'}), 404

    device_info = connected_devices[device_id]
    stream_url_public = device_info.get('stream_url_public')

    if not stream_url_public:
        return jsonify({
            'error': 'Stream no disponible',
            'hint': 'El RPi aún no envió la URL del túnel',
            'device_id': device_id
        }), 503

    try:
        resp = requests.get(stream_url_public, timeout=15, stream=True)
        resp.raise_for_status()

        html_content = resp.text
        base_url = stream_url_public.rsplit('/', 1)[0]

        html_content = html_content.replace(
            base_url,
            f"/api/stream-proxy/{device_id}"
        )

        return Response(
            html_content,
            content_type='text/html',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Stream no responde',
            'url': stream_url_public,
            'detail': str(e)
        }), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-proxy/<device_id>/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def stream_proxy_assets(device_id, subpath):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if device_id not in connected_devices:
        return jsonify({'error': 'No conectado'}), 404

    stream_url_public = connected_devices[device_id].get('stream_url_public')
    if not stream_url_public:
        return jsonify({'error': 'URL pública no disponible'}), 503

    target_url = f"{stream_url_public.rsplit('/', 1)[0]}/{subpath}"

    try:
        if request.method == 'GET':
            resp = requests.get(target_url, timeout=15)
        elif request.method == 'POST':
            resp = requests.post(
                target_url,
                data=request.get_data(),
                headers={'Content-Type': request.content_type},
                timeout=15
            )

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type'),
            headers={'Access-Control-Allow-Origin': '*'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - DETECCIÓN IA (omitido por brevedad, sin cambios)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'connected_devices': len(connected_devices),
        'backend_version': '4.3 - Fotos optimizado',
        'photos_in_memory': len(latest_photos_metadata),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Análisis IA de grietas"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if not model_loaded:
            return jsonify({'error': 'Modelo no cargado'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400

        use_tta = request.form.get('use_tta', 'true').lower() == 'true'

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"📥 Procesando: {filename}")

        img_original, pred_mask, confidence_map, original_dims = procesar_imagen(filepath, use_tta)
        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)

        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay': imagen_a_base64(overlay),
            'timestamp': datetime.now().isoformat()
        }

        os.remove(filepath)

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PROCESAMIENTO IA (mantener las originales)
# ═══════════════════════════════════════════════════════════════════════════

def analizar_orientacion_grieta_mejorada(contour):
    if len(contour) < 5:
        return None, "indefinido"

    try:
        moments = cv2.moments(contour)
        if moments['mu20'] - moments['mu02'] != 0:
            angle_moments = 0.5 * np.arctan2(2 * moments['mu11'],
                                            moments['mu20'] - moments['mu02'])
            angle_moments = np.degrees(angle_moments)
        else:
            angle_moments = None

        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_fit = np.arctan2(vy[0], vx[0]) * 180 / np.pi

        angle = angle_moments if angle_moments is not None else angle_fit
        angle = angle % 180
        if angle < 0:
            angle += 180

        tol = config.ANGLE_TOLERANCE

        if angle < tol or angle > (180 - tol):
            tipo = "horizontal"
        elif abs(angle - 90) < tol:
            tipo = "vertical"
        elif abs(angle - 45) < tol:
            tipo = "diagonal"
        elif abs(angle - 135) < tol:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron': 'superficial',
            'descripcion': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]]

    orientaciones = []

    for idx in top_indices:
        angle, tipo = analizar_orientacion_grieta_mejorada(contours[idx])
        if angle is not None:
            orientaciones.append(tipo)

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
    porcentaje_dominante = tipo_counts[tipo_dominante] / len(orientaciones)
    diversidad = len(tipo_counts)

    if diversidad >= 3 and porcentaje_dominante < 0.5:
        return {
            'patron': 'ramificada_mapa',
            'descripcion': 'Patrón ramificado - Contracción térmica',
            'causa_probable': 'Cambios térmicos, secado del material',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }
    elif tipo_dominante == "horizontal" and porcentaje_dominante > 0.55:
        return {
            'patron': 'horizontal',
            'descripcion': 'Grietas predominantemente horizontales',
            'causa_probable': 'Flexión estructural, presión lateral',
            'severidad_ajuste': 1.1,
            'recomendacion': 'Inspección de muros y cimentación'
        }
    elif tipo_dominante == "vertical" and porcentaje_dominante > 0.55:
        return {
            'patron': 'vertical',
            'descripcion': 'Grietas verticales - ⚠️ CRÍTICO',
            'causa_probable': 'Cargas verticales excesivas, asentamientos',
            'severidad_ajuste': 1.3,
            'recomendacion': '⚠️ Inspección estructural URGENTE'
        }
    elif tipo_dominante == "diagonal" and porcentaje_dominante > 0.45:
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

    longitudes = np.array([cv2.arcLength(c, False) for c in contours])
    indices_validos = np.where(longitudes >= config.MIN_CRACK_LENGTH)[0]

    if len(indices_validos) == 0:
        return {
            'patron_general': 'superficial',
            'descripcion_patron': 'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }

    top_indices = indices_validos[np.argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][::-1]

    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0}

    for rank, idx in enumerate(top_indices, 1):
        contour = contours[idx]
        length = longitudes[idx]
        area = cv2.contourArea(contour)
        angle, tipo = analizar_orientacion_grieta_mejorada(contour)

        width = area / length if length > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        orientaciones_count[tipo] += 1

        grietas_detalle.append({
            'id': rank,
            'longitud_px': round(float(length), 2),
            'area_px': int(area),
            'ancho_promedio_px': round(float(width), 2),
            'angulo_grados': round(angle, 1) if angle else None,
            'orientacion': tipo,
            'aspect_ratio': round(float(aspect_ratio), 2),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })

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

@lru_cache(maxsize=1)
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

    h_orig, w_orig = img.shape[:2]
    original_dimensions = (w_orig, h_orig)

    if max(h_orig, w_orig) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

    return img_rgb, pred_mask, confidence_map, original_dimensions

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)

    if config.OVERLAY_COLOR == 'red':
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

    if num_contours == 0 or porcentaje_grietas < 0.1:
        return {
            'total_pixeles': int(total_pixeles),
            'pixeles_con_grietas': 0,
            'porcentaje_grietas': 0.0,
            'num_grietas_detectadas': 0,
            'longitud_total_px': 0.0,
            'longitud_promedio_px': 0.0,
            'longitud_maxima_px': 0.0,
            'ancho_promedio_px': 0.0,
            'severidad': "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map.max()),
            'confidence_mean': float(confidence_map.mean()),
            'analisis_morfologico': None
        }

    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes.mean()
    max_length = longitudes.max()
    avg_width = pixeles_positivos / total_length if total_length > 0 else 0

    morfologia = analizar_morfologia_detallada(mask, contours)

    severidad_ajuste = morfologia['severidad_ajuste']
    porcentaje_ajustado = porcentaje_grietas * severidad_ajuste

    if porcentaje_ajustado < 1:
        severidad = "Baja"
        estado = "Grietas Menores"
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
        'longitud_total_px': round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max': float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def imagen_a_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if config.USE_JPEG_OUTPUT:
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        _, buffer = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION])
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v4.3 - FOTOS OPTIMIZADO")
print("═" * 100)
print(f"📡 URL: https://crackguard.angelproyect.com")
print(f"🤖 IA: UNet++ + {config.ENCODER}")
print(f"⚡ Device: {config.DEVICE}")
print(f"🧹 Limpieza automática: Cada {config.CLEANUP_INTERVAL}s")
print(f"📸 Max fotos/device: {config.MAX_PHOTOS_PER_DEVICE}")
print(f"⏰ Retención fotos: {config.PHOTO_RETENTION_HOURS}h")
print(f"\n📍 Endpoints:")
print(f"   • POST /api/rpi/register")
print(f"   • POST /api/rpi/heartbeat")
print(f"   • POST /api/rpi/photo (optimizado)")
print(f"   • GET  /api/rpi/latest-photo/<id> (archivo directo)")
print(f"   • GET  /api/rpi/latest-photo-info/<id> (solo metadata)")
print(f"   • GET  /api/rpi/devices")
print(f"   • POST /api/rpi/streaming/start/<id>")
print(f"   • POST /api/rpi/streaming/stop/<id>")
print(f"   • POST /api/rpi/capture/<id>")
print(f"   • GET  /api/stream/<id>")
print(f"   • POST /api/predict")
print(f"   • GET  /health")
print("═" * 100 + "\n")

if cargar_modelo():
    print("✅ Modelo IA cargado")
else:
    print("⚠️  Backend sin IA")

print("✅ Backend en 0.0.0.0:5000\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)