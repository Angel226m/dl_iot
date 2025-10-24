"""
═══════════════════════════════════════════════════════════════════════════════
🚀 API BACKEND CRACKGUARD - DETECCIÓN DE GRIETAS
Sistema de inferencia con SegFormer B5
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

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

class Config:
    # Directorios
    UPLOAD_FOLDER = '/app/uploads'
    RESULTS_FOLDER = '/app/results'
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model_epoch003_dice0.6205.pth')
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # Modelo
    TARGET_SIZE = 640
    MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    THRESHOLD = 0.5
    
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
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        traceback.print_exc()
        raise

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

def predecir(img_tensor):
    """Realiza predicción"""
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

def post_procesar(mask, threshold=0.5):
    """Post-procesamiento de máscara"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    result = np.zeros_like(mask_binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 10:
            result[labels == i] = 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return result.astype(np.float32)

def calcular_metricas(mask, threshold=0.5):
    """Calcula métricas de la predicción"""
    mask_binary = (mask > threshold).astype(np.uint8)
    
    total_pixeles = mask.size
    pixeles_positivos = mask_binary.sum()
    porcentaje_grietas = (pixeles_positivos / total_pixeles) * 100
    
    num_grietas, _, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    num_grietas -= 1
    
    if num_grietas > 0:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_grietas + 1)]
        area_promedio = float(np.mean(areas))
        area_max = float(np.max(areas))
    else:
        area_promedio = 0
        area_max = 0
    
    # Determinar severidad
    if porcentaje_grietas < 1:
        severidad = "Baja"
        estado = "Sin Grietas"
    elif porcentaje_grietas < 5:
        severidad = "Baja"
        estado = "Grietas Menores"
    elif porcentaje_grietas < 15:
        severidad = "Media"
        estado = "Grietas Moderadas"
    else:
        severidad = "Alta"
        estado = "Grietas Severas"
    
    confianza = min(95.0, 85.0 + (porcentaje_grietas * 0.5))
    
    return {
        'total_pixeles': int(total_pixeles),
        'pixeles_con_grietas': int(pixeles_positivos),
        'porcentaje_grietas': round(float(porcentaje_grietas), 2),
        'num_grietas_detectadas': int(num_grietas),
        'area_promedio_grieta': round(area_promedio, 2),
        'area_max_grieta': round(area_max, 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1)
    }

def crear_visualizacion(img_original, mask, threshold=0.5):
    """Crea visualización overlay"""
    h, w = img_original.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask_resized > threshold).astype(np.uint8) * 255
    
    mask_colored = cv2.applyColorMap(mask_binary, cv2.COLORMAP_HOT)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    overlay = img_original.copy()
    alpha = 0.5
    overlay[mask_binary > 127] = (overlay[mask_binary > 127] * (1 - alpha) + 
                                   mask_colored[mask_binary > 127] * alpha).astype(np.uint8)
    
    return overlay

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
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint de predicción"""
    try:
        # Validar archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido'}), 400
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Procesar
        img_tensor, img_original = procesar_imagen(filepath)
        mask = predecir(img_tensor)
        mask_procesada = post_procesar(mask, config.THRESHOLD)
        
        # Calcular métricas
        metricas = calcular_metricas(mask_procesada, config.THRESHOLD)
        
        # Crear visualización
        overlay = crear_visualizacion(img_original, mask_procesada, config.THRESHOLD)
        
        # Guardar resultado
        result_filename = f"result_{filename}"
        result_path = os.path.join(config.RESULTS_FOLDER, result_filename)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, overlay_bgr)
        
        # Eliminar archivo original
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'metricas': metricas,
            'result_image': f'/api/results/{result_filename}',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
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

@app.route('/api/status', methods=['GET'])
def status():
    """Estado del sistema"""
    return jsonify({
        'system': 'CrackGuard Backend',
        'version': '1.0.0',
        'model': 'SegFormer B5',
        'device': str(config.DEVICE),
        'model_loaded': model is not None,
        'threshold': config.THRESHOLD,
        'target_size': config.TARGET_SIZE,
        'timestamp': datetime.now().isoformat()
    }), 200

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 80)
    print("🚀 CRACKGUARD BACKEND API")
    print("═" * 80)
    
    cargar_modelo()
    
    print("\n✅ Servidor listo")
    print("═" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)