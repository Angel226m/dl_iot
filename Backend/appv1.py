#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CRACKGUARD BACKEND v6.0 - DUAL MODEL (MULTIPART OPTIMIZADO) 🚀
✅ Soporte para 2 modelos (best_model. pth y modelof.pth)
✅ Selección dinámica de modelo por request
✅ Subida de fotos con multipart/form-data (3× más rápido que base64)
✅ Sin túnel Cloudflare (FRP directo)
✅ Limpieza automática de fotos antiguas
URL: https://crackguard.angelproyect.com
═══════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import threading
import glob
import hashlib
import timm

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
        "expose_headers": ["Content-Type", "Content-Length", "ETag"]
    }
})

class Config:
    UPLOAD_FOLDER = '/app/uploads'
    
    # 🆕 DUAL MODEL PATHS
    MODEL_1_PATH = os.getenv('MODEL_1_PATH', '/app/model/best_model.pth')
    MODEL_2_PATH = os.getenv('MODEL_2_PATH', '/app/model/modelof.pth')
    
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

    # MODELO 1 (UnetPlusPlus + EfficientNet-B8)
    MODEL_1_ARCHITECTURE = 'UnetPlusPlus'
    MODEL_1_ENCODER = 'timm-efficientnet-b8'
    MODEL_1_TARGET_SIZE = 640
    MODEL_1_MEAN = [0.485, 0.456, 0.406]
    MODEL_1_STD = [0.229, 0.224, 0.225]

    # MODELO 2 (UNet++ + ConvNeXt-Base + CBAM)
    MODEL_2_ARCHITECTURE = 'UNetPlusPlus+CBAM'
    MODEL_2_ENCODER = 'convnext_base.fb_in22k_ft_in1k'
    MODEL_2_TARGET_SIZE = 640
    MODEL_2_MEAN = [0.485, 0.456, 0.406]
    MODEL_2_STD = [0.229, 0.224, 0.225]
    MODEL_2_USE_CBAM = True

    THRESHOLD = 0.5
    MIN_CRACK_COVERAGE = 0.25

    # TTA MODELO 1
    MODEL_1_USE_TTA = True
    MODEL_1_TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']

    # MODELO 2 (SIN TTA)
    MODEL_2_USE_TTA = False

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

    # OPTIMIZACIÓN MULTIPART
    DEVICE_HEARTBEAT_TIMEOUT = 30
    DEVICE_CHECK_INTERVAL = 10
    MAX_PHOTOS_PER_DEVICE = 5
    PHOTO_RETENTION_HOURS = 24
    CLEANUP_INTERVAL = 3600

    ENABLE_ETAG = True

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
device_commands = {}
device_lock = threading.Lock()

latest_photos_metadata = {}

# 🆕 DUAL MODELS
model_1 = None
model_2 = None
model_1_loaded = False
model_2_loaded = False

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 ARQUITECTURA MODELO 2 (CBAM + UNet++)
# ═══════════════════════════════════════════════════════════════════════════

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if use_attention and config.MODEL_2_USE_CBAM:  
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        return self.attention(self.conv(x))


class UNetPlusPlusModel(nn.Module):
    """UNet++ COMPLETO con CBAM (Modelo 2)"""
    
    def __init__(self, encoder_name='convnext_base.fb_in22k_ft_in1k', deep_supervision=False):
        super(UNetPlusPlusModel, self).__init__()
        self.deep_supervision = deep_supervision

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            feats = self.encoder(dummy)
            enc_channels = [f.shape[1] for f in feats]
            del dummy, feats

        nb_filter = [64, enc_channels[0], enc_channels[1], enc_channels[2], enc_channels[3]]

        self.adapt1 = nn.Conv2d(enc_channels[0], nb_filter[1], 1)
        self.adapt2 = nn.Conv2d(enc_channels[1], nb_filter[2], 1)
        self.adapt3 = nn.Conv2d(enc_channels[2], nb_filter[3], 1)
        self.adapt4 = nn.Conv2d(enc_channels[3], nb_filter[4], 1)

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], use_attention=True)
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], use_attention=True)
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], use_attention=True)
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], use_attention=True)

        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], use_attention=False)
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], use_attention=False)
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], use_attention=False)

        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], use_attention=False)
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], use_attention=False)

        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], use_attention=False)

        self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_channels[3], 1)
        )

    def _upsample_like(self, src, target):
        return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        orig_size = x.shape[2:]

        X0_0 = self.stem(x)
        encoder_outputs = self.encoder(x)
        X1_0 = self.adapt1(encoder_outputs[0])
        X2_0 = self.adapt2(encoder_outputs[1])
        X3_0 = self.adapt3(encoder_outputs[2])
        X4_0 = self.adapt4(encoder_outputs[3])

        cls_out = self.classification_head(encoder_outputs[3])

        X0_1 = self.conv0_1(torch.cat([X0_0, self._upsample_like(X1_0, X0_0)], 1))
        X1_1 = self.conv1_1(torch.cat([X1_0, self._upsample_like(X2_0, X1_0)], 1))
        X2_1 = self.conv2_1(torch.cat([X2_0, self._upsample_like(X3_0, X2_0)], 1))
        X3_1 = self.conv3_1(torch.cat([X3_0, self._upsample_like(X4_0, X3_0)], 1))

        X0_2 = self.conv0_2(torch.cat([X0_0, X0_1, self._upsample_like(X1_1, X0_0)], 1))
        X1_2 = self.conv1_2(torch.cat([X1_0, X1_1, self._upsample_like(X2_1, X1_0)], 1))
        X2_2 = self.conv2_2(torch.cat([X2_0, X2_1, self._upsample_like(X3_1, X2_0)], 1))

        X0_3 = self.conv0_3(torch.cat([X0_0, X0_1, X0_2, self._upsample_like(X1_2, X0_0)], 1))
        X1_3 = self.conv1_3(torch.cat([X1_0, X1_1, X1_2, self._upsample_like(X2_2, X1_0)], 1))

        X0_4 = self.conv0_4(torch.cat([X0_0, X0_1, X0_2, X0_3, self._upsample_like(X1_3, X0_0)], 1))

        if self.deep_supervision:
            output1 = F.interpolate(self.final1(X0_1), size=orig_size, mode='bilinear', align_corners=False)
            output2 = F.interpolate(self.final2(X0_2), size=orig_size, mode='bilinear', align_corners=False)
            output3 = F.interpolate(self.final3(X0_3), size=orig_size, mode='bilinear', align_corners=False)
            output4 = F.interpolate(self.final4(X0_4), size=orig_size, mode='bilinear', align_corners=False)
            return [output1, output2, output3, output4], cls_out
        else:
            output = F.interpolate(self.final4(X0_4), size=orig_size, mode='bilinear', align_corners=False)
            return output, cls_out

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

            all_photos = glob.glob(os.path.join(config.UPLOAD_FOLDER, '*.jpg'))
            all_photos.extend(glob.glob(os.path.join(config.UPLOAD_FOLDER, '*.jpeg')))
            all_photos.extend(glob.glob(os.path. join(config.UPLOAD_FOLDER, '*.png')))

            device_photos = {}
            temp_files = []

            for photo in all_photos:
                basename = os.path.basename(photo)

                if basename.startswith('overlay_') or basename.startswith('input_'):
                    file_time = datetime.fromtimestamp(os.path.getmtime(photo))
                    temp_files.append({
                        'path': photo,
                        'time': file_time
                    })
                    continue

                parts = basename.split('_')
                if len(parts) >= 2:
                    device_id = parts[0]
                else:
                    continue

                if device_id not in device_photos:
                    device_photos[device_id] = []

                file_time = datetime.fromtimestamp(os.path.getmtime(photo))
                device_photos[device_id].append({
                    'path': photo,
                    'time': file_time
                })

            deleted_count = 0

            for device_id, photos in device_photos.items():
                photos.sort(key=lambda x: x['time'], reverse=True)

                for idx, photo_info in enumerate(photos):
                    photo_age = now - photo_info['time']

                    if photo_age > retention_time or idx >= config.MAX_PHOTOS_PER_DEVICE:
                        try:
                            os.remove(photo_info['path'])
                            deleted_count += 1

                            if device_id in latest_photos_metadata:
                                if latest_photos_metadata[device_id].get('filepath') == photo_info['path']: 
                                    del latest_photos_metadata[device_id]
                        except Exception as e:
                            print(f"⚠️ Error borrando {photo_info['path']}: {e}")

            temp_retention = timedelta(hours=1)
            for temp_file in temp_files: 
                file_age = now - temp_file['time']
                if file_age > temp_retention:
                    try:
                        os.remove(temp_file['path'])
                        deleted_count += 1
                    except Exception as e: 
                        print(f"⚠️ Error borrando temp {temp_file['path']}: {e}")

            if deleted_count > 0:
                print(f"🧹 Limpieza:  {deleted_count} archivos eliminados")

        except Exception as e:
            print(f"❌ Error en cleanup_old_photos: {e}")

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
# 🔧 UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def calculate_etag(filepath):
    """Generar ETag basado en contenido del archivo"""
    if not config.ENABLE_ETAG:
        return None

    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f'"{file_hash}"'
    except: 
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 🆕 CARGAR MODELOS DUALES
# ═══════════════════════════════════════════════════════════════════════════

def cargar_modelo_1():
    """Cargar Modelo 1: UNet++ + EfficientNet-B8 (CON TTA)"""
    global model_1, model_1_loaded
    try:
        print(f"🤖 Cargando MODELO 1: {config.MODEL_1_ARCHITECTURE} + {config.MODEL_1_ENCODER}...")
        if not os.path.exists(config.MODEL_1_PATH):
            print(f"   ⚠️ Modelo 1 no encontrado: {config.MODEL_1_PATH}")
            return False

        model_1 = smp.UnetPlusPlus(
            encoder_name=config.MODEL_1_ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        checkpoint = torch.load(config.MODEL_1_PATH, map_location=config.DEVICE, weights_only=False)

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

        model_1.load_state_dict(state_dict, strict=False)
        model_1 = model_1.to(config.DEVICE)
        model_1.eval()

        if config.DEVICE.type == 'cpu':
            torch.set_grad_enabled(False)

        print(f"   ✅ Modelo 1 cargado en {config.DEVICE}")
        model_1_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error cargando modelo 1: {e}")
        traceback.print_exc()
        return False


def cargar_modelo_2():
    """Cargar Modelo 2: UNet++ + ConvNeXt-Base + CBAM (SIN TTA)"""
    global model_2, model_2_loaded
    try:
        print(f"🤖 Cargando MODELO 2: {config.MODEL_2_ARCHITECTURE} + {config.MODEL_2_ENCODER}...")
        if not os.path.exists(config.MODEL_2_PATH):
            print(f"   ⚠️ Modelo 2 no encontrado: {config.MODEL_2_PATH}")
            return False

        model_2 = UNetPlusPlusModel(config.MODEL_2_ENCODER, deep_supervision=False)

        checkpoint = torch.load(config.MODEL_2_PATH, map_location=config.DEVICE, weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint

        missing, unexpected = model_2.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"   ⚠️ Claves faltantes:  {len(missing)}")
        if unexpected:
            print(f"   ⚠️ Claves inesperadas:  {len(unexpected)}")

        model_2 = model_2.to(config.DEVICE)
        model_2.eval()

        if config.DEVICE.type == 'cpu': 
            torch.set_grad_enabled(False)

        print(f"   ✅ Modelo 2 cargado en {config.DEVICE}")
        model_2_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error cargando modelo 2: {e}")
        traceback.print_exc()
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
        tunnel_type = data.get('tunnel', 'frp')

        if not device_id:
            return jsonify({'error': 'device_id requerido'}), 400

        with device_lock:
            connected_devices[device_id] = {
                'type': device_type,
                'capabilities': capabilities,
                'ip_local': ip_local,
                'stream_port': stream_port,
                'streaming_active': False,
                'tunnel_type': tunnel_type,
                'connected_at': datetime.now().isoformat(),
                'last_seen': time.time()
            }

            if device_id not in device_commands:
                device_commands[device_id] = []

        print(f"\n{'='*70}")
        print(f"✅ Raspberry Pi registrado: {device_id}")
        print(f"   IP:  {ip_local}")
        print(f"   Stream port: {stream_port}")
        print(f"   Túnel: {tunnel_type}")
        print(f"{'='*70}\n")

        return jsonify({
            'status': 'registered',
            'device_id':  device_id,
            'backend_version': '6.0',
            'upload_method': 'multipart',
            'available_models': ['model_1', 'model_2'],
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error register:  {e}")
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
            return jsonify({'error':  'No registrado'}), 404

        with device_lock: 
            connected_devices[device_id]['last_seen'] = time.time()
            connected_devices[device_id]['streaming_active'] = streaming_status

        commands = []
        with device_lock:
            if device_id in device_commands and device_commands[device_id]: 
                commands = device_commands[device_id].copy()
                device_commands[device_id] = []

        return jsonify({
            'status':  'ok',
            'commands': commands,
            'timestamp': time.time()
        }), 200

    except Exception as e: 
        print(f"❌ Error heartbeat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/photo', methods=['POST', 'OPTIONS'])
def rpi_upload_photo():
    """🚀 Recibir foto con multipart/form-data (3× MÁS RÁPIDO)"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        device_id = request.form.get('device_id') or request.headers.get('X-Device-ID')

        if not device_id: 
            return jsonify({'error': 'device_id requerido'}), 400

        if device_id not in connected_devices: 
            return jsonify({'error':  'Dispositivo no registrado'}), 404

        if 'image' not in request.files:
            return jsonify({'error': 'No se envió imagen'}), 400

        file = request.files['image']

        if not file or file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400

        if '.' in file.filename:
            original_ext = file.filename.rsplit('.',1)[1].lower()
        else:
            original_ext = 'jpg'

        if original_ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({'error': f'Formato no permitido: {original_ext}'}), 400

        with device_lock:
            connected_devices[device_id]['last_seen'] = time.time()

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device_id}_{timestamp_str}.{original_ext}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)

        file.save(filepath)

        file_size_kb = os.path.getsize(filepath) / 1024

        metadata = {}
        if 'resolution' in request.form:
            metadata['resolution'] = request.form.get('resolution')
        if 'color_profile' in request.form:
            metadata['color_profile'] = request.form.get('color_profile')

        etag = calculate_etag(filepath)

        latest_photos_metadata[device_id] = {
            'filename': filename,
            'filepath':  filepath,
            'timestamp': datetime.now().isoformat(),
            'size_kb': round(file_size_kb, 2),
            'metadata': metadata,
            'etag': etag
        }

        print(f"\n📸 Foto de {device_id}:  {filename} ({file_size_kb:.1f}KB) [MULTIPART]")

        return jsonify({
            'status': 'saved',
            'device_id': device_id,
            'filename': filename,
            'size_kb': round(file_size_kb, 2),
            'upload_method': 'multipart',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error upload_photo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo(device_id):
    """Ver última foto - devuelve archivo directamente CON ETAG"""
    if request.method == 'OPTIONS': 
        return jsonify({'status':  'ok'}), 200

    try:
        if device_id not in latest_photos_metadata:
            return jsonify({'error': 'No hay foto'}), 404

        photo_info = latest_photos_metadata[device_id]
        filepath = photo_info['filepath']

        if not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404

        client_etag = request.headers.get('If-None-Match')
        server_etag = photo_info.get('etag')

        if config.ENABLE_ETAG and client_etag and server_etag: 
            if client_etag == server_etag:
                return Response(status=304)

        response = send_file(
            filepath,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=photo_info['filename']
        )

        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        if server_etag:
            response.headers['ETag'] = server_etag

        return response

    except Exception as e:
        print(f"❌ Error latest_photo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpi/latest-photo-info/<device_id>', methods=['GET', 'OPTIONS'])
def get_latest_photo_info(device_id):
    """Obtener solo metadata de la última foto"""
    if request.method == 'OPTIONS': 
        return jsonify({'status':  'ok'}), 200

    try:
        if device_id not in latest_photos_metadata:
            return jsonify({'error': 'No hay foto'}), 404

        photo_info = latest_photos_metadata[device_id].copy()
        photo_info.pop('filepath', None)

        return jsonify({
            'success': True,
            'device_id': device_id,
            **photo_info
        }), 200

    except Exception as e: 
        print(f"❌ Error latest_photo_info: {e}")
        return jsonify({'error':  str(e)}), 500

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
                    'tunnel_type': info.get('tunnel_type', 'frp'),
                    'capabilities': info['capabilities'],
                    'connected_at': info['connected_at'],
                    'last_seen_ago': round(time.time() - info.get('last_seen', 0), 1)
                }

                if dev_id in latest_photos_metadata:
                    device_data['has_photo'] = True
                    device_data['last_photo_time'] = latest_photos_metadata[dev_id]['timestamp']
                    device_data['last_photo_url'] = f"/api/rpi/latest-photo/{dev_id}"
                    device_data['last_photo_size_kb'] = latest_photos_metadata[dev_id]['size_kb']
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
    """Iniciar streaming"""
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
    """Detener streaming"""
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
    """Capturar foto"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if device_id not in connected_devices:
            return jsonify({'error': 'Dispositivo no encontrado'}), 404

        data = request.get_json() if request.is_json else {}
        resolution = data.get('resolution', '1920x1080')

        command = {
            'action': 'capture',
            'params':  {
                'resolution': resolution,
                'format': 'jpg'
            },
            'timestamp':  time.time()
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
    """Recibir URL del túnel FRP"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        stream_url = data.get('stream_url')
        streaming_active = data.get('streaming_active', True)
        tunnel_type = data.get('tunnel_type', 'frp')

        if not device_id: 
            return jsonify({'error':  'device_id requerido'}), 400

        if device_id not in connected_devices: 
            return jsonify({'error':  'No registrado'}), 404

        with device_lock:
            if stream_url:
                connected_devices[device_id]['stream_url_public'] = stream_url
                connected_devices[device_id]['tunnel_type'] = tunnel_type
            connected_devices[device_id]['streaming_active'] = streaming_active
            connected_devices[device_id]['last_seen'] = time.time()

        if stream_url:
            print(f"\n🌐 URL pública actualizada: {device_id}")
            print(f"   {stream_url} ({tunnel_type})\n")
        else:
            print(f"🛑 Streaming detenido:  {device_id}")

        return jsonify({
            'status':  'ok',
            'device_id':  device_id,
            'streaming_active': streaming_active
        }), 200

    except Exception as e:
        print(f"❌ Error stream_url: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ✅ API REST - DETECCIÓN IA (DUAL MODEL)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_1_loaded': model_1_loaded,
        'model_2_loaded':  model_2_loaded,
        'connected_devices': len(connected_devices),
        'backend_version': '6.0',
        'upload_method': 'multipart',
        'photos_stored': len(latest_photos_metadata),
        'available_models': {
            'model_1': {
                'loaded': model_1_loaded,
                'architecture': config.MODEL_1_ARCHITECTURE,
                'encoder': config.MODEL_1_ENCODER,
                'tta': config.MODEL_1_USE_TTA
            },
            'model_2': {
                'loaded': model_2_loaded,
                'architecture': config.MODEL_2_ARCHITECTURE,
                'encoder':  config.MODEL_2_ENCODER,
                'tta': config.MODEL_2_USE_TTA
            }
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """🆕 Análisis IA de grietas - DUAL MODEL SUPPORT"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        # 🆕 SELECCIÓN DE MODELO
        selected_model = request.form.get('model', 'model_1')  # Por defecto modelo 1
        
        if selected_model not in ['model_1', 'model_2']:
            return jsonify({'error': 'Modelo inválido.Usar: model_1 o model_2'}), 400

        if selected_model == 'model_1' and not model_1_loaded: 
            return jsonify({'error':  'Modelo 1 no cargado'}), 503
        
        if selected_model == 'model_2' and not model_2_loaded:
            return jsonify({'error': 'Modelo 2 no cargado'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre vacío'}), 400

        use_tta = request.form. get('use_tta', 'true').lower() == 'true'

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"input_{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"📥 Procesando IA con {selected_model. upper()}:  {filename}")

        # 🆕 PROCESAR CON MODELO SELECCIONADO
        if selected_model == 'model_1':
            img_original, pred_mask, confidence_map, original_dims = procesar_imagen_modelo1(
                filepath, use_tta
            )
            model_info = {
                'model': 'model_1',
                'architecture': config.MODEL_1_ARCHITECTURE,
                'encoder':  config.MODEL_1_ENCODER,
                'tta_usado': use_tta,
                'tta_transforms': len(config.MODEL_1_TTA_TRANSFORMS) if use_tta else 1
            }
        else: 
            img_original, pred_mask, confidence_map, original_dims = procesar_imagen_modelo2(
                filepath
            )
            model_info = {
                'model': 'model_2',
                'architecture': config.MODEL_2_ARCHITECTURE,
                'encoder': config.MODEL_2_ENCODER,
                'tta_usado':  False,
                'cbam_enabled': config.MODEL_2_USE_CBAM
            }

        overlay = crear_overlay(img_original, pred_mask)
        metricas = calcular_metricas(pred_mask, confidence_map)

        overlay_filename, overlay_filepath = guardar_imagen_temp(overlay, prefix='overlay')

        response_data = {
            'success': True,
            'metricas': metricas,
            'imagen_overlay_url': f'/api/result-image/{overlay_filename}',
            'timestamp': datetime.now().isoformat(),
            'procesamiento':  {
                **model_info,
                'threshold': config.THRESHOLD,
                'target_size': config.MODEL_1_TARGET_SIZE if selected_model == 'model_1' else config.MODEL_2_TARGET_SIZE,
                'original_dimensions': {
                    'width': original_dims[0],
                    'height':  original_dims[1]
                }
            }
        }

        os.remove(filepath)

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error predict: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/result-image/<filename>', methods=['GET', 'OPTIONS'])
def get_result_image(filename):
    """Servir imagen procesada (overlay) - MULTIPART"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if not filename.startswith('overlay_') or '. .' in filename:
            return jsonify({'error': 'Nombre de archivo inválido'}), 400

        filepath = os.path. join(config.UPLOAD_FOLDER, filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404

        etag = calculate_etag(filepath)

        client_etag = request.headers. get('If-None-Match')
        if config.ENABLE_ETAG and client_etag and etag: 
            if client_etag == etag:
                return Response(status=304)

        response = send_file(
            filepath,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=filename
        )

        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        if etag:
            response.headers['ETag'] = etag

        return response

    except Exception as e:
        print(f"❌ Error result_image: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES IA - COMUNES
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
            'descripcion':  'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion':  'Monitoreo periódico'
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
            'descripcion':  'Grietas superficiales menores',
            'causa_probable': 'Desgaste superficial',
            'severidad_ajuste': 0.8,
            'recomendacion': 'Monitoreo periódico'
        }

    tipo_counts = Counter(orientaciones)
    tipo_dominante = tipo_counts. most_common(1)[0][0]
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
            'descripcion':  'Grietas diagonales - ⚠️ MUY CRÍTICO',
            'causa_probable': 'Esfuerzos cortantes, movimiento del terreno',
            'severidad_ajuste': 1.4,
            'recomendacion': '🔴 Evaluación estructural CRÍTICA'
        }
    elif diversidad >= 2:
        return {
            'patron': 'mixto',
            'descripcion':  'Patrón mixto de agrietamiento',
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
            'severidad_ajuste':  0.8,
            'recomendacion': 'Monitoreo periódico',
            'distribucion_orientaciones': {
                "horizontal": 0, "vertical": 0, "diagonal": 0, "irregular": 0
            },
            'num_grietas_analizadas': 0,
            'grietas_principales': []
        }

    top_indices = indices_validos[np. argsort(longitudes[indices_validos])[-config.MAX_GRIETAS_ANALIZAR:]][: :-1]

    grietas_detalle = []
    orientaciones_count = {"horizontal": 0, "vertical":  0, "diagonal": 0, "irregular": 0}

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
        'patron_general':  patron_info['patron'],
        'descripcion_patron': patron_info['descripcion'],
        'causa_probable': patron_info['causa_probable'],
        'severidad_ajuste': patron_info['severidad_ajuste'],
        'recomendacion': patron_info. get('recomendacion', 'Monitoreo'),
        'distribucion_orientaciones': orientaciones_count,
        'num_grietas_analizadas': len(grietas_detalle),
        'grietas_principales': grietas_detalle[: 5]
    }

@lru_cache(maxsize=2)
def get_transform(target_size, mean, std):
    """Transform con caché para cada modelo"""
    return A. Compose([
        A. Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC),
        A. Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def advanced_postprocess(mask):
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

    if config.USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

    if config.USE_CONNECTED_COMPONENTS:
        mask_binary = (mask_np > 0. 5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

        cleaned_mask = np.zeros_like(mask_np)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_SIZE:
                cleaned_mask[labels == i] = mask_np[labels == i]

        mask_np = cleaned_mask

    mask_binary = (mask_np > 0.5).astype(bool)
    mask_filled = binary_fill_holes(mask_binary)

    return mask_filled. astype(np.float32)

def crear_overlay(img_original, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    color_mask = np.zeros_like(img_original)

    if config.OVERLAY_COLOR == 'red':
        color_mask[: , :, 0] = mask_binary * 255

    overlay = cv2.addWeighted(img_original, 1. 0, color_mask, config.OVERLAY_ALPHA, 0)
    return overlay

def calcular_metricas(mask, confidence_map):
    mask_binary = (mask > 127).astype(np.uint8)

    total_pixeles = mask. size
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
            'severidad':  "Sin Grietas",
            'estado': "Sin Grietas Significativas",
            'confianza': 95.0,
            'confidence_max': float(confidence_map. max()),
            'confidence_mean': float(confidence_map. mean()),
            'analisis_morfologico': None
        }

    longitudes = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    total_length = longitudes.sum()
    avg_length = longitudes. mean()
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
        'longitud_total_px':  round(float(total_length), 2),
        'longitud_promedio_px': round(float(avg_length), 2),
        'longitud_maxima_px': round(float(max_length), 2),
        'ancho_promedio_px': round(float(avg_width), 2),
        'severidad': severidad,
        'estado': estado,
        'confianza': round(confianza, 1),
        'confidence_max':  float(confidence_map.max()),
        'confidence_mean': float(confidence_map.mean()),
        'analisis_morfologico': morfologia
    }

def guardar_imagen_temp(img_rgb, prefix='overlay'):
    """Guardar imagen procesada temporalmente y devolver ruta"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)

    cv2.imwrite(filepath, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, config. JPEG_QUALITY])

    return filename, filepath

# ═══════════════════════════════════════════════════════════════════════════
# 🆕 FUNCIONES IA - MODELO 1 (CON TTA)
# ═══════════════════════════════════════════════════════════════════════════

def predict_with_tta_modelo1(model, img_tensor):
    """TTA para Modelo 1"""
    preds = []

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        preds.append(pred)

    if 'hflip' in config.MODEL_1_TTA_TRANSFORMS:
        img_hflip = torch.flip(img_tensor, dims=[3])
        with torch.no_grad():
            pred = model(img_hflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[3])
            preds.append(pred)

    if 'vflip' in config. MODEL_1_TTA_TRANSFORMS:
        img_vflip = torch.flip(img_tensor, dims=[2])
        with torch.no_grad():
            pred = model(img_vflip)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, dims=[2])
            preds.append(pred)

    if 'rotate90' in config.MODEL_1_TTA_TRANSFORMS:
        img_rot90 = torch.rot90(img_tensor, k=1, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot90)
            pred = torch.sigmoid(pred)
            pred = torch. rot90(pred, k=-1, dims=[2, 3])
            preds.append(pred)

    if 'rotate180' in config.MODEL_1_TTA_TRANSFORMS: 
        img_rot180 = torch.rot90(img_tensor, k=2, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot180)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-2, dims=[2, 3])
            preds.append(pred)

    if 'rotate270' in config.MODEL_1_TTA_TRANSFORMS:
        img_rot270 = torch.rot90(img_tensor, k=3, dims=[2, 3])
        with torch.no_grad():
            pred = model(img_rot270)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, k=-3, dims=[2, 3])
            preds.append(pred)

    return torch.stack(preds).mean(dim=0)


def procesar_imagen_modelo1(image_path, use_tta=True):
    """Procesamiento con MODELO 1 (UNet++ + EfficientNet-B8 + TTA)"""
    if not model_1_loaded:
        raise RuntimeError("Modelo 1 no cargado")

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
    original_size = (img. shape[1], img.shape[0])

    transform = get_transform(
        config.MODEL_1_TARGET_SIZE,
        tuple(config.MODEL_1_MEAN),
        tuple(config.MODEL_1_STD)
    )
    img_tensor = transform(image=img_rgb)['image']. unsqueeze(0).to(config. DEVICE)

    if use_tta and config.MODEL_1_USE_TTA:
        pred = predict_with_tta_modelo1(model_1, img_tensor)
    else:
        with torch.no_grad():
            pred = model_1(img_tensor)
            pred = torch.sigmoid(pred)

    confidence_map = pred.cpu().numpy()[0, 0]
    confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
    confidence_map = advanced_postprocess(torch.from_numpy(confidence_map))
    pred_mask = (confidence_map > config. THRESHOLD).astype(np.uint8) * 255

    return img_rgb, pred_mask, confidence_map, original_dimensions

# ═══════════════════════════════════════════════════════════════════════════
# 🆕 FUNCIONES IA - MODELO 2 (SIN TTA, CON CBAM)
# ═══════════════════════════════════════════════════════════════════════════

def procesar_imagen_modelo2(image_path):
    """Procesamiento con MODELO 2 (UNet++ + ConvNeXt-Base + CBAM - SIN TTA)"""
    if not model_2_loaded: 
        raise RuntimeError("Modelo 2 no cargado")

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

    transform = get_transform(
        config.MODEL_2_TARGET_SIZE,
        tuple(config.MODEL_2_MEAN),
        tuple(config.MODEL_2_STD)
    )
    img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(config.DEVICE)

    # SIN TTA - inferencia directa
    with torch.no_grad():
        output, cls_output = model_2(img_tensor)
        
        # Si deep_supervision está activo, tomar la última salida
        if isinstance(output, list):
            output = output[-1]
        
        confidence_map_tensor = torch.sigmoid(output)
        cls_prob = torch.sigmoid(cls_output)

    confidence_map = confidence_map_tensor.cpu().numpy()[0, 0]
    confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
    confidence_map = advanced_postprocess(torch. from_numpy(confidence_map))
    pred_mask = (confidence_map > config.THRESHOLD).astype(np.uint8) * 255

    return img_rgb, pred_mask, confidence_map, original_dimensions

# ═══════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("🚀 CRACKGUARD BACKEND v6.0 - DUAL MODEL SUPPORT")
print("═" * 100)
print(f"📡 URL: https://crackguard.angelproyect.com")
print(f"⚡ Device: {config.DEVICE}")
print(f"📤 Upload:  multipart/form-data (3× más rápido que base64)")
print(f"🔒 Túnel:  FRP directo (sin Cloudflare)")
print(f"🧹 Limpieza automática: Cada {config.CLEANUP_INTERVAL}s")
print(f"📸 Max fotos/device: {config.MAX_PHOTOS_PER_DEVICE}")
print(f"⏰ Retención fotos: {config.PHOTO_RETENTION_HOURS}h")
print(f"\n🤖 MODELOS DISPONIBLES:")
print(f"   • MODELO 1: {config.MODEL_1_ARCHITECTURE} + {config.MODEL_1_ENCODER}")
print(f"     └─ Path: {config.MODEL_1_PATH}")
print(f"     └─ TTA: {'Activado' if config.MODEL_1_USE_TTA else 'Desactivado'}")
print(f"   • MODELO 2: {config.MODEL_2_ARCHITECTURE} + {config.MODEL_2_ENCODER}")
print(f"     └─ Path: {config.MODEL_2_PATH}")
print(f"     └─ TTA:  {'Activado' if config. MODEL_2_USE_TTA else 'Desactivado'}")
print(f"     └─ CBAM: {'Activado' if config.MODEL_2_USE_CBAM else 'Desactivado'}")
print(f"\n📍 Endpoints:")
print(f"   • POST /api/rpi/register")
print(f"   • POST /api/rpi/heartbeat")
print(f"   • POST /api/rpi/photo (multipart optimizado) 🚀")
print(f"   • GET  /api/rpi/latest-photo/<id> (con ETag)")
print(f"   • GET  /api/rpi/latest-photo-info/<id>")
print(f"   • GET  /api/rpi/devices")
print(f"   • POST /api/rpi/streaming/start/<id>")
print(f"   • POST /api/rpi/streaming/stop/<id>")
print(f"   • POST /api/rpi/capture/<id>")
print(f"   • POST /api/predict (multipart) 🆕 [model=model_1|model_2]")
print(f"   • GET  /health")
print("═" * 100 + "\n")

# Cargar ambos modelos
print("🔄 Cargando modelos...")
model_1_status = cargar_modelo_1()
model_2_status = cargar_modelo_2()

if model_1_status: 
    print("✅ Modelo 1 cargado correctamente")
else:
    print("⚠️  Modelo 1 no disponible")

if model_2_status:
    print("✅ Modelo 2 cargado correctamente")
else:
    print("⚠️  Modelo 2 no disponible")

if not model_1_status and not model_2_status:
    print("⚠️  Backend sin modelos IA (solo funciones de dispositivos)")
else:
    print(f"✅ Backend con {sum([model_1_status, model_2_status])} modelo(s) activo(s)")

print("\n✅ Backend corriendo en 0.0.0.0:5000\n")
print("═" * 100)
print("📖 USO DE MODELOS:")
print("   Para usar Modelo 1 (CON TTA):")
print("     POST /api/predict")
print("     FormData: image=<file>, model=model_1, use_tta=true")
print("")
print("   Para usar Modelo 2 (SIN TTA, CON CBAM):")
print("     POST /api/predict")
print("     FormData:  image=<file>, model=model_2")
print("═" * 100 + "\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)