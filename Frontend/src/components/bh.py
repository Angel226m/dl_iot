"""
╔═══════════════════════════════════════════════════════════════════╗
║  CRACK SEGMENTATION ULTIMATE 2025 v3.2 - VERSIÓN CORREGIDA       ║
║  Fixes: SAM+AMP GradScaler, MixUp, Boundary F1, autocast         ║
║  Mejoras: Optimizaciones críticas para 0.95+ Dice                ║
╚═══════════════════════════════════════════════════════════════════╝
"""

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os, gc, json, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from collections import defaultdict
import random
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════
#                        CONFIG
# ═════════════════════════════════════════════════════════════════
class Config:
    BASE_DATASET_DIR = '/content/drive/MyDrive/dataset_640_full_masks23'
    EXP_DIR = f"/content/drive/MyDrive/experiments_segmentation/CrackFinal_{datetime.now():%Y%m%d_%H%M%S}"

    # Progressive Sizing
    IMG_SIZE_PHASE1 = 160
    IMG_SIZE_PHASE2 = 320
    IMG_SIZE_PHASE3 = 640

    BATCH_SIZE_160 = 24
    BATCH_SIZE_320 = 16
    BATCH_SIZE_640 = 8

    ACCUM_STEPS = 3
    NUM_WORKERS = 2

    EPOCHS = 60
    PATIENCE = 10

    # Augmentations
    USE_MIXUP = True
    USE_CUTMIX = True
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.25
    CUTMIX_PROB = 0.25
    AUG_START_EPOCH = 5

    # Model Features
    USE_DUAL_ATTENTION = True
    USE_CBAM = True
    USE_DEEP_SUPERVISION = True

    # Optimizer
    USE_SAM = True
    SAM_RHO = 0.05
    LEARNING_RATE_160 = 1e-4
    LEARNING_RATE_320 = 3e-4
    LEARNING_RATE_640 = 5e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_0 = 10
    T_mult = 2
    eta_min = 1e-7

    # Training Features
    USE_EMA = True
    EMA_DECAY = 0.996
    USE_OHEM = True
    OHEM_RATIO = 0.25
    USE_TTA = True
    TTA_MODES = 4
    LABEL_SMOOTHING = 0.05
    GRAD_CLIP_NORM = 2.0

    # System
    SEED = 2025
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FP16 = True
    SAVE_TOP_K = 3

config = Config()

# Crear directorios
for subdir in ['models', 'logs', 'checkpoints', 'visualizations']:
    os.makedirs(f"{config.EXP_DIR}/{subdir}", exist_ok=True)

# Seed
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ═════════════════════════════════════════════════════════════════
#                      LOGGER
# ═════════════════════════════════════════════════════════════════
class MetricsLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / 'training.log'
        self.metrics_file = self.log_dir / 'metrics.json'
        self.best_models = []
        self.history = {'train': [], 'val': [], 'test': []}

    def log(self, message, print_console=True):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        if print_console:
            print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

    def update_history(self, epoch, phase, metrics):
        self.history[phase].append({'epoch': epoch, **metrics})
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def update_best_models(self, epoch, dice, iou, model_path):
        self.best_models.append({'epoch': epoch, 'dice': dice, 'iou': iou, 'path': model_path})
        self.best_models = sorted(self.best_models, key=lambda x: x['dice'], reverse=True)[:config.SAVE_TOP_K]

logger = MetricsLogger(f"{config.EXP_DIR}/logs")

# ═════════════════════════════════════════════════════════════════
#          SAM OPTIMIZER (FIX: Compatible con AMP)
# ═════════════════════════════════════════════════════════════════
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization - Versión corregida para AMP"""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Primer paso: subir a la cima"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Segundo paso: bajar desde la cima"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Paso estándar del optimizador (para compatibilidad)"""
        self.base_optimizer.step(closure)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# ═════════════════════════════════════════════════════════════════
#          EMA
# ═════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=0.996):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ═════════════════════════════════════════════════════════════════
#          PROGRESSIVE SIZING
# ═════════════════════════════════════════════════════════════════
def get_current_img_size(epoch):
    if epoch <= 3:
        return config.IMG_SIZE_PHASE1
    elif epoch <= 6:
        return config.IMG_SIZE_PHASE2
    else:
        return config.IMG_SIZE_PHASE3

def get_current_lr(epoch):
    size = get_current_img_size(epoch)
    if size == 160:
        return config.LEARNING_RATE_160
    elif size == 320:
        return config.LEARNING_RATE_320
    else:
        return config.LEARNING_RATE_640

def get_batch_size(epoch):
    size = get_current_img_size(epoch)
    if size == 160:
        return config.BATCH_SIZE_160
    elif size == 320:
        return config.BATCH_SIZE_320
    else:
        return config.BATCH_SIZE_640

# ═════════════════════════════════════════════════════════════════
#          AUGMENTATIONS
# ═════════════════════════════════════════════════════════════════
def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1)
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 40), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=5, p=1)
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ═════════════════════════════════════════════════════════════════
#          MIXUP & CUTMIX
# ═════════════════════════════════════════════════════════════════
def mixup_data(images, masks, has_cracks, alpha=0.4):
    """MixUp - usar OR lógico para clasificación"""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_masks = lam * masks + (1 - lam) * masks[index]
    mixed_has_cracks = torch.max(has_cracks, has_cracks[index])
    
    return mixed_images, mixed_masks, mixed_has_cracks

def cutmix_data(images, masks, has_cracks, alpha=1.0):
    """CutMix - Clasificación basada en área"""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    _, _, H, W = images.shape
    cut_rat = np.sqrt(1.- lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    images_cutmix = images.clone()
    masks_cutmix = masks.clone()
    
    images_cutmix[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    masks_cutmix[:, :, bby1:bby2, bbx1:bbx2] = masks[index, :, bby1:bby2, bbx1:bbx2]
    has_cracks_cutmix = torch.max(has_cracks, has_cracks[index])
    
    return images_cutmix, masks_cutmix, has_cracks_cutmix

# ═════════════════════════════════════════════════════════════════
#          DATASET
# ═════════════════════════════════════════════════════════════════
class CrackDatasetBalanced(Dataset):
    def __init__(self, json_path, img_dir, transform):
        self.img_dir = Path(img_dir)
        self.transform = transform

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        annotations = defaultdict(list)
        for ann in data['annotations']:
            annotations[ann['image_id']].extend(ann['segmentation'])

        self.samples = []
        for img_id, img_info in self.images.items():
            img_path = self.img_dir / img_info['file_name']
            if not img_path.exists():
                continue

            has_cracks = (img_id in annotations and len(annotations[img_id]) > 0)
            is_negative_flag = img_info.get('is_negative', False)

            sample = {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'polygons': annotations[img_id] if has_cracks else [],
                'has_cracks': has_cracks and not is_negative_flag
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def get_stats(self):
        pos = sum(1 for s in self.samples if s['has_cracks'])
        return {'total': len(self.samples), 'with_cracks': pos, 'without_cracks': len(self.samples) - pos}

    def update_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.img_dir / sample['file_name']
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        mask = np.zeros((orig_h, orig_w), dtype=np.float32)

        if sample['has_cracks']:
            for poly in sample['polygons']:
                if len(poly) >= 6:
                    pts = np.array(poly).reshape(-1, 2)
                    pts = np.clip(pts, 0, max(orig_w, orig_h) - 1).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1.0)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        mask = mask.unsqueeze(0).float()
        return img, mask, torch.tensor(sample['has_cracks'], dtype=torch.float32)

# ═════════════════════════════════════════════════════════════════
#          DUAL ATTENTION MODULE
# ═════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class DualAttention(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(DualAttention, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# ═════════════════════════════════════════════════════════════════
#          UNET++
# ═════════════════════════════════════════════════════════════════
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
        if use_attention and config.USE_DUAL_ATTENTION:
            self.attention = DualAttention(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        return self.attention(self.conv(x))

class UNetPlusPlusModel(nn.Module):
    def __init__(self, encoder_name='convnext_base.fb_in22k_ft_in1k', deep_supervision=True):
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
            pretrained=True, 
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
            nn.Dropout(0.3),
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

# ═════════════════════════════════════════════════════════════════
#          LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()

class LovaszHingeLoss(nn.Module):
    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.- intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        losses = []
        for p, t in zip(pred, target):
            errors = (p.view(-1) - t.view(-1)).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            grad = self.lovasz_grad(t.view(-1)[perm])
            losses.append(torch.dot(errors_sorted, grad))
        return torch.stack(losses).mean()

class BoundaryLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(BoundaryLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).clamp(self.eps, 1 - self.eps)
        target = target.clamp(self.eps, 1 - self.eps)
        
        loss_x = F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :], 
                           target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_y = F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], 
                           target[:, :, :, 1:] - target[:, :, :, :-1])
        return (loss_x + loss_y) / 2

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.focal_tversky = FocalTverskyLoss()
        self.lovasz = LovaszHingeLoss()
        self.boundary = BoundaryLoss()
        
        self.seg_weights = {
            'focal_tversky': 0.35,
            'lovasz': 0.30,
            'boundary': 0.25,
            'bce': 0.10
        }
        
        self.seg_budget = 0.85
        self.cls_budget = 0.15

    def _compute_seg_loss(self, pred, target):
        smooth = config.LABEL_SMOOTHING
        target_smooth = target * (1 - smooth) + smooth / 2

        losses = {
            'focal_tversky': self.focal_tversky(pred, target),
            'lovasz': self.lovasz(pred, target),
            'boundary': self.boundary(pred, target),
            'bce': F.binary_cross_entropy_with_logits(pred, target_smooth)
        }

        for k, v in losses.items():
            if torch.isnan(v) or torch.isinf(v):
                losses[k] = torch.tensor(0.0, device=v.device)

        total = sum(losses[k] * self.seg_weights[k] for k in losses)
        return total, losses

    def forward(self, pred, target, cls_pred=None, has_cracks=None):
        loss_dict = {}

        if isinstance(pred, list):
            ds_weights = [0.10, 0.20, 0.30, 0.40]
            seg_loss_total = 0
            for i, p in enumerate(pred):
                seg_loss_i, losses_i = self._compute_seg_loss(p, target)
                seg_loss_total += ds_weights[i] * seg_loss_i
                if i == 0:
                    loss_dict.update(losses_i)
        else:
            seg_loss_total, losses_i = self._compute_seg_loss(pred, target)
            loss_dict.update(losses_i)

        total_loss = self.seg_budget * seg_loss_total

        if cls_pred is not None and has_cracks is not None:
            cls_target = has_cracks.float().unsqueeze(1) if has_cracks.dim() == 1 else has_cracks
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target)
            if not (torch.isnan(cls_loss) or torch.isinf(cls_loss)):
                total_loss += self.cls_budget * cls_loss
                loss_dict['cls'] = cls_loss

        return total_loss, loss_dict

class OHEMComboLoss(ComboLoss):
    """Combo Loss con OHEM"""
    def __init__(self, ratio=0.25):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, pred, target, cls_pred=None, has_cracks=None):
        total_loss, loss_dict = super().forward(pred, target, cls_pred, has_cracks)
        
        if config.USE_OHEM and self.training:
            final_pred = pred[-1] if isinstance(pred, list) else pred
            
            pixel_loss = F.binary_cross_entropy_with_logits(
                final_pred, target, reduction='none'
            ).view(-1)
            
            k = int(self.ratio * pixel_loss.numel())
            hard_pixels, _ = torch.topk(pixel_loss, k)
            
            ohem_loss = hard_pixels.mean()
            total_loss = 0.7 * total_loss + 0.3 * ohem_loss
            loss_dict['ohem'] = ohem_loss
        
        return total_loss, loss_dict

# ═════════════════════════════════════════════════════════════════
#          TTA
# ═════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, image):
    """TTA con 4 transformaciones"""
    preds = []
    
    out, cls_out = model(image)
    if isinstance(out, list):
        out = out[-1]
    preds.append(torch.sigmoid(out))
    
    if config.TTA_MODES >= 2:
        out_hflip, _ = model(torch.flip(image, dims=[3]))
        if isinstance(out_hflip, list):
            out_hflip = out_hflip[-1]
        preds.append(torch.flip(torch.sigmoid(out_hflip), dims=[3]))
    
    if config.TTA_MODES >= 3:
        out_vflip, _ = model(torch.flip(image, dims=[2]))
        if isinstance(out_vflip, list):
            out_vflip = out_vflip[-1]
        preds.append(torch.flip(torch.sigmoid(out_vflip), dims=[2]))
    
    if config.TTA_MODES >= 4:
        out_rot90, _ = model(torch.rot90(image, k=1, dims=[2, 3]))
        if isinstance(out_rot90, list):
            out_rot90 = out_rot90[-1]
        preds.append(torch.rot90(torch.sigmoid(out_rot90), k=-1, dims=[2, 3]))
    
    return torch.stack(preds).mean(dim=0), cls_out

# ═════════════════════════════════════════════════════════════════
#          MÉTRICAS
# ═════════════════════════════════════════════════════════════════
def compute_metrics(pred, target, threshold=0.5):
    """Métricas extendidas con Boundary F1 corregido"""
    pred_binary = (pred > threshold).float()

    tp = (pred_binary * target).sum(dim=(2, 3))
    fp = (pred_binary * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - pred_binary) * target).sum(dim=(2, 3))
    tn = ((1 - pred_binary) * (1 - target)).sum(dim=(2, 3))

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    specificity = (tn + 1e-7) / (tn + fp + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    boundary_f1 = compute_boundary_f1_fixed(pred_binary, target)

    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'specificity': specificity.mean().item(),
        'f1': f1.mean().item(),
        'accuracy': accuracy.mean().item(),
        'boundary_f1': boundary_f1
    }

def compute_boundary_f1_fixed(pred, target, dilation=2):
    """Boundary F1 correctamente calculado"""
    kernel = torch.ones(1, 1, dilation*2+1, dilation*2+1, device=pred.device) / ((dilation*2+1)**2)
    
    pred_dilated = (F.conv2d(pred.float(), kernel, padding=dilation) > 0.1).float()
    target_dilated = (F.conv2d(target.float(), kernel, padding=dilation) > 0.1).float()
    
    pred_boundary = ((pred_dilated > 0) & (pred == 0)).float()
    target_boundary = ((target_dilated > 0) & (target == 0)).float()
    
    tp = (pred_boundary * target_boundary).sum()
    fp = (pred_boundary * (1 - target_boundary)).sum()
    fn = ((1 - pred_boundary) * target_boundary).sum()
    
    boundary_f1 = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    return boundary_f1.item()

# ═════════════════════════════════════════════════════════════════
#          TRAINING LOOP (FIX: SAM + AMP correctamente implementado)
# ═════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, epoch, ema, use_sam=False):
    model.train()
    running_metrics = defaultdict(list)

    use_aug_this_epoch = epoch >= config.AUG_START_EPOCH

    pbar = tqdm(
        enumerate(loader), 
        total=len(loader),
        desc=f"🔥 E{epoch:03d} ({get_current_img_size(epoch)}px) {'AUG' if use_aug_this_epoch else 'BASE'}",
        bar_format="{desc} {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )

    for batch_idx, (images, masks, has_cracks) in pbar:
        images = images.to(config.DEVICE, non_blocking=True)
        masks = masks.to(config.DEVICE, non_blocking=True)
        has_cracks = has_cracks.to(config.DEVICE, non_blocking=True)

        # Augmentaciones avanzadas
        if use_aug_this_epoch:
            aug_choice = random.random()
            if config.USE_MIXUP and aug_choice < config.MIXUP_PROB:
                images, masks, has_cracks = mixup_data(images, masks, has_cracks, config.MIXUP_ALPHA)
            elif config.USE_CUTMIX and aug_choice < (config.MIXUP_PROB + config.CUTMIX_PROB):
                images, masks, has_cracks = cutmix_data(images, masks, has_cracks, config.CUTMIX_ALPHA)

        # ═══════════════════════════════════════════════════════════
        # FIX: SAM + AMP correctamente implementado
        # El problema era que scaler.unscale_() se llamaba dos veces
        # sin scaler.update() entre medio.
        # 
        # Solución: Usar scaler.step() que internamente hace unscale
        # o manejar manualmente el estado del scaler
        # ═══════════════════════════════════════════════════════════
        if use_sam and config.USE_SAM:
            optimizer.zero_grad(set_to_none=True)
            
            # ─────────────────────────────────────────────────────────
            # PASO 1: Calcular gradiente y perturbar pesos (first_step)
            # ─────────────────────────────────────────────────────────
            with autocast(enabled=config.FP16):
                predictions, cls_out = model(images)
                loss, loss_dict = criterion(predictions, masks, cls_pred=cls_out, has_cracks=has_cracks)
                loss = loss / config.ACCUM_STEPS
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backward con scaling
            scaler.scale(loss).backward()
            
            # Unscale para poder usar los gradientes reales
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            
            # SAM first step: perturbar pesos en dirección del gradiente
            # (esto NO actualiza los pesos, solo los perturba temporalmente)
            optimizer.first_step(zero_grad=True)
            
            # ─────────────────────────────────────────────────────────
            # PASO 2: Recalcular gradiente en punto perturbado y actualizar
            # ─────────────────────────────────────────────────────────
            with autocast(enabled=config.FP16):
                predictions_2, cls_out_2 = model(images)
                loss_2, _ = criterion(predictions_2, masks, cls_pred=cls_out_2, has_cracks=has_cracks)
                loss_2 = loss_2 / config.ACCUM_STEPS
            
            # Backward del segundo paso
            scaler.scale(loss_2).backward()
            
            # FIX: Usar scaler.step() que internamente hace unscale si es necesario
            # En lugar de llamar manualmente unscale + optimizer.second_step
            # Restauramos los pesos y aplicamos el paso del optimizador base
            
            # Restaurar pesos originales antes de step
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None and p in optimizer.state and "old_p" in optimizer.state[p]:
                            p.data = optimizer.state[p]["old_p"]
            
            # Ahora step hace unscale + optimizador base
            scaler.step(optimizer.base_optimizer)
            scaler.update()
            
            scheduler.step()
            
            if ema:
                ema.update()
                
        else:
            # ─────────────────────────────────────────────────────────
            # Entrenamiento normal (sin SAM) con acumulación de gradientes
            # ─────────────────────────────────────────────────────────
            with autocast(enabled=config.FP16):
                predictions, cls_out = model(images)
                loss, loss_dict = criterion(predictions, masks, cls_pred=cls_out, has_cracks=has_cracks)
                loss = loss / config.ACCUM_STEPS
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if ema:
                    ema.update()

        # Métricas
        with torch.no_grad():
            pred_sig = torch.sigmoid(predictions[-1] if isinstance(predictions, list) else predictions)
            batch_metrics = compute_metrics(pred_sig, masks)

        running_metrics['loss'].append(loss.item() * config.ACCUM_STEPS)
        for k, v in batch_metrics.items():
            running_metrics[k].append(v)

        pbar.set_postfix({
            'L': f"{np.mean(running_metrics['loss']):.4f}",
            'D': f"{np.mean(running_metrics['dice']):.4f}",
            'BF1': f"{np.mean(running_metrics['boundary_f1']):.3f}"
        })

    return {k: np.mean(v) for k, v in running_metrics.items()}

@torch.no_grad()
def validate_epoch(model, loader, criterion, phase='Val', use_tta=False):
    model.eval()
    running_metrics = defaultdict(list)

    pbar = tqdm(
        loader, 
        desc=f"📊 {phase:>4s}", 
        bar_format="{desc} {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} {postfix}"
    )

    for images, masks, has_cracks in pbar:
        images = images.to(config.DEVICE, non_blocking=True)
        masks = masks.to(config.DEVICE, non_blocking=True)
        has_cracks = has_cracks.to(config.DEVICE, non_blocking=True)

        with autocast(enabled=config.FP16):
            if use_tta and config.USE_TTA:
                predictions_sig, cls_out = tta_predict(model, images)
                predictions = torch.logit(predictions_sig.clamp(1e-7, 1 - 1e-7))
            else:
                predictions, cls_out = model(images)
                predictions_sig = torch.sigmoid(predictions[-1] if isinstance(predictions, list) else predictions)

            total_loss, _ = criterion(predictions, masks, cls_pred=cls_out, has_cracks=has_cracks)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue

        batch_metrics = compute_metrics(predictions_sig, masks)

        running_metrics['loss'].append(total_loss.item())
        for k, v in batch_metrics.items():
            running_metrics[k].append(v)

        pbar.set_postfix({
            'D': f"{np.mean(running_metrics['dice']):.4f}",
            'BF1': f"{np.mean(running_metrics['boundary_f1']):.3f}"
        })

    return {k: np.mean(v) for k, v in running_metrics.items()}

# ═════════════════════════════════════════════════════════════════
#          VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════
@torch.no_grad()
def visualize_predictions(model, dataset, num_samples=4, save_path=None):
    """Visualiza predicciones del modelo"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx, img_idx in enumerate(indices):
        image, mask, _ = dataset[img_idx]
        image_input = image.unsqueeze(0).to(config.DEVICE)
        
        with autocast(enabled=config.FP16):
            pred, _ = model(image_input)
            if isinstance(pred, list):
                pred = pred[-1]
            pred = torch.sigmoid(pred)
        
        img_show = image.cpu().numpy().transpose(1, 2, 0)
        img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_show = np.clip(img_show, 0, 1)
        
        mask_show = mask.cpu().numpy()[0]
        pred_show = pred.cpu().numpy()[0, 0]
        
        axes[idx, 0].imshow(img_show)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_show, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pred_show, cmap='gray')
        axes[idx, 2].set_title(f'Pred (max={pred_show.max():.2f})')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(img_show)
        axes[idx, 3].imshow(pred_show > 0.5, cmap='jet', alpha=0.5)
        axes[idx, 3].set_title('Overlay')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ═════════════════════════════════════════════════════════════════
#                      MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    logger.log("╔" + "═"*68 + "╗")
    logger.log("║" + " "*10 + "CRACK SEGMENTATION ULTIMATE 2025 v3.2" + " "*19 + "║")
    logger.log("╚" + "═"*68 + "╝")
    logger.log(f"📁 Experimento: {config.EXP_DIR}\n")
    
    logger.log("🚀 CONFIGURACIÓN:")
    logger.log(f"   ├─ Progressive Sizing: 160px(E1-3) → 320px(E4-6) → 640px(E7+)")
    logger.log(f"   ├─ Batch Size: {config.BATCH_SIZE_160} → {config.BATCH_SIZE_320} → {config.BATCH_SIZE_640}")
    logger.log(f"   ├─ Learning Rate: {config.LEARNING_RATE_160:.0e} → {config.LEARNING_RATE_320:.0e} → {config.LEARNING_RATE_640:.0e}")
    logger.log(f"   ├─ Augmentations: MixUp={config.USE_MIXUP} | CutMix={config.USE_CUTMIX}")
    logger.log(f"   ├─ SAM Optimizer: {'✅ Enabled' if config.USE_SAM else '❌ Disabled'}")
    logger.log(f"   ├─ OHEM: {'✅ Enabled' if config.USE_OHEM else '❌ Disabled'}")
    logger.log(f"   ├─ Dual Attention: {'✅ Enabled' if config.USE_DUAL_ATTENTION else '❌ Disabled'}")
    logger.log(f"   ├─ EMA Decay: {config.EMA_DECAY}")
    logger.log(f"   ├─ TTA Modes: {config.TTA_MODES}")
    logger.log(f"   └─ Device: {config.DEVICE}")
    logger.log("─" * 70)

    logger.log("\n📁 Cargando datasets...")

    current_size = get_current_img_size(1)
    current_bs = get_batch_size(1)

    train_dataset = CrackDatasetBalanced(
        os.path.join(config.BASE_DATASET_DIR, 'train', '_annotations.coco.json'),
        os.path.join(config.BASE_DATASET_DIR, 'train'), 
        get_train_transform(current_size)
    )
    val_dataset = CrackDatasetBalanced(
        os.path.join(config.BASE_DATASET_DIR, 'valid', '_annotations.coco.json'),
        os.path.join(config.BASE_DATASET_DIR, 'valid'), 
        get_val_transform(current_size)
    )
    test_dataset = CrackDatasetBalanced(
        os.path.join(config.BASE_DATASET_DIR, 'test', '_annotations.coco.json'),
        os.path.join(config.BASE_DATASET_DIR, 'test'), 
        get_val_transform(640)
    )

    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()
    test_stats = test_dataset.get_stats()
    
    logger.log(f"   Train: {train_stats['total']} imgs | +{train_stats['with_cracks']} -{train_stats['without_cracks']}")
    logger.log(f"   Val:   {val_stats['total']} imgs | +{val_stats['with_cracks']} -{val_stats['without_cracks']}")
    logger.log(f"   Test:  {test_stats['total']} imgs | +{test_stats['with_cracks']} -{test_stats['without_cracks']}")

    train_loader = DataLoader(
        train_dataset, batch_size=current_bs, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=current_bs, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE_640, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    logger.log(f"\n🏗  Construyendo UNet++ con Dual Attention...")
    model = UNetPlusPlusModel(
        encoder_name='convnext_base.fb_in22k_ft_in1k',
        deep_supervision=config.USE_DEEP_SUPERVISION
    ).to(config.DEVICE)
    
    criterion = OHEMComboLoss(ratio=config.OHEM_RATIO).to(config.DEVICE)

    initial_lr = get_current_lr(1)
    
    if config.USE_SAM:
        optimizer = SAM(
            model.parameters(),
            AdamW,
            lr=initial_lr,
            weight_decay=config.WEIGHT_DECAY,
            rho=config.SAM_RHO
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer.base_optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min
        )
    else:
        optimizer = AdamW(
            model.parameters(), 
            lr=initial_lr, 
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min
        )

    scaler = GradScaler(enabled=config.FP16)

    ema = None
    if config.USE_EMA:
        ema = EMA(model, decay=config.EMA_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"✓ Parámetros totales: {total_params:,}")
    logger.log(f"✓ Parámetros entrenables: {trainable_params:,}")

    start_epoch = 1
    best_dice = 0.0
    patience_counter = 0
    checkpoint_path = f"{config.EXP_DIR}/checkpoints/last.pth"

    if os.path.exists(checkpoint_path):
        logger.log(f"\n📥 Recuperando checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        patience_counter = ckpt.get('patience', 0)
        if ema and 'ema_state' in ckpt:
            ema.shadow = ckpt['ema_state']
        logger.log(f"✅ Reanudando desde época {start_epoch} | Best Dice: {best_dice:.5f}")

    logger.log("\n" + "─"*70)
    logger.log("🎯 INICIANDO ENTRENAMIENTO\n")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        new_size = get_current_img_size(epoch)
        new_bs = get_batch_size(epoch)

        if new_size != current_size or new_bs != current_bs:
            current_size = new_size
            current_bs = new_bs
            logger.log(f"\n🔄 CAMBIO DE FASE - Época {epoch}")
            logger.log(f"   ├─ Tamaño: {current_size}px")
            logger.log(f"   ├─ Batch Size: {current_bs}")

            new_lr = get_current_lr(epoch)
            if config.USE_SAM:
                for param_group in optimizer.base_optimizer.param_groups:
                    param_group['lr'] = new_lr
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer.base_optimizer,
                    T_0=config.T_0,
                    T_mult=config.T_mult,
                    eta_min=config.eta_min
                )
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=config.T_0,
                    T_mult=config.T_mult,
                    eta_min=config.eta_min
                )
            
            logger.log(f"   ├─ Learning Rate: {new_lr:.0e}")
            logger.log(f"   └─ Scheduler reiniciado")

            train_dataset.update_transform(get_train_transform(current_size))
            val_dataset.update_transform(get_val_transform(current_size))
            
            train_loader = DataLoader(
                train_dataset, batch_size=current_bs, shuffle=True, 
                num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=current_bs, shuffle=False, 
                num_workers=config.NUM_WORKERS, pin_memory=True
            )

            gc.collect()
            torch.cuda.empty_cache()

        # Entrenamiento
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            criterion, epoch, ema, use_sam=config.USE_SAM
        )
        
        # Validación
        val_metrics = validate_epoch(model, val_loader, criterion, phase='Val')

        logger.update_history(epoch, 'train', train_metrics)
        logger.update_history(epoch, 'val', val_metrics)

        current_lr = optimizer.base_optimizer.param_groups[0]['lr'] if config.USE_SAM else optimizer.param_groups[0]['lr']

        logger.log(f"""
╔═══════════════════════════════════════════════════════════════════╗
║ ÉPOCA {epoch:03d}/{config.EPOCHS} │ {get_current_img_size(epoch)}px │ LR={current_lr:.2e}
╠═══════════════════════════════════════════════════════════════════╣
║ TRAIN:
║   ├─ Loss:        {train_metrics['loss']:.5f}
║   ├─ Dice:        {train_metrics['dice']:.5f}
║   ├─ IoU:         {train_metrics['iou']:.5f}
║   ├─ F1 Score:    {train_metrics['f1']:.5f}
║   ├─ Boundary F1: {train_metrics['boundary_f1']:.5f}
║   ├─ Precision:   {train_metrics['precision']:.5f}
║   ├─ Recall:      {train_metrics['recall']:.5f}
║   └─ Accuracy:    {train_metrics['accuracy']:.5f}
╠═══════════════════════════════════════════════════════════════════╣
║ VALIDATION:
║   ├─ Loss:        {val_metrics['loss']:.5f}
║   ├─ Dice:        {val_metrics['dice']:.5f}
║   ├─ IoU:         {val_metrics['iou']:.5f}
║   ├─ F1 Score:    {val_metrics['f1']:.5f}
║   ├─ Boundary F1: {val_metrics['boundary_f1']:.5f}
║   ├─ Precision:   {val_metrics['precision']:.5f}
║   ├─ Recall:      {val_metrics['recall']:.5f}
║   └─ Accuracy:    {val_metrics['accuracy']:.5f}
╚═══════════════════════════════════════════════════════════════════╝
""")

        # Guardar mejor modelo
        if val_metrics['dice'] > best_dice:
            improvement = val_metrics['dice'] - best_dice
            best_dice = val_metrics['dice']
            patience_counter = 0

            model_path = f"{config.EXP_DIR}/models/BEST_E{epoch:03d}_D{best_dice:.5f}_BF1{val_metrics['boundary_f1']:.4f}.pth"
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'metrics': val_metrics,
                'config': vars(config)
            }
            
            if ema:
                save_dict['ema_state'] = ema.shadow
            
            torch.save(save_dict, model_path)
            logger.update_best_models(epoch, best_dice, val_metrics['iou'], model_path)
            logger.log(f"✅ MEJOR MODELO GUARDADO!    Mejora: +{improvement:.5f}")
            
            # Visualizar
            vis_path = f"{config.EXP_DIR}/visualizations/best_E{epoch:03d}.png"
            visualize_predictions(model, val_dataset, num_samples=4, save_path=vis_path)
            logger.log(f"📸 Visualizaciones: {vis_path}")
        else:
            patience_counter += 1
            logger.log(f"⏳ Patience: {patience_counter}/{config.PATIENCE} (Best: {best_dice:.5f})")

        # Guardar checkpoint
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'patience': patience_counter
        }
        
        if ema:
            checkpoint_dict['ema_state'] = ema.shadow
        
        torch.save(checkpoint_dict, checkpoint_path)

        # Early stopping
        if patience_counter >= config.PATIENCE:
            logger.log(f"\n⏹  EARLY STOPPING (sin mejora en {config.PATIENCE} épocas)")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════
    #          EVALUACIÓN EN TEST SET
    # ═════════════════════════════════════════════════════════════
    logger.log("\n" + "═"*70)
    logger.log("🧪 EVALUACIÓN EN TEST SET")
    logger.log("═"*70)

    best_model_path = logger.best_models[0]['path']
    logger.log(f"\n📥 Cargando mejor modelo: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test sin TTA
    logger.log("\n📊 Test sin TTA...")
    test_metrics = validate_epoch(model, test_loader, criterion, phase='Test', use_tta=False)

    # Test con TTA
    logger.log("\n🔄 Test con TTA x4...")
    test_metrics_tta = validate_epoch(model, test_loader, criterion, phase='TTA', use_tta=True)

    # Test con EMA
    test_metrics_ema = None
    test_metrics_ema_tta = None
    if ema and 'ema_state' in checkpoint:
        logger.log("\n🔬 Test con EMA...")
        ema.shadow = checkpoint['ema_state']
        ema.apply_shadow()
        test_metrics_ema = validate_epoch(model, test_loader, criterion, phase='EMA', use_tta=False)
        
        logger.log("\n🔬🔄 Test con EMA + TTA...")
        test_metrics_ema_tta = validate_epoch(model, test_loader, criterion, phase='EMA+TTA', use_tta=True)
        ema.restore()

    # Guardar resultados
    logger.update_history('final', 'test', test_metrics)
    logger.update_history('final', 'test_tta', test_metrics_tta)
    if test_metrics_ema:
        logger.update_history('final', 'test_ema', test_metrics_ema)
        logger.update_history('final', 'test_ema_tta', test_metrics_ema_tta)

    # Visualizaciones finales
    logger.log("\n📸 Generando visualizaciones finales...")
    vis_path_test = f"{config.EXP_DIR}/visualizations/test_predictions.png"
    visualize_predictions(model, test_dataset, num_samples=8, save_path=vis_path_test)
    logger.log(f"✅ Guardadas en: {vis_path_test}")

    # ═════════════════════════════════════════════════════════════
    #          RESULTADOS FINALES
    # ═════════════════════════════════════════════════════════════
    logger.log("\n" + "╔" + "═"*68 + "╗")
    logger.log("║" + " "*20 + "RESULTADOS FINALES" + " "*30 + "║")
    logger.log("╠" + "═"*68 + "╣")
    
    logger.log(f"║ TEST (sin TTA):")
    logger.log(f"║   ├─ Dice:        {test_metrics['dice']:.5f}")
    logger.log(f"║   ├─ IoU:         {test_metrics['iou']:.5f}")
    logger.log(f"║   ├─ F1:          {test_metrics['f1']:.5f}")
    logger.log(f"║   ├─ Boundary F1: {test_metrics['boundary_f1']:.5f}")
    logger.log(f"║   ├─ Precision:   {test_metrics['precision']:.5f}")
    logger.log(f"║   ├─ Recall:      {test_metrics['recall']:.5f}")
    logger.log(f"║   └─ Accuracy:    {test_metrics['accuracy']:.5f}")
    
    logger.log("╠" + "═"*68 + "╣")
    logger.log(f"║ TEST (con TTA x{config.TTA_MODES}):")
    logger.log(f"║   ├─ Dice:        {test_metrics_tta['dice']:.5f} (Δ {test_metrics_tta['dice'] - test_metrics['dice']:+.5f})")
    logger.log(f"║   ├─ IoU:         {test_metrics_tta['iou']:.5f} (Δ {test_metrics_tta['iou'] - test_metrics['iou']:+.5f})")
    logger.log(f"║   ├─ F1:          {test_metrics_tta['f1']:.5f} (Δ {test_metrics_tta['f1'] - test_metrics['f1']:+.5f})")
    logger.log(f"║   └─ Boundary F1: {test_metrics_tta['boundary_f1']:.5f} (Δ {test_metrics_tta['boundary_f1'] - test_metrics['boundary_f1']:+.5f})")
    
    if test_metrics_ema:
        logger.log("╠" + "═"*68 + "╣")
        logger.log(f"║ TEST (EMA):")
        logger.log(f"║   ├─ Dice:        {test_metrics_ema['dice']:.5f}")
        logger.log(f"║   ├─ IoU:         {test_metrics_ema['iou']:.5f}")
        logger.log(f"║   ├─ F1:          {test_metrics_ema['f1']:.5f}")
        logger.log(f"║   └─ Boundary F1: {test_metrics_ema['boundary_f1']:.5f}")
        
        logger.log("╠" + "═"*68 + "╣")
        logger.log(f"║ TEST (EMA + TTA) - 🏆 MEJOR RESULTADO:")
        logger.log(f"║   ├─ Dice:        {test_metrics_ema_tta['dice']:.5f}")
        logger.log(f"║   ├─ IoU:         {test_metrics_ema_tta['iou']:.5f}")
        logger.log(f"║   ├─ F1:          {test_metrics_ema_tta['f1']:.5f}")
        logger.log(f"║   └─ Boundary F1: {test_metrics_ema_tta['boundary_f1']:.5f}")
    
    logger.log("╚" + "═"*68 + "╝")

    # Top modelos
    logger.log(f"\n🏆 TOP {config.SAVE_TOP_K} MEJORES MODELOS:")
    for i, model_info in enumerate(logger.best_models, 1):
        logger.log(f"   {i}.Época {model_info['epoch']:03d} | Dice: {model_info['dice']:.5f} | IoU: {model_info['iou']:.5f}")
        logger.log(f"      {model_info['path']}")

    # Resumen de técnicas
    logger.log("\n📋 TÉCNICAS IMPLEMENTADAS:")
    techniques = [
        ("✅ Progressive Sizing", "160→320→640px"),
        ("✅ UNet++", "Nested skip connections"),
        ("✅ ConvNeXt Encoder", "ImageNet-22k pretrained"),
        ("✅ Dual Attention (CBAM)", "Enabled" if config.USE_DUAL_ATTENTION else "Disabled"),
        ("✅ Deep Supervision", "4 auxiliary outputs"),
        ("✅ SAM Optimizer", f"ρ={config.SAM_RHO}" if config.USE_SAM else "Disabled"),
        ("✅ OHEM Loss", f"Ratio={config.OHEM_RATIO}" if config.USE_OHEM else "Disabled"),
        ("✅ Combo Loss", "FocalTversky+Lovász+Boundary+BCE"),
        ("✅ MixUp", f"α={config.MIXUP_ALPHA}" if config.USE_MIXUP else "Disabled"),
        ("✅ CutMix", f"α={config.CUTMIX_ALPHA}" if config.USE_CUTMIX else "Disabled"),
        ("✅ EMA", f"Decay={config.EMA_DECAY}" if config.USE_EMA else "Disabled"),
        ("✅ TTA", f"{config.TTA_MODES} modes" if config.USE_TTA else "Disabled"),
        ("✅ Mixed Precision (FP16)", "Enabled"),
        ("✅ Gradient Clipping", f"Max norm={config.GRAD_CLIP_NORM}"),
        ("✅ Label Smoothing", f"ε={config.LABEL_SMOOTHING}"),
        ("✅ Cosine Annealing", f"T0={config.T_0}, T_mult={config.T_mult}"),
        ("✅ Advanced Augmentations", "Elastic, Grid, Optical"),
        ("✅ Boundary F1 Metric", "Edge-aware evaluation")
    ]
    
    for technique, detail in techniques:
        logger.log(f"   {technique:<40} {detail}")

    logger.log("\n" + "═"*70)
    logger.log("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.log(f"📁 Resultados: {config.EXP_DIR}")
    logger.log("═"*70 + "\n")

    return {
        'best_dice': best_dice,
        'test_metrics': test_metrics,
        'test_metrics_tta': test_metrics_tta,
        'test_metrics_ema': test_metrics_ema,
        'test_metrics_ema_tta': test_metrics_ema_tta,
        'experiment_dir': config.EXP_DIR
    }

# ═════════════════════════════════════════════════════════════════
#          EJECUCIÓN
# ═════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        results = main()
        
        print("\n" + "🎉"*35)
        print("PROCESO COMPLETADO CON ÉXITO")
        print("🎉"*35)
        print(f"\n📊 Mejor Dice Score: {results['best_dice']:.5f}")
        
        if results['test_metrics_ema_tta']:
            print(f"🏆 Mejor resultado (EMA+TTA):")
            print(f"   ├─ Dice: {results['test_metrics_ema_tta']['dice']:.5f}")
            print(f"   ├─ IoU:  {results['test_metrics_ema_tta']['iou']:.5f}")
            print(f"   └─ BF1:  {results['test_metrics_ema_tta']['boundary_f1']:.5f}")
        
        print(f"\n📁 Experimento: {results['experiment_dir']}")
        
    except KeyboardInterrupt:
        logger.log("\n⚠️  Entrenamiento interrumpido por el usuario")
        logger.log("💾 Checkpoint guardado - puedes reanudar")
        
    except Exception as e:
        logger.log(f"\n❌ ERROR CRÍTICO: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise
    
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        logger.log("\n🧹 Limpieza completada")