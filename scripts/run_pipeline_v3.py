"""
v3 Pipeline: Glass-focused improvements
- Class-weighted Focal Loss (pixel-frequency weights)
- Oversample Glass-containing training images via WeightedRandomSampler
- Same DeepLabV3-MobileNetV3 backbone as v2
"""
import os, sys, json, warnings, random, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models.segmentation as seg_models

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import multiprocessing as mp
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12})

RANDOM_SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
CLASS_NAMES = ['Background', 'Wheat', 'Wheat_Bran', 'Straw', 'Weed', 'Gravel', 'Glass']
NUM_CLASSES = len(CLASS_NAMES)
LABEL_COLORS = {'Wheat': '#2ecc71', 'Wheat_Bran': '#e74c3c', 'Straw': '#f39c12', 'Weed': '#9b59b6', 'Gravel': '#3498db', 'Glass': '#1abc9c'}
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#1abc9c']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Using device:', DEVICE)

# ---------- Dataset Loading ----------
import kagglehub
dataset_path = kagglehub.dataset_download('byh0007/wheat-images-with-impurity')
DATA_ROOT = Path(dataset_path) / 'train'
IMG_DIR = DATA_ROOT / 'images'
MASK_DIR = DATA_ROOT / 'masks'
RATE_DIR = DATA_ROOT / 'impurity_rate'
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
sample_ids = [f.replace('.jpg', '') for f in image_files]
print(f'\nTotal samples: {len(sample_ids)}')

# Load impurity rates
impurity_rates = {}
for sid in sample_ids:
    with open(RATE_DIR / f'{sid}.txt') as f:
        impurity_rates[sid] = float(f.read().strip())

# ---------- Mask Color Map ----------
MASK_COLOR_MAP = {
    (0, 0, 0): 0, (70, 70, 70): 1, (244, 35, 232): 2,
    (128, 64, 128): 3, (102, 102, 156): 4, (190, 153, 153): 5, (153, 153, 153): 6,
}
COLOR_TO_CLASS = np.zeros(256**3, dtype=np.int64)
for (r, g, b), cls in MASK_COLOR_MAP.items():
    COLOR_TO_CLASS[r * 256**2 + g * 256 + b] = cls

def rgb_mask_to_class(mask_rgb):
    idx = mask_rgb[:,:,0].astype(np.int64) * 256**2 + mask_rgb[:,:,1].astype(np.int64) * 256 + mask_rgb[:,:,2].astype(np.int64)
    return COLOR_TO_CLASS[idx]

# ---------- Dataset ----------
class WheatDataset(Dataset):
    def __init__(self, sample_ids, img_dir, mask_dir, transform=None):
        self.sample_ids = sample_ids; self.img_dir = img_dir; self.mask_dir = mask_dir; self.transform = transform
    def __len__(self): return len(self.sample_ids)
    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        image = cv2.imread(str(self.img_dir / f'{sid}.jpg')); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.imread(str(self.mask_dir / f'{sid}.png')); mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class(mask_rgb)
        if self.transform:
            aug = self.transform(image=image, mask=mask); image = aug['image']; mask = aug['mask']
        else:
            image = torch.from_numpy(image).permute(2,0,1).float()/255.0; mask = torch.from_numpy(mask).long()
        return image, mask

# ---------- Split ----------
train_ids, temp_ids = train_test_split(sample_ids, test_size=0.3, random_state=RANDOM_SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=RANDOM_SEED)
print(f'\nTrain: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')

# ---------- Compute Pixel-Level Class Weights & Glass IDs ----------
print('\n=== Computing class weights from training masks ===')
class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
train_glass_ids = set()

for sid in tqdm(train_ids, desc='Analyzing train masks'):
    mask_rgb = cv2.imread(str(MASK_DIR / f'{sid}.png'))
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    mask = rgb_mask_to_class(mask_rgb)
    for c in range(NUM_CLASSES):
        class_pixel_counts[c] += int((mask == c).sum())
    # Check if this image has Glass annotations
    with open(IMG_DIR / f'{sid}.json') as f:
        ann = json.load(f)
    for shape in ann['shapes']:
        if shape['label'] == 'Glass':
            train_glass_ids.add(sid)
            break

print('Pixel-level class distribution:')
total_px = class_pixel_counts.sum()
for i, name in enumerate(CLASS_NAMES):
    print(f'  {name:15s}: {class_pixel_counts[i]:>12,} px ({100*class_pixel_counts[i]/total_px:.2f}%)')

# Inverse frequency weights (clipped to avoid extreme values)
class_weights = total_px / (NUM_CLASSES * class_pixel_counts.astype(np.float64))
class_weights = np.clip(class_weights, 0.1, 10.0)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalize so mean=1
print('\nClass weights (for loss):')
for i, name in enumerate(CLASS_NAMES):
    print(f'  {name:15s}: {class_weights[i]:.4f}')

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------- Oversample Glass Images ----------
n_train = len(train_ids)
n_glass_train = len(train_glass_ids)
print(f'\nGlass images in train set: {n_glass_train}/{n_train} ({100*n_glass_train/n_train:.1f}%)')

# Sample weights: glass images get higher weight
sample_weights = np.ones(n_train, dtype=np.float64)
glass_boost = n_train / max(n_glass_train, 1)  # e.g., 560/107 ≈ 5.2x
for i, sid in enumerate(train_ids):
    if sid in train_glass_ids:
        sample_weights[i] = glass_boost

print(f'Glass sample weight: {glass_boost:.2f}x, Non-Glass: 1.0x')
print(f'Effective glass samples per epoch: ~{int(n_glass_train * glass_boost)}')

# ---------- Augmentations ----------
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.7),
    A.GaussNoise(var_limit=(5, 25), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# ---------- DataLoaders ----------
train_ds = WheatDataset(train_ids, IMG_DIR, MASK_DIR, train_transform)
val_ds = WheatDataset(val_ids, IMG_DIR, MASK_DIR, val_transform)
test_ds = WheatDataset(test_ids, IMG_DIR, MASK_DIR, val_transform)

train_sampler = WeightedRandomSampler(sample_weights, num_samples=n_train, replacement=True)
train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

images, masks = next(iter(train_loader))
print(f'Batch: {images.shape} | Mask unique: {torch.unique(masks).tolist()}')

# ---------- Model ----------
print('\n=== MODEL: DeepLabV3-MobileNetV3 ===')

def create_deeplab_model(num_classes):
    model = seg_models.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
    return model

model = create_deeplab_model(NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params:,} | Trainable: {trainable:,}')

# ---------- Losses ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__(); self.gamma = gamma; self.alpha = alpha; self.ignore_index = ignore_index
    def forward(self, logits, targets):
        targets = targets.long()
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        targets = targets.long(); nc = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        th = F.one_hot(targets, nc).permute(0,3,1,2).float()
        inter = (probs * th).sum(dim=(2,3)); union = probs.sum(dim=(2,3)) + th.sum(dim=(2,3))
        return 1 - ((2*inter+self.smooth)/(union+self.smooth)).mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, w_focal=0.3, w_dice=0.7, alpha=None):
        super().__init__(); self.focal = FocalLoss(gamma=gamma, alpha=alpha); self.dice = DiceLoss()
        self.w_focal = w_focal; self.w_dice = w_dice
    def forward(self, logits, targets):
        return self.w_focal * self.focal(logits, targets) + self.w_dice * self.dice(logits, targets)

def compute_iou(logits, targets, nc):
    preds = logits.argmax(dim=1); ious = []
    for c in range(nc):
        pi=(preds==c); ti=(targets==c); inter=(pi&ti).sum().float(); union=(pi|ti).sum().float()
        ious.append(inter/union if union>0 else torch.tensor(1.0, device=logits.device))
    return torch.stack(ious)

def compute_dice(logits, targets, nc):
    preds = logits.argmax(dim=1); dices = []
    for c in range(nc):
        pi=(preds==c); ti=(targets==c); inter=(pi&ti).sum().float(); s=pi.sum()+ti.sum()
        dices.append(2*inter/(s+1e-6) if s>0 else torch.tensor(1.0, device=logits.device))
    return torch.stack(dices)

# ---------- Training ----------
print('\n=== TRAINING (v3: class-weighted + oversampled Glass) ===')

# Use class weights in FocalLoss
criterion = FocalDiceLoss(gamma=2.0, w_focal=0.3, w_dice=0.7, alpha=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
best_val_loss = float('inf')
train_hist = {'loss':[],'iou':[],'dice':[]}
val_hist = {'loss':[],'iou':[],'dice':[]}

def train_epoch(model, loader, criterion, optim, device):
    model.train(); tl=0; ti=0; td=0; nb=0
    pbar = tqdm(loader, desc='Train')
    for im, ma in pbar:
        im, ma = im.to(device), ma.to(device)
        optim.zero_grad()
        logits = model(im)['out']; loss = criterion(logits, ma)
        loss.backward(); optim.step()
        iou = compute_iou(logits, ma, NUM_CLASSES).mean(); dice = compute_dice(logits, ma, NUM_CLASSES).mean()
        tl+=loss.item(); ti+=iou.item(); td+=dice.item(); nb+=1
        pbar.set_postfix({'loss':f'{loss.item():.4f}','iou':f'{iou.item():.4f}'})
    return tl/nb, ti/nb, td/nb

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval(); tl=0; ti=0; td=0; nb=0
    pbar = tqdm(loader, desc='Val')
    for im, ma in pbar:
        im, ma = im.to(device), ma.to(device)
        logits = model(im)['out']; loss = criterion(logits, ma)
        iou = compute_iou(logits, ma, NUM_CLASSES).mean(); dice = compute_dice(logits, ma, NUM_CLASSES).mean()
        tl+=loss.item(); ti+=iou.item(); td+=dice.item(); nb+=1
        pbar.set_postfix({'loss':f'{loss.item():.4f}','iou':f'{iou.item():.4f}'})
    return tl/nb, ti/nb, td/nb

print('Starting training...')
for epoch in range(1, NUM_EPOCHS+1):
    train_l, train_i, train_d = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_l, val_i, val_d = val_epoch(model, val_loader, criterion, DEVICE)
    train_hist['loss'].append(train_l); train_hist['iou'].append(train_i); train_hist['dice'].append(train_d)
    val_hist['loss'].append(val_l); val_hist['iou'].append(val_i); val_hist['dice'].append(val_d)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:2d}/{NUM_EPOCHS} | Train L:{train_l:.4f} IoU:{train_i:.4f} | Val L:{val_l:.4f} IoU:{val_i:.4f} | LR:{lr:.2e}')
    if val_l < best_val_loss:
        best_val_loss = val_l
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_loss':val_l,'val_iou':val_i}, 'best_model_v3.pth')
        print(f'  -> Saved (val_loss: {val_l:.4f})')
print('Training complete!')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = range(1, len(train_hist['loss'])+1)
for ax, key, title in zip(axes, ['loss','iou','dice'], ['Loss','IoU','Dice']):
    ax.plot(ep, train_hist[key], 'b-', label='Train'); ax.plot(ep, val_hist[key], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(f'{title} Curves'); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig('training_curves_v3.png', dpi=150, bbox_inches='tight'); plt.close()

# ---------- Evaluation ----------
print('\n=== EVALUATION ===')
checkpoint = torch.load('best_model_v3.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded epoch {checkpoint["epoch"]} val_loss={checkpoint["val_loss"]:.4f}')
model.eval()
test_loss=0; test_iou_list=[]; class_iou_list=[[] for _ in range(NUM_CLASSES)]
test_images_list=[]; test_preds_list=[]; test_masks_list=[]
with torch.no_grad():
    for im, ma in tqdm(test_loader, desc='Test'):
        im, ma = im.to(DEVICE), ma.to(DEVICE)
        logits = model(im)['out']; loss = criterion(logits, ma); test_loss += loss.item()
        ious = compute_iou(logits, ma, NUM_CLASSES)
        test_iou_list.append(ious.mean().item())
        for c in range(NUM_CLASSES): class_iou_list[c].append(ious[c].item())
        test_images_list.append(im.cpu()); test_preds_list.append(logits.argmax(dim=1).cpu()); test_masks_list.append(ma.cpu())
test_loss /= len(test_loader)
mean_iou = np.mean(test_iou_list); mean_class_iou = [np.mean(c) for c in class_iou_list]
print(f'Test Loss: {test_loss:.4f} | Mean IoU: {mean_iou:.4f}')
print('Per-class IoU:')
for i, (name, iou) in enumerate(zip(CLASS_NAMES, mean_class_iou)):
    print(f'  {name:15s}: {iou:.4f}')

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.barh(CLASS_NAMES, mean_class_iou, color=colors[:NUM_CLASSES])
ax.axvline(mean_iou, color='black', linestyle='--', label=f'Mean IoU: {mean_iou:.4f}')
ax.set_xlabel('IoU'); ax.set_title('Per-Class IoU (v3: Glass-improved)')
for bar, val in zip(bars, mean_class_iou): ax.text(max(val+0.01,0.01), bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center')
ax.legend(); plt.tight_layout(); plt.savefig('evaluation_per_class_iou_v3.png', dpi=150, bbox_inches='tight'); plt.close()

test_images_t = torch.cat(test_images_list, dim=0)
test_preds_t = torch.cat(test_preds_list, dim=0)
test_masks_t = torch.cat(test_masks_list, dim=0)

n_samples = min(10, len(test_images_t)); indices = random.sample(range(len(test_images_t)), n_samples)
fig, axes = plt.subplots(n_samples, 3, figsize=(15, n_samples*4))
for row, idx in enumerate(indices):
    img = np.clip(test_images_t[idx].permute(1,2,0).numpy()*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]), 0, 1)
    gt, pred = test_masks_t[idx].numpy(), test_preds_t[idx].numpy()
    axes[row,0].imshow(img); axes[row,0].set_title(f'Input {idx}'); axes[row,0].axis('off')
    axes[row,1].imshow(gt, cmap='tab10', vmin=0, vmax=NUM_CLASSES-1); axes[row,1].set_title('GT'); axes[row,1].axis('off')
    axes[row,2].imshow(pred, cmap='tab10', vmin=0, vmax=NUM_CLASSES-1); axes[row,2].set_title('Pred'); axes[row,2].axis('off')
plt.tight_layout(); plt.savefig('evaluation_qualitative_v3.png', dpi=150, bbox_inches='tight'); plt.close()

# Impurity rate estimation
impurity_class_ids = [i for i,n in enumerate(CLASS_NAMES) if n in ['Wheat_Bran','Straw','Weed','Gravel','Glass']]
pred_rates, true_rates = [], []
for idx in range(len(test_images_t)):
    pred = test_preds_t[idx].numpy(); total = pred.size
    imp_px = sum((pred==c).sum() for c in impurity_class_ids)
    pred_rates.append(imp_px/total); true_rates.append(impurity_rates[test_ids[idx]])
pred_rates = np.array(pred_rates); true_rates = np.array(true_rates)
mae = mean_absolute_error(true_rates, pred_rates); rmse = np.sqrt(mean_squared_error(true_rates, pred_rates)); r2 = r2_score(true_rates, pred_rates)
print(f'\nImpurity Rate: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(true_rates, pred_rates, alpha=0.5, s=30, c='#3498db', edgecolors='white')
ax.plot([0,1],[0,1],'r--',lw=2,label='Perfect')
z = np.polyfit(true_rates, pred_rates, 1)
ax.plot([0,1], np.poly1d(z)([0,1]), 'g-', lw=1.5, label=f'Fit (slope={z[0]:.3f})')
ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title(f'Impurity Rate Estimation (v3)\nMAE={mae:.4f}, R2={r2:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('impurity_rate_v3.png', dpi=150, bbox_inches='tight'); plt.close()

# ---------- Export ----------
print('\n=== EXPORT ===')
model.cpu()
torch.jit.script(model).save('wheat_segmentation_v3.pt')
print('Exported TorchScript')
dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
try:
    torch.onnx.export(model, dummy, 'wheat_segmentation_v3.onnx',
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}, opset_version=11)
    print('Exported ONNX')
except Exception as e:
    print(f'ONNX export skipped ({e})')
model.to(DEVICE)

print(f'\n{"="*60}')
print('V3 PIPELINE COMPLETE')
print(f'{"="*60}')
print('Improvements over v2:')
print('  1. Class-weighted FocalLoss (inverse pixel-frequency weights)')
print(f'  2. Oversampled Glass images {glass_boost:.1f}x in training')
print(f'  Files: best_model_v3.pth, wheat_segmentation_v3.pt')
