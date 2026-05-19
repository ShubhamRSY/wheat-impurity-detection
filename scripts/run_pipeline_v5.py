"""
v5 Pipeline: Glass Copy-Paste Augmentation
- Extracts real Glass polygon patches from training images
- Pastes them into random training images (30% probability)
- No class-weighted loss, no oversampling — same as v2 baseline
- Only adds more Glass training examples without distorting distribution
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
from torch.utils.data import Dataset, DataLoader
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJ_DIR / 'models'
REPORT_DIR = PROJ_DIR / 'reports'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
CLASS_NAMES = ['Background', 'Wheat', 'Wheat_Bran', 'Straw', 'Weed', 'Gravel', 'Glass']
NUM_CLASSES = len(CLASS_NAMES)
GLASS_CLASS_ID = 6
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
try:
    import kagglehub
    dataset_path = kagglehub.dataset_download('byh0007/wheat-images-with-impurity')
except (ImportError, Exception) as e:
    print(f'Dataset download failed ({e}). Trying local path...')
    dataset_path = str(PROJ_DIR / 'data' / 'wheat-impurity-detection')
DATA_ROOT = Path(dataset_path) / 'train'
IMG_DIR = DATA_ROOT / 'images'
MASK_DIR = DATA_ROOT / 'masks'
RATE_DIR = DATA_ROOT / 'impurity_rate'
if not IMG_DIR.exists():
    print(f'Error: Data not found at {IMG_DIR}')
    print('Please download the dataset from Kaggle: https://www.kaggle.com/datasets/byh0007/wheat-images-with-impurity')
    sys.exit(1)
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

# ---------- Build Glass Patch Pool from Training Images ----------
print('\n=== Building Glass patch pool ===')
train_ids, temp_ids = train_test_split(sample_ids, test_size=0.3, random_state=RANDOM_SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=RANDOM_SEED)
print(f'Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')

glass_patch_pool = []  # list of (image_patch_rgb, mask_patch_binary)
MIN_PATCH_SIZE = 20  # skip tiny patches
MAX_PATCH_SIZE = 200

for sid in tqdm(train_ids, desc='Extracting Glass patches'):
    with open(IMG_DIR / f'{sid}.json') as f:
        ann = json.load(f)
    img = cv2.imread(str(IMG_DIR / f'{sid}.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    for shape in ann['shapes']:
        if shape['label'] != 'Glass':
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        pw, ph = x_max - x_min, y_max - y_min
        if pw < MIN_PATCH_SIZE or ph < MIN_PATCH_SIZE or pw > MAX_PATCH_SIZE or ph > MAX_PATCH_SIZE:
            continue
        # Add padding
        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        # Create polygon mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 1)
        # Crop to bounding box
        crop_mask = poly_mask[y_min:y_max, x_min:x_max]
        crop_img = img[y_min:y_max, x_min:x_max]
        glass_patch_pool.append((crop_img, crop_mask))

print(f'Glass patches extracted: {len(glass_patch_pool)}')
if len(glass_patch_pool) == 0:
    print('WARNING: No Glass patches found in training set!')

GLASS_PASTE_PROB = 0.30  # probability of pasting Glass into a training image

# ---------- Dataset with Glass Copy-Paste ----------
class WheatDatasetWithGlassPaste(Dataset):
    def __init__(self, sample_ids, img_dir, mask_dir, glass_pool, paste_prob=0.3, transform=None):
        self.sample_ids = sample_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.glass_pool = glass_pool
        self.paste_prob = paste_prob
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        image = cv2.imread(str(self.img_dir / f'{sid}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.imread(str(self.mask_dir / f'{sid}.png'))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class(mask_rgb)

        # Glass copy-paste augmentation (at full resolution, before Resize)
        if len(self.glass_pool) > 0 and random.random() < self.paste_prob:
            image, mask = self._paste_glass(image, mask)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        return image, mask

    def _paste_glass(self, image, mask):
        h, w = image.shape[:2]
        patch_img, patch_mask = random.choice(self.glass_pool)
        ph, pw = patch_img.shape[:2]
        # Ensure patch fits
        if ph > h or pw > w:
            # Resize patch to fit
            scale = min(h / ph, w / w) * 0.8
            new_w = max(int(pw * scale), 10)
            new_h = max(int(ph * scale), 10)
            new_w = min(new_w, w)
            new_h = min(new_h, h)
            patch_img = cv2.resize(patch_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            ph, pw = new_h, new_w
        # Random rotation
        angle = random.uniform(-30, 30)
        center = (pw // 2, ph // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        patch_img = cv2.warpAffine(patch_img, M, (pw, ph), borderMode=cv2.BORDER_REFLECT)
        patch_mask = cv2.warpAffine(patch_mask, M, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Random location
        x = random.randint(0, w - pw)
        y = random.randint(0, h - ph)
        # Paste pixels onto image
        paste_region = patch_mask > 0
        if paste_region.any():
            for c in range(3):
                image[y:y+ph, x:x+pw, c][paste_region] = patch_img[:, :, c][paste_region]
            mask[y:y+ph, x:x+pw][paste_region] = GLASS_CLASS_ID
        return image, mask


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

class BaseWheatDataset(Dataset):
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

train_ds = WheatDatasetWithGlassPaste(train_ids, IMG_DIR, MASK_DIR, glass_patch_pool,
                                        paste_prob=GLASS_PASTE_PROB, transform=train_transform)
val_ds = BaseWheatDataset(val_ids, IMG_DIR, MASK_DIR, val_transform)
test_ds = BaseWheatDataset(test_ids, IMG_DIR, MASK_DIR, val_transform)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
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

# ---------- Losses (same as v2 baseline - no class weights) ----------
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
    def __init__(self, gamma=2.0, w_focal=0.3, w_dice=0.7):
        super().__init__(); self.focal = FocalLoss(gamma=gamma); self.dice = DiceLoss()
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
print('\n=== TRAINING (v5: Glass copy-paste augmentation) ===')

criterion = FocalDiceLoss(gamma=2.0, w_focal=0.3, w_dice=0.7)
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
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_loss':val_l,'val_iou':val_i}, str(MODEL_DIR / 'best_model_v5.pth'))
        print(f'  -> Saved (val_loss: {val_l:.4f})')
print('Training complete!')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = range(1, len(train_hist['loss'])+1)
for ax, key, title in zip(axes, ['loss','iou','dice'], ['Loss','IoU','Dice']):
    ax.plot(ep, train_hist[key], 'b-', label='Train'); ax.plot(ep, val_hist[key], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(f'{title} Curves'); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(str(REPORT_DIR / 'training_curves_v5.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------- Evaluation ----------
print('\n=== EVALUATION ===')
checkpoint = torch.load(str(MODEL_DIR / 'best_model_v5.pth'), map_location=DEVICE)
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
ax.set_xlabel('IoU'); ax.set_title('Per-Class IoU (v5: Glass copy-paste)')
for bar, val in zip(bars, mean_class_iou): ax.text(max(val+0.01,0.01), bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center')
ax.legend(); plt.tight_layout(); plt.savefig(str(REPORT_DIR / 'evaluation_per_class_iou_v5.png'), dpi=150, bbox_inches='tight'); plt.close()

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
plt.tight_layout(); plt.savefig(str(REPORT_DIR / 'evaluation_qualitative_v5.png'), dpi=150, bbox_inches='tight'); plt.close()

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
ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title(f'Impurity Rate Estimation (v5)\nMAE={mae:.4f}, R2={r2:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(str(REPORT_DIR / 'impurity_rate_v5.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------- Export ----------
print('\n=== EXPORT ===')
model.cpu()
torch.jit.script(model).save(str(MODEL_DIR / 'wheat_segmentation_v5.pt'))
print('Exported TorchScript')
dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
try:
    torch.onnx.export(model, dummy, str(MODEL_DIR / 'wheat_segmentation_v5.onnx'),
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}, opset_version=11)
    print('Exported ONNX')
except Exception as e:
    print(f'ONNX export skipped ({e})')
model.to(DEVICE)

print(f'\n{"="*60}')
print('V5 PIPELINE COMPLETE')
print(f'{"="*60}')
print(f'Glass copy-paste augmentation ({GLASS_PASTE_PROB*100:.0f}% probability, {len(glass_patch_pool)} patches)')
print('No class weights, no oversampling — same loss as v2 baseline')
print('  Files: best_model_v5.pth, wheat_segmentation_v5.pt')
