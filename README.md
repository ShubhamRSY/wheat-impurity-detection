# Wheat Quality Inspection — Impurity Detection

## Why This Project

In food processing, impurities like **straw, weed seeds, gravel, and glass shards** in wheat grains pose serious quality and safety risks. Manual inspection is slow, inconsistent, and expensive. This project builds an automated semantic segmentation system that:

- **Locates impurities pixel-wise** in wheat grain images
- **Estimates impurity rate** (percentage of impure area)
- **Compares multiple strategies** for handling the hardest class: **glass** (rare but critical)

Beyond the application, this project serves as a **showcase of end-to-end computer vision practices** — from data exploration and class imbalance strategies to model export and containerized deployment.

---

## What This Project Covers

| Practice | How It's Applied |
|---|---|
| **Semantic Segmentation** | DeepLabV3 with MobileNetV3-Large backbone replacing a heavy U-Net baseline |
| **Transfer Learning** | ImageNet-pretrained backbone, fine-tuned on wheat impurity data |
| **Handling Class Imbalance** | 3 different approaches explored: Focal Loss, class-weighted loss, and oversampling |
| **Data Augmentation** | Albumentations pipeline (color jitter, blur, noise, affine) |
| **Copy-Paste Augmentation** | v5 extracts real Glass polygon patches and pastes them into training images |
| **Exploratory Data Analysis (EDA)** | Class distribution plots, polygon counts, image size analysis |
| **Multi-metric Evaluation** | Per-class IoU, qualitative mask comparisons, impurity rate regression (MAE, R²) |
| **Model Export / Deployment** | TorchScript tracing + ONNX export + FastAPI server + Docker + Prometheus |

---

## Dataset

[Kaggle — Wheat Images with Impurity](https://www.kaggle.com/datasets/byh0007/wheat-images-with-impurity) — polygon-annotated images with 7 classes:

| Class        | Description                        | Frequency |
|-------------|------------------------------------|-----------|
| Background  | Non-relevant areas                 | Dominant  |
| Wheat       | Wheat grain                        | High      |
| Wheat_Bran  | Broken wheat / bran particles      | Medium    |
| Straw       | Straw / stalk fragments            | Medium    |
| Weed        | Weed seeds                         | Medium    |
| Gravel      | Small stones / gravel              | Low       |
| Glass       | Glass shards **(most critical)**   | **Very low** |

Glass is the rarest class but the most safety-critical — making it the central challenge of this project.

---

## Experiment Evolution

Each version builds on the same **DeepLabV3 + MobileNetV3-Large** backbone but varies the **class imbalance strategy**:

```
v2 (baseline) ─── Focal Loss + augmentations
  │
  ├── v3 ─── Glass oversampling + class-weighted Focal Loss
  │
  ├── v4 ─── Class-weighted Focal Loss only (isolates weighting effect)
  │
  └── v5 ─── Glass copy-paste augmentation (adds glass examples)
```

| Version | Key Technique | Why |
|---------|--------------|-----|
| **v2** / **advanced** | Focal Loss + Augmentations | Baseline — Focal Loss reduces loss contribution from easy (background) pixels, forcing the model to focus on hard impurity pixels. |
| **v3** | Glass Oversampling | Class-weighted Focal Loss + `WeightedRandomSampler` shows ~2.3× more glass images per epoch. Targets the root problem: glass appears rarely. |
| **v4** | Class-Weighted Loss Only | Same weights as v3, **no oversampling**. Isolates whether the improvement comes from loss weighting or seeing more glass samples. |
| **v5** | Glass Copy-Paste | Extracts real glass polygon patches and pastes them into random training images (30% probability). No loss changes — purely feeding more varied glass examples. |

---

## Architecture

- **Backbone**: MobileNetV3-Large (3.5M params, pretrained on ImageNet) — lightweight, efficient for deployment
- **Segmentation Head**: DeepLabV3 with **ASPP** (Atrous Spatial Pyramid Pooling) — captures multi-scale context using dilated convolutions at different rates
- **Loss**: Focal Loss (v2, v5) / Class-weighted Focal Loss (v3, v4)
- **Input**: 256×256 RGB images
- **Output**: 7-class segmentation mask + per-image impurity rate (via pixel counting)
- **Training**: 30 epochs, Adam optimizer, learning rate 1e-3, batch size 16

### Why DeepLabV3 + MobileNetV3?

Replaced a 31.4M-parameter scratch U-Net with a 3.5M-parameter pretrained model — achieving **~10× faster training** with better accuracy through transfer learning.

---

## Results

Per-class IoU, qualitative predictions, impurity rate scatter plots (with MAE and R²), and training curves are available under [`reports/`](reports/).

| Report | What It Shows |
|--------|--------------|
| `eda_samples.png` / `eda_summary.png` | Sample images with masks, class frequency distribution |
| `training_curves_v*.png` | Loss and mIoU over training epochs |
| `evaluation_per_class_iou_v*.png` | Per-class IoU bar chart |
| `evaluation_qualitative_v*.png` | Side-by-side: input → ground truth → predicted mask |
| `impurity_rate_v*.png` | True vs predicted impurity rate with MAE / R² |

---

## Project Structure

```
├── scripts/
│   ├── run_pipeline_v2.py         # Baseline — Focal Loss + augmentations
│   ├── run_pipeline_v3.py         # Glass oversampling + class-weighted loss
│   ├── run_pipeline_v4.py         # Class-weighted loss only
│   ├── run_pipeline_v5.py         # Glass copy-paste augmentation
│   └── run_pipeline_advanced.py   # Advanced variant
├── notebooks/
│   └── wheat_quality_inspection_v2.ipynb   # Interactive exploration
├── models/                        # Trained checkpoints (.pth, .pt, .onnx)
├── reports/                       # EDA + evaluation figures for all versions
├── deployment/                    # Production inference setup
│   ├── api_server.py              # FastAPI inference endpoint
│   ├── Dockerfile                 # Containerized deployment
│   ├── docker-compose.yml         # Multi-service orchestration
│   └── prometheus.yml             # Metrics monitoring
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline_v2.py   # or v3, v4, v5, advanced
```

Each script is self-contained — it downloads the dataset (via kagglehub), runs EDA, trains the model, evaluates, and exports TorchScript + ONNX artifacts.

## Deployment

See [`deployment/`](deployment/) for a FastAPI inference server with Docker and Prometheus monitoring:

```bash
cd deployment
docker compose up --build
```

---

## Key Takeaways

- **Transfer learning with a lightweight backbone** (MobileNetV3) dramatically reduces training time vs scratch U-Net
- **Focal Loss** alone handles moderate class imbalance well
- **Oversampling the rare class** (v3) improves detection but can distort the overall distribution
- **Copy-paste augmentation** (v5) adds realistic rare-class examples without altering loss or sampling — a clean, effective approach
- The full pipeline from **EDA → training → evaluation → export → deployment** is automated in each script, making it easy to iterate and compare strategies
