# Wheat Quality Inspection — Impurity Detection

Semantic segmentation pipeline to detect impurities (straw, weed, gravel, glass) in wheat grain images using **DeepLabV3 + MobileNetV3-Large**.

## Dataset

[Kaggle — Wheat Images with Impurity](https://www.kaggle.com/datasets/byh0007/wheat-images-with-impurity) — polygon-annotated images with 7 classes:

| Class        | Description                        |
|-------------|------------------------------------|
| Background  | Non-relevant areas                 |
| Wheat       | Wheat grain                        |
| Wheat_Bran  | Broken wheat / bran particles      |
| Straw       | Straw / stalk fragments            |
| Weed        | Weed seeds                         |
| Gravel      | Small stones / gravel              |
| Glass       | Glass shards (most critical)       |

## Experiments

| Version | Key Technique | Description |
|---------|--------------|-------------|
| **v2** / **advanced** | Focal Loss + Augmentations | Baseline DeepLabV3-MobileNetV3-Large with Focal Loss to handle class imbalance and advanced image augmentations (color jitter, blur, noise, etc.). |
| **v3** | Glass Oversampling | Class-weighted Focal Loss (inverse pixel-frequency weights) + `WeightedRandomSampler` to oversample glass-containing images (~2.3x boost per epoch). |
| **v4** | Class-Weighted Loss Only | Same class-weighted Focal Loss as v3, **without** oversampling — isolates the effect of re-weighting alone. |
| **v5** | Glass Copy-Paste Augmentation | Extracts real glass polygon patches from training images and pastes them into random training images (30% probability). No loss weighting or oversampling — purely data augmentation. |

## Architecture

- **Backbone**: MobileNetV3-Large (3.5M params, pretrained on ImageNet)
- **Head**: DeepLabV3 with ASPP (Atrous Spatial Pyramid Pooling)
- **Loss**: Focal Loss (v2, v5) / Class-weighted Focal Loss (v3, v4)
- **Input**: 256×256 RGB images
- **Output**: 7-class segmentation mask + impurity rate estimation

## Results

Per-class IoU, qualitative predictions, impurity rate scatter plots, and training curves are available under [`reports/`](reports/).

## Project Structure

```
├── scripts/
│   ├── run_pipeline_v2.py         # Baseline pipeline
│   ├── run_pipeline_v3.py         # Glass oversampling
│   ├── run_pipeline_v4.py         # Class-weighted loss only
│   ├── run_pipeline_v5.py         # Glass copy-paste augmentation
│   └── run_pipeline_advanced.py   # Advanced variant
├── notebooks/
│   └── wheat_quality_inspection_v2.ipynb
├── models/                        # Trained checkpoints (.pth, .pt, .onnx)
├── reports/                       # Evaluation figures (IoU, curves, etc.)
├── deployment/                    # Docker, FastAPI server, Prometheus monitoring
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline_v2.py   # or v3, v4, v5, advanced
```

## Deployment

See [`deployment/`](deployment/) for a FastAPI inference server with Docker and Prometheus metrics.
