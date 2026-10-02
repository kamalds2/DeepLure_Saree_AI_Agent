# 🧵 DeepLure Saree AI Agent

> **"Face Recognition for Textiles"** — Identify saree designs independent of color palette

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://kaggle.com)

---

## 🎯 Objective

Build and evaluate a system that identifies a saree by the **design on its surface**, independent of the color palette in which that design is rendered.

- **Identification**: Given a query image, rank a gallery of known designs by similarity
- **Verification**: Given a pair of images, decide whether they carry the same design
- **Core requirement**: Color invariance — same motif in different palettes must match; different motifs must not, even in identical palettes

---

## 🏗️ Architecture

```
Input Image (224×224 RGB)
    ↓  [Color-Invariant Augmentation]
    ↓  (ColorJitter hue±0.3, sat±0.8 · RandomGrayscale 30%)
EfficientNet-B3 (ImageNet pretrained)
    ↓  Global Average Pooling → 1536-d
FC Layer → 512-d Embedding
    ↓  L2 Normalization (unit hypersphere)
ArcFace Head (training) │ Cosine Similarity (inference)
```

| Component | Choice | Params |
|---|---|---|
| Backbone | EfficientNet-B3 | ~10.7M |
| Embedding | 512-d + L2 norm | ~0.8M |
| Loss | ArcFace (m=0.5, s=64) | ~2K |
| **Total** | | **~12.5M** |

---

## 📊 Datasets

| Dataset | Images | Labels | License | Usage |
|---|---|---|---|---|
| Indian Saree Patterns (Kaggle) | 1,468 | Banarasi/Bandhani/Ikat/Pichwai | MIT | Backbone training |
| DeepLure Handloom Corpus | 165 | None (proprietary) | Proprietary | Gallery + Query eval |

> ⚠️ The handloom corpus is proprietary to 3rd-party vendors. It is **not included** in this repo and must be deleted after the exercise.

---

## 📁 Repository Structure

```
DeepLure_Saree_AI_Agent/
├── README.md
├── requirements.txt
├── DeepLure_Architecture_Plan.md   # Full decision rationale
├── configs/
│   └── config.yaml                 # All hyperparameters
├── data/
│   ├── dataset.py                  # KaggleSareeDataset + HandloomDataset
│   ├── transforms.py               # Color-invariant augmentation pipeline
│   └── splits.py                   # Gallery/query split logic
├── models/
│   ├── backbone.py                 # EfficientNet-B3 feature extractor
│   ├── embedding_head.py           # 512-d embedding + L2 norm
│   └── arcface.py                  # ArcFace loss (angular margin)
├── train.py                        # 3-phase training loop
├── evaluate.py                     # Recall@K, mAP, ROC-AUC, EER
├── inference.py                    # Single / batch inference
├── gallery_builder.py              # Build + save gallery embeddings
├── notebooks/
│   └── DeepLure_Kaggle.ipynb       # Complete Kaggle-runnable notebook
├── utils/
│   ├── metrics.py                  # Recall@K, mAP, ROC-AUC, EER
│   ├── visualization.py            # t-SNE, similarity heatmaps
│   └── logger.py                   # Training logger
└── reports/
    ├── approach_note.md            # 500-char deliverable
    └── efficiency_report.md        # FLOPs, latency, params
```

---

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train (local)
```bash
python train.py --config configs/config.yaml --data_root ./kaggle
```

### Build Gallery
```bash
python gallery_builder.py \
    --checkpoint checkpoints/best_model.pth \
    --gallery_dir ./handloom_sarees \
    --output gallery_embeddings.npy
```

### Evaluate (Identification + Verification)
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --gallery_embeddings gallery_embeddings.npy \
    --query_dir ./handloom_sarees \
    --split_seed 42
```

### Inference on single image
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --query path/to/saree.jpg \
    --gallery_embeddings gallery_embeddings.npy \
    --top_k 5
```

---

## 📏 Evaluation Protocol

### Identification
- Gallery: 80% of handloom corpus (random seed=42)
- Query: 20% of handloom corpus
- Metrics: **Recall@1, Recall@5, Recall@10, mAP**

### Verification
- Positive pairs: same style class (Kaggle) / same filename group
- Negative pairs: different style class
- Metrics: **ROC-AUC, EER, Accuracy@optimal_threshold**

---

## 📐 Efficiency (Bonus)

| Metric | Value |
|---|---|
| Parameters | ~12.5M |
| FLOPs (224×224) | ~1.8B |
| Embedding size | 512-d float32 = 2 KB |
| Inference latency | < 20ms (T4 GPU) |
| Gallery search (N=1000) | < 5ms |

---

## 🔍 Color Invariance Strategy

1. **ColorJitter** (hue=0.3, sat=0.8) — teaches model to ignore color during training
2. **RandomGrayscale(p=0.3)** — 30% of training sees grayscale: strongest color invariance teacher
3. **ArcFace on unit hypersphere** — angular margin is color-agnostic
4. **Style-class training** — Banarasi in red & blue are both "Banarasi" → model learns motif structure, not color

---

## 📝 Approach Note (500 chars)

> "We frame saree design identification as metric learning. EfficientNet-B3 (ImageNet pretrained) extracts texture embeddings; an ArcFace head with angular margin m=0.5 maps images to a 512-d unit hypersphere. Color invariance is achieved via aggressive ColorJitter (hue±0.3, sat±0.8) and 30% RandomGrayscale augmentation. The 4-class Kaggle set (Banarasi/Bandhani/Ikat/Pichwai) provides style labels for training. Retrieval uses cosine similarity; verification thresholds the same score. Evaluated by Recall@1/5 and ROC-AUC."

---

## 📌 Disclosures

- **Pretrained weights**: EfficientNet-B3 from `torchvision.models` (ImageNet)
- **External data**: Indian Saree Patterns from Kaggle (MIT License, Roboflow export)
- **Proprietary data**: DeepLure Handloom Corpus (not redistributed, deleted post-exercise)
- **No third-party metric learning libraries**: ArcFace implemented from scratch

---

*Built for the DeepLure AI Challenge | Framework: PyTorch | Compute: Kaggle Free GPU*
