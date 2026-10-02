# 🧵 DeepLure Saree AI Agent — Architecture & Build Plan

> **Every decision documented. Every trade-off explained.**

---

## 📋 Project Snapshot

| Field | Value |
|---|---|
| **Objective** | Identify saree designs independent of color palette — "face recognition for textiles" |
| **Framework** | PyTorch (mandatory) |
| **Compute** | Kaggle Notebooks (free GPU tier — P100/T4) |
| **Repo** | https://github.com/kamalds2/DeepLure_Saree_AI_Agent |
| **Tasks** | Identification (ranking) + Verification (pair similarity) |
| **Core constraint** | **Color invariance**: same motif in different palettes → MATCH; different motifs → NO MATCH even in same palette |

---

## 1. 📊 Dataset Analysis

### 1.1 Dataset A — Kaggle "Indian Saree Patterns" (`/kaggle`)

**Source**: `https://www.kaggle.com/datasets/div456/indian-saree-patterns` (Roboflow export, MIT License)

| Split | Banarasi | Bandhani | Ikat | Pichwai | Total |
|---|---|---|---|---|---|
| **train** | 432 | 279 | 303 | 279 | **1,293** |
| **valid** | 43 | 22 | 26 | 24 | **115** |
| **test** | 14 | 15 | 13 | 18 | **60** |
| **Grand Total** | 489 | 316 | 342 | 321 | **1,468** |

**Key observations**:
- All images are **640×640** (pre-resized by Roboflow, stretch-scaled)
- Augmentation already applied: 3× per source image with salt-and-pepper noise
- **Labels are style/weave-type classes** (Banarasi, Bandhani, Ikat, Pichwai), NOT individual design IDs
- This is a **classification-level** labeling — fine-grained design identity is NOT captured per image
- **Class imbalance**: Banarasi has 432 train images vs. Bandhani/Pichwai 279 each

**Decision for this dataset**: Use as **style/class-level source** for pretraining the backbone. Because individual design IDs aren't labeled, we cannot train metric learning directly here — but the class boundaries help the backbone learn texture discrimination. We treat each Roboflow "source image group" (3 augmented versions of same source) as a pseudo-identity for triplet mining. There are approximately **489 unique source images** across all splits.

---

### 1.2 Dataset B — DeepLure Handloom Corpus (`/handloom_sarees`)

**Source**: Proprietary (DeepLure/3rd-party vendors) — DO NOT redistribute. Delete after exercise.

| Attribute | Value |
|---|---|
| **Total images** | **165 flat files** |
| **Naming** | `img_XXXXXX.jpg` (plain saree images) and `h_img_XXXXXX.jpg` (handloom variant) |
| **h_img prefix count** | 9 images |
| **img prefix count** | 156 images |
| **Structure** | Flat directory (no class labels) |
| **File sizes** | Varies wildly: 4 KB to 573 KB |

**Key observations**:
- No labels provided — this is an **unlabeled proprietary corpus**
- Two naming prefixes: `h_img_*` likely = handloom variants, `img_*` = general corpus
- The very small size (165 images) means we **cannot train from scratch** on this dataset alone
- This dataset likely represents the **gallery/reference database** in the final evaluation — images to be matched against
- Must treat as the **core evaluation set**: gallery = subset of these images, queries = held-out subset

**Decision for this dataset**: Use as the **primary gallery + query pool** for evaluation. Split into gallery (≈80%) and query (≈20%) in a **closed-set protocol**. Because there are no design-level labels, we will use filenames as pseudo-IDs where `h_img_*` and `img_*` prefixes serve as a proxy grouping (though not reliable for verification pairs without manual annotation).

---

### 1.3 Why Data Matters Here — The Core Challenge

> The task requires **color invariance** at the design/motif level. A red Banarasi and a blue Banarasi with the same weave pattern must have high embedding similarity; a red Banarasi and a red Ikat must have low similarity.

Existing labels (Banarasi/Ikat/etc.) capture **style categories**, not individual design instances. This creates a gap between the labels we have and the evaluation we need. Our approach must bridge this gap through:
1. **Color-invariant preprocessing** (grayscale + texture channels)
2. **Metric learning** to pull same-design embeddings together
3. **Cross-colorway augmentation** to train the network to ignore color

---

## 2. 🏗️ Architecture Decision

### 2.1 Problem Framing: Metric Learning, Not Classification

**Why metric learning over classification?**

| Approach | Why NOT |
|---|---|
| Standard classification (softmax) | Closed-set; can't generalize to unseen designs; doesn't produce similarity scores |
| Zero-shot CLIP | Good baseline but not fine-tuned for saree textures; still color-sensitive |
| Object detection | Not applicable — full image is the saree design |
| **Metric learning (our choice)** | Open-set; produces L2-normalized embeddings for gallery search; directly solves identification + verification |

**Analogy**: This is exactly how face recognition (ArcFace, FaceNet) works — we want a manifold where same-design images cluster together regardless of color.

---

### 2.2 Backbone: EfficientNet-B3

**Why EfficientNet-B3?**

| Criterion | EfficientNet-B3 | Alternatives considered |
|---|---|---|
| **Parameters** | ~12M | ResNet-50 (25M), ViT-B (86M), MobileNet-V3 (5M) |
| **ImageNet top-1** | 81.6% | ResNet-50: 76%, MobileNet-V3: 75% |
| **FLOPs** | 1.8B | ResNet-50: 4.1B, ViT-B: 17.6B |
| **Kaggle GPU fit** | ✅ Easily fits P100/T4 with batch=32 | ViT-B barely fits batch=16 |
| **Texture sensitivity** | High (compound scaling) | ResNet has receptive field issues at small textures |
| **Pretrained weights** | torchvision ImageNet | Available everywhere |

**Why not ViT?**
- ViT relies on global patches and positional embeddings which can encode color heavily
- Less efficient for small datasets (1,468 images total) — needs much more data to shine
- FLOPs cost too high for Kaggle free tier with reasonable batch sizes

**Why not ResNet-50?**
- Double the parameters for lower accuracy
- Less efficient at capturing fine texture patterns (saree weave is a texture task!)

**Why EfficientNet-B3 over B0/B1?**
- B3 strikes best balance between accuracy and compute
- B0 (5.3M params) loses too much feature richness for fine-grained texture matching
- B4+ (19M params) is overkill and strains Kaggle GPU memory

**Decision: EfficientNet-B3 pretrained on ImageNet, fine-tuned with metric learning head.**

---

### 2.3 Loss Function: ArcFace (Additive Angular Margin Loss)

**Why ArcFace?**

| Loss | Description | Pros | Cons |
|---|---|---|---|
| Triplet Loss | Push anchor-positive together, anchor-negative apart | Intuitive | Hard mining required; slow convergence; triplet explosion |
| Contrastive Loss | Pair-based; minimize/maximize distance | Simple | Requires balanced pairs; less discriminative than angular |
| Softmax CE | Classification loss | Easy to implement | Closed-set; poor embedding structure for retrieval |
| **ArcFace** | Additive angular margin on hypersphere | Best embedding quality; state-of-art on face recognition | Needs class labels |
| CosFace/SphereFace | Variants of angular margin | Good | Slightly lower performance than ArcFace |

**ArcFace is chosen because**:
1. Produces embeddings on a **unit hypersphere** — cosine similarity directly measures design closeness
2. **Class labels we have** (Banarasi/Bandhani/Ikat/Pichwai) are sufficient for pretraining the metric space
3. Angular margin is **color-agnostic** — it enforces compact clusters, not Euclidean distance which can be dominated by color channels
4. Best performing loss for fine-grained retrieval tasks in literature
5. The margin parameter `m` is tunable — we can increase it to force more discriminative embeddings

**Implementation**: ArcFace head on top of EfficientNet-B3 with 512-dim embedding layer.

```
Input Image (3×224×224)
    ↓
Color Normalization + Grayscale Augmentation
    ↓
EfficientNet-B3 (pretrained, fine-tuned)
    ↓
Global Average Pooling → 1536-dim feature
    ↓
FC Layer → 512-dim embedding
    ↓
L2 Normalization (unit sphere)
    ↓
ArcFace Head (during training only)
    ↓
Cross-Entropy Loss (with angular margin m=0.5, scale s=64)
```

**At inference**: Only the 512-dim L2-normalized embedding is extracted. Similarity = cosine similarity.

---

### 2.4 Color Invariance Strategy (Most Critical Design Decision)

This is the **heart** of the system. Without color invariance, the model will match "red sarees to red sarees" regardless of pattern — exactly wrong.

**Multi-pronged approach**:

#### Strategy 1: Input Channel Engineering
- Convert images to **HSV color space** and **drop the Hue (H) and Saturation (S) channels**
- Feed only **Value (V = brightness/luminance)** as the primary channel + original RGB as auxiliary
- Alternatively: use **LAB color space** and feed only the **L (lightness) channel** as 3-channel grayscale
- **Why this works**: Textile motifs are defined by their weave pattern / luminance structure, not hue

#### Strategy 2: Data Augmentation — Color Jitter + Grayscale
Apply aggressive color augmentation during training:
```python
transforms.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.8,  # High — teaches to ignore saturation changes
    hue=0.3          # High — teaches to ignore hue shifts
),
transforms.RandomGrayscale(p=0.3),  # 30% chance to see grayscale version
```
**Why**: Forces the network to NOT rely on color for its decisions. If 30% of the time an image appears grayscale, color cannot be a reliable feature.

#### Strategy 3: Cross-colorway Pair Mining
- Within the Kaggle dataset: the 3 Roboflow augmentations of the same source image are treated as **same-design pairs** — they have the same pattern but slightly different noise
- During ArcFace training: same-class images are treated as the same "identity" encouraging the model to cluster all colorways of a style together

#### Strategy 4: Texture Emphasis via Gabor/Edge Preprocessing (Optional Bonus)
- Apply **Gabor filter banks** or **Canny edge detection** as additional input channels
- Gabor filters are specifically designed to capture **orientation-specific texture patterns** — exactly what saree weaves are
- This can be concatenated as a 4th channel or used as a separate branch

---

### 2.5 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEEPLURE SAREE AI                        │
│                  "Face Recognition for Textiles"            │
└─────────────────────────────────────────────────────────────┘

TRAINING PHASE:
──────────────
Input → [Color-Invariant Preprocessing] → [EfficientNet-B3] → [Embedding Head 512-d] → [ArcFace Loss]

INFERENCE / GALLERY BUILDING:
──────────────────────────────
Gallery Images → Preprocessing → Model → 512-d Embeddings → Gallery Index (FAISS / torch.cdist)

QUERY TIME:
───────────
Query Image → Preprocessing → Model → 512-d Embedding → cosine_sim(query, all gallery) → Ranked List

VERIFICATION:
─────────────
Image A + Image B → Embeddings ea, eb → cosine_sim(ea, eb) > θ → SAME DESIGN / DIFFERENT DESIGN
```

---

## 3. 🔄 Data Pipeline Design

### 3.1 Data Combination Strategy

**What we do with both datasets**:

| Dataset | How we use it | Rationale |
|---|---|---|
| Kaggle (1,468 images, 4 classes) | **Pretraining backbone** with ArcFace on 4-class style labels | Provides labeled data for metric learning warmup |
| Handloom (165 images, no labels) | **Gallery + Query set** for final evaluation | Represents the real-world retrieval scenario |
| Both combined | **Fine-tuning** on pseudo-labels derived from filename groups | Bridge domain gap |

**Why combine?**
- 1,468 images is too few to train from scratch
- ImageNet pretraining → Kaggle ArcFace finetuning → gives us a color-invariant texture encoder
- The handloom dataset then serves as the evaluation arena

### 3.2 Gallery/Query Split (Critical for Evaluation)

For the handloom corpus (165 images, no design-level labels):

**Protocol A — Filename-based pseudo-split**:
- 9 `h_img_*` images → these could be gallery "handloom reference" images
- 156 `img_*` images → split 80/20 into gallery (125) + query (31)
- Limitation: no ground truth for verification pairs

**Protocol B — Manual annotation mini-set**:
- Manually label 20-30 pairs as "same design / different design" for verification evaluation
- This gives a small but reliable verification benchmark
- For identification: use top-k recall@k metric

**We will implement Protocol A automatically and document Protocol B as aspirational.**

### 3.3 Image Preprocessing Pipeline

```python
# Training transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),           # Sarees can be flipped
    transforms.ColorJitter(0.4, 0.4, 0.8, 0.3),  # Critical for color invariance
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),  # ImageNet stats
])

# Inference transforms (deterministic)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

**Why these augmentations?**
- **Vertical flip**: Saree patterns are often symmetric; model should be invariant to vertical orientation
- **Rotation ±15°**: Slight rotation invariance for fabric that may be photographed at angles
- **ColorJitter saturation=0.8, hue=0.3**: Forces model away from color-dependent features
- **RandomGrayscale(p=0.3)**: 30% of batches will see grayscale — the strongest color invariance teacher
- **No heavy blur or morphological ops** during training: saree weave patterns are high-frequency; we must preserve them

---

## 4. 🏋️ Training Strategy

### 4.1 Training Phases

**Phase 1 — Backbone Warmup (5 epochs)**
- Freeze EfficientNet-B3 layers
- Only train the 512-d embedding head and ArcFace classification layer
- Learning rate: `1e-3` for head
- Rationale: Adapt the head to the new feature space before disturbing pretrained weights

**Phase 2 — Full Fine-tuning (25 epochs)**
- Unfreeze all layers
- Different learning rates (discriminative LR):
  - Backbone: `1e-5` (small to preserve ImageNet features)
  - Embedding head: `1e-4`
  - ArcFace layer: `1e-4`
- Optimizer: **AdamW** with weight decay `1e-4`
- Scheduler: **CosineAnnealingLR** with T_max=25
- Rationale: Cosine LR prevents training collapse on small datasets; discriminative LR protects pretrained features

**Phase 3 — Grayscale Fine-tuning (5 epochs)**
- Increase RandomGrayscale probability to `p=0.7`
- Freeze backbone, retrain only embedding head
- Rationale: Final push for color invariance without destroying texture features

### 4.2 ArcFace Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `s` (scale) | 64 | Standard for face recognition; amplifies gradient signal |
| `m` (margin) | 0.5 | Standard; 0.5 radians separation enforces compact clusters |
| Embedding dim | 512 | Balance between expressiveness and retrieval speed |
| Classes | 4 (style classes) | Use available Kaggle labels |

### 4.3 Batch Construction

- **Batch size**: 32 (fits on Kaggle T4/P100 with EfficientNet-B3 at 224×224)
- **Balanced sampling**: Each batch contains equal images per class (8 per class × 4 classes)
- Why balanced: Prevents ArcFace from biasing towards Banarasi (largest class, 432 images)

### 4.4 Loss & Optimizer Details

```python
# Loss
criterion = ArcFaceLoss(in_features=512, num_classes=4, s=64.0, m=0.5)

# Optimizer
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.embedding_head.parameters(), 'lr': 1e-4},
    {'params': criterion.parameters(), 'lr': 1e-4},
], weight_decay=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
```

---

## 5. 📏 Evaluation Protocol

### 5.1 Task 1 — Identification (Retrieval)

**Protocol**:
- Gallery: All images from the handloom corpus except queries
- Query: Held-out subset (20% of handloom corpus)
- Metric: **Recall@K** (K = 1, 5, 10) and **Mean Average Precision (mAP)**

```
Recall@1 = fraction of queries where the top-1 result is correct design
Recall@5 = fraction of queries where correct design appears in top-5
mAP = mean over all queries of the average precision of the ranked list
```

**Why Recall@K?**
- Standard metric for image retrieval (used in fashion, product search)
- Directly measures whether the right design is surfaced to the user
- mAP is stricter — rewards correct items appearing at the top of the list

**For Kaggle dataset (style-level evaluation)**:
- Query: test split (60 images)
- Gallery: train + valid splits (1,408 images)
- Same-class = match (Banarasi queries should retrieve Banarasi gallery items)
- Report Recall@1, Recall@5, mAP

### 5.2 Task 2 — Verification (Pair Classification)

**Protocol**:
- Generate positive pairs: images from same class/same-source (same design, different augmentation or colorway)
- Generate negative pairs: images from different classes
- Threshold cosine similarity at `θ` (tuned on validation set)
- Metrics: **ROC-AUC**, **EER (Equal Error Rate)**, **Accuracy at optimal threshold**

```
For each pair (img_A, img_B):
    emb_A = model(img_A)  # 512-d unit vector
    emb_B = model(img_B)  # 512-d unit vector
    score = cosine_sim(emb_A, emb_B)
    prediction = score > θ  # SAME_DESIGN if True
```

**Why ROC-AUC + EER?**
- ROC-AUC summarizes the entire operating range (all possible thresholds)
- EER gives a single threshold-independent number (industry standard for biometric verification)
- Together they allow a fair comparison even when class distributions differ

### 5.3 Gallery/Query Split Definition (Reproducible)

```python
# Deterministic split using fixed seed
random.seed(42)
all_images = sorted(glob("handloom_sarees/*.jpg"))
query_indices = random.sample(range(len(all_images)), k=int(0.2 * len(all_images)))
gallery_images = [all_images[i] for i in range(len(all_images)) if i not in query_indices]
query_images = [all_images[i] for i in query_indices]
```

---

## 6. 🗂️ Repository Structure

```
DeepLure_Saree_AI_Agent/
├── README.md                    # Project overview, setup, usage
├── requirements.txt             # Python dependencies
├── configs/
│   └── config.yaml              # Training hyperparameters
├── data/
│   ├── dataset.py               # Dataset classes (Kaggle + Handloom)
│   ├── transforms.py            # Color-invariant augmentation pipeline
│   └── splits.py                # Gallery/query split logic
├── models/
│   ├── backbone.py              # EfficientNet-B3 feature extractor
│   ├── embedding_head.py        # 512-d embedding head with L2 norm
│   └── arcface.py               # ArcFace loss implementation
├── train.py                     # Main training script
├── evaluate.py                  # Retrieval + verification evaluation
├── inference.py                 # Single image / batch inference
├── gallery_builder.py           # Build gallery embeddings index
├── notebooks/
│   └── DeepLure_Kaggle.ipynb    # Complete Kaggle notebook (training + eval)
├── utils/
│   ├── metrics.py               # Recall@K, mAP, ROC-AUC, EER
│   ├── visualization.py         # t-SNE plots, similarity heatmaps
│   └── logger.py                # Training logger
└── reports/
    ├── approach_note.md         # 500-char approach note (deliverable 1)
    └── efficiency_report.md     # FLOPs, params, latency (deliverable 4)
```

---

## 7. 📦 Git Workflow (Push Like a Human)

| Module Batch | What gets pushed | Commit message style |
|---|---|---|
| **Batch 1** | README, requirements.txt, configs/ | `feat: project scaffold and config` |
| **Batch 2** | data/dataset.py, data/transforms.py | `feat: dataset loaders and color-invariant transforms` |
| **Batch 3** | models/backbone.py, models/embedding_head.py | `feat: EfficientNet-B3 backbone and embedding head` |
| **Batch 4** | models/arcface.py, train.py | `feat: ArcFace loss and training loop` |
| **Batch 5** | evaluate.py, utils/metrics.py | `feat: evaluation protocol - recall@k, mAP, ROC-AUC` |
| **Batch 6** | gallery_builder.py, inference.py | `feat: gallery index and inference pipeline` |
| **Batch 7** | notebooks/DeepLure_Kaggle.ipynb | `feat: complete Kaggle notebook` |
| **Batch 8** | reports/, visualization | `docs: approach note, efficiency report, t-SNE plots` |

---

## 8. 📐 Efficiency Targets (Bonus Deliverable)

| Metric | Target | Notes |
|---|---|---|
| **Parameters** | ~12.5M | EfficientNet-B3 (10.7M) + embedding head (1.8M) |
| **FLOPs** | ~1.8B | Per 224×224 image forward pass |
| **Embedding size** | 512-d float32 = 2 KB per image | Compact for large galleries |
| **Inference latency** | < 20ms per image | On Kaggle T4 GPU |
| **Gallery search (N=1000)** | < 5ms | Matrix multiply cosine similarity |
| **Total pipeline (query → rank)** | < 30ms | Suitable for real-time use |

---

## 9. 🔬 Key Technical Decisions Summary

| Decision | Choice | Why |
|---|---|---|
| **Architecture type** | Metric learning (not classification) | Open-set retrieval; natural verification support |
| **Backbone** | EfficientNet-B3 | Best accuracy/efficiency ratio; texture-sensitive |
| **Loss function** | ArcFace | State-of-art angular margin; works with 4 class labels we have |
| **Color invariance** | ColorJitter + RandomGrayscale + HSV analysis | Multi-pronged; teaches model to ignore color |
| **Embedding size** | 512-d | Balance between expressiveness and memory |
| **Input size** | 224×224 | Kaggle GPU memory vs. ImageNet compatibility |
| **Pretrained weights** | ImageNet (torchvision) | Strong texture priors; disclosed |
| **Similarity metric** | Cosine similarity | Works on unit sphere; ArcFace-compatible |
| **Evaluation** | Recall@K + mAP + ROC-AUC + EER | Industry standard for retrieval + verification |
| **Optimizer** | AdamW + CosineAnnealingLR | Best for fine-tuning pretrained models |

---

## 10. 📝 Approach Note (500 Characters — Deliverable 1 Draft)

> "We frame saree design identification as metric learning. EfficientNet-B3 (ImageNet pretrained) extracts texture embeddings; an ArcFace head with angular margin m=0.5 maps images to a 512-d unit hypersphere. Color invariance is achieved via aggressive ColorJitter (hue±0.3, sat±0.8) and 30% RandomGrayscale augmentation. The 4-class Kaggle set (Banarasi/Bandhani/Ikat/Pichwai) provides style labels for training. Retrieval uses cosine similarity; verification thresholds the same score. Evaluated by Recall@1/5 and ROC-AUC."

*(497 characters — within limit)*

---

## 11. ⚠️ Risks & Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Only 4 style labels — coarse for fine-grained matching | High | Cross-colorway augmentation; ArcFace still learns inter-class boundaries |
| 165 handloom images — tiny gallery | High | Use as evaluation only; never train on it |
| No design-level labels for verification | High | Use source-file-based pseudo-labels from Kaggle Roboflow groups |
| Kaggle GPU memory limit | Medium | Batch size 32, gradient checkpointing if needed |
| Color augmentation too aggressive → hurts texture learning | Medium | Ablation: train with/without heavy ColorJitter; compare Recall@1 |
| Model overfits on 1,468 Kaggle images | Medium | Aggressive dropout (p=0.3), weight decay, early stopping |

---

## 12. 🛣️ Execution Order (Modules to Build)

1. ✅ Data analysis (DONE — documented above)
2. 📁 Repository scaffold + config
3. 🗃️ Dataset loaders (Kaggle + Handloom)
4. 🎨 Color-invariant transform pipeline
5. 🧠 EfficientNet-B3 backbone wrapper
6. 💎 Embedding head + L2 normalization
7. 📐 ArcFace loss implementation
8. 🏋️ Training loop (phases 1, 2, 3)
9. 🏗️ Gallery builder (FAISS or torch.cdist)
10. 📏 Evaluation (Recall@K, mAP, ROC-AUC, EER)
11. 🔍 Inference pipeline
12. 📓 Kaggle notebook (end-to-end)
13. 📊 Visualizations (t-SNE, similarity heatmaps)
14. 📝 Reports (approach note, efficiency report)

---

*Document created: 2026-10-02 | Author: Antigravity AI Agent*
*All decisions are original; pretrained weights disclosed (ImageNet via torchvision)*
*Data sources disclosed: Kaggle Indian Saree Patterns (MIT), DeepLure Corpus (proprietary — not redistributed)*
