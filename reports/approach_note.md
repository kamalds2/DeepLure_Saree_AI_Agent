# Approach Note — DeepLure Saree AI Agent
## Deliverable 1 (≤ 500 characters)

> We frame saree design identification as color-invariant visual metric learning. A pretrained ResNet50 backbone extracts spatial motif features, mapped via a projection head to 512-d unit hypersphere embeddings. Self-supervised contrastive learning (SimCLR InfoNCE, tau=0.07) pairs two distinct color-perturbed views of each saree, forcing representations to encode weave structure and motifs independent of palette. Retrieval uses cosine similarity on the gallery index, evaluated on invariance and latency.

**Character count: 494** ✅

---

## Extended Rationale (for live defense)

### 1. Visual Design Identity Over Filenames / Categories
- **Core Principle**: Saree designs are recognized purely from visual motifs, weave textures, border structures, and geometric layouts. Filenames and folder names never enter the model.
- **Why Self-Supervised Two-View Contrastive Learning?**: The Kaggle dataset contains single images per saree across 4 broad categories (`Banarasi`, `Bandhani`, `Ikat`, `Pichwai`), which are coarse styles rather than fine-grained design IDs. Generating two colorway-perturbed views of the same original image creates legitimate positive pairs with identical weave structures but distinct palettes, teaching the model pure color-invariance.

### 2. Architecture & Normalization
- **Pretrained Vision Backbone**: ResNet-50 / EfficientNet initialized with ImageNet weights to leverage high-fidelity low-level edge, gradient, and texture filters.
- **512-D L2-Normalized Embedding**: Features are projected and normalized to the unit hypersphere ($S^{511}$), ensuring that cosine similarity is equivalent to a high-speed dot product.

### 3. Training & Optimization Strategy
- **InfoNCE Temperature $\tau = 0.07$**: Sharpens the contrastive penalty, pushing hard negative visual patterns apart while pulling identical weave motifs together.
- **AdamW + CosineAnnealingLR**: Prevents overfitting on the 1,468-image dataset while ensuring smooth learning rate decay.
- **Gradient Clipping ($1.0$)**: Eliminates gradient instability during fine-tuning.

### 4. Retrieval & Efficiency
- Gallery embeddings pre-indexed as a contiguous tensor for sub-millisecond retrieval.
- 512 float32 values per image = **2 KB per gallery index entry**.
