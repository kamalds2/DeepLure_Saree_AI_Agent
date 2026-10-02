# Approach Note — DeepLure Saree AI Agent
## Deliverable 1 (≤ 500 characters)

> We frame saree design identification as metric learning. EfficientNet-B3 (ImageNet pretrained) extracts texture embeddings; an ArcFace head with angular margin m=0.5 maps images to a 512-d unit hypersphere. Color invariance is achieved via aggressive ColorJitter (hue±0.3, sat±0.8) and 30% RandomGrayscale augmentation. The 4-class Kaggle set (Banarasi/Bandhani/Ikat/Pichwai) provides style labels for training. Retrieval uses cosine similarity; verification thresholds the same score. Evaluated by Recall@1/5 and ROC-AUC.

**Character count: 497** ✅

---

## Extended Rationale (for live defense)

### Architecture Choice
- **Metric learning over classification**: The task is open-set retrieval — the system must handle unseen designs at test time. A softmax classifier cannot generalize beyond its training classes; metric learning produces a generalizable embedding space.
- **EfficientNet-B3**: 12M parameters, 1.8B FLOPs, 81.6% ImageNet top-1. Best accuracy/compute ratio for texture-sensitive tasks on Kaggle free GPU. Compound scaling (width×depth×resolution) gives richer texture features than ResNet at lower parameter cost.
- **ArcFace loss**: Angular margin loss forces same-class embeddings into compact cones on the unit sphere. Angular separation (vs. cosine/Euclidean) is geometrically uniform — all classes have equal angular "territory". Works with the 4 style labels available.

### Pre-processing Pipeline
1. `Resize(256)` → `RandomCrop(224)`: Random crop adds spatial augmentation while keeping 224px for ImageNet compatibility
2. `RandomHorizontalFlip + RandomVerticalFlip (p=0.5)`: Saree designs are symmetric; orientation-invariant
3. `RandomRotation(±15°)`: Product photography angle variation; capped at 15° to preserve weave periodicity
4. `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.8, hue=0.3)`: **Core color invariance teacher** — high saturation and hue jitter forces the network away from color-dependent features
5. `RandomGrayscale(p=0.3 → 0.7)`: In 30% of Phase 1+2 training, images appear grayscale — the strongest color-invariance signal
6. `Normalize(ImageNet mean/std)`: Required for pretrained EfficientNet-B3

### Post-processing (Inference)
- L2 normalization to unit sphere → cosine similarity = dot product (fast)
- Gallery embeddings pre-computed and saved as `.npy` → < 1ms search for N=1000

### Training Strategy
| Phase | Epochs | What changes | Why |
|---|---|---|---|
| 1 | 5 | Backbone frozen, head LR=1e-3 | Let new head adapt without disrupting ImageNet features |
| 2 | 25 | Full model, backbone LR=1e-5, head LR=1e-4 | Discriminative LR: gentle backbone nudge, aggressive head update |
| 3 | 5 | Backbone re-frozen, grayscale p=0.7 | Color-invariance calibration without catastrophic forgetting |

- **AdamW** (not Adam): Weight decay regularization prevents overfitting on small dataset (1,468 images)
- **CosineAnnealingLR**: Smooth LR decay; avoids sharp drops that destabilize fine-tuning
- **WeightedRandomSampler**: Corrects Banarasi class imbalance (432 vs 279 images in other classes)
- **BatchNorm frozen during Phase 2**: Running statistics from ImageNet pretraining are better calibrated than what 1,468 saree images can provide
- **Gradient clipping (max_norm=1.0)**: Prevents gradient explosion during early fine-tuning
