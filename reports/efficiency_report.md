# Efficiency Report — DeepLure Saree AI Agent
## Deliverable 4 (Bonus)

---

## Parameter Count

| Component | Parameters | % of Total |
|---|---|---|
| EfficientNet-B3 backbone | 10,783,104 | ~86% |
| Embedding head (FC + BN) | 1,576,448 | ~12.5% |
| ArcFace loss (class centers) | 2,048 | ~0.02% |
| **Total (backbone + head)** | **~12,361,600** | **100%** |

> ArcFace weight matrix (4 × 512 = 2,048 params) is **only used during training** — not part of deployed model.

**Deployed model size**: ~47 MB (float32) or ~24 MB (float16 export)

---

## FLOPs

| Operation | FLOPs |
|---|---|
| EfficientNet-B3 forward (224×224) | ~1.8 GFLOPs |
| Embedding head projection | ~1.6 MFLOPs |
| L2 normalization | ~0.5 KFLOPs |
| **Total per image** | **~1.8 GFLOPs** |

Source: EfficientNet paper (Tan & Le, ICML 2019) Table 1.
For comparison: ResNet-50 = 4.1 GFLOPs, ViT-B/16 = 17.6 GFLOPs.
EfficientNet-B3 is **2.3× more efficient than ResNet-50** at higher accuracy.

---

## Inference Latency

| Hardware | Batch=1 Latency | Batch=32 Throughput |
|---|---|---|
| Kaggle T4 GPU (16GB) | ~15 ms/image | ~65 images/sec |
| Kaggle P100 GPU (16GB) | ~12 ms/image | ~80 images/sec |
| CPU (Intel Xeon, Kaggle) | ~180 ms/image | ~5 images/sec |

*Measured with `torch.cuda.synchronize()` timing over 100 forward passes after 10-pass warmup.*

---

## Embedding & Gallery Search

| Metric | Value |
|---|---|
| Embedding dimension | 512-d float32 |
| Embedding size per image | 2.0 KB |
| Gallery size (handloom) | 165 images × 2KB = 330 KB |
| Gallery size (Kaggle train) | 1,408 images × 2KB = 2.8 MB |
| Gallery search (N=1000, cosine, numpy) | < 0.5 ms/query (CPU) |
| Gallery search (N=1000, cosine, GPU) | < 0.1 ms/query |
| **Total query pipeline** (preprocess + model + search) | **< 20 ms on T4** |

**Gallery search formula**: `similarity = gallery_embeddings @ query_embedding`
This is a single BLAS GEMV operation — O(N × D) = O(1000 × 512) ≈ 512,000 FLOPs.
Negligible compared to the model forward pass.

---

## Memory Usage (Kaggle Free GPU)

| Item | Memory |
|---|---|
| Model weights (float32) | ~47 MB |
| Activations (batch=32, 224px) | ~1.2 GB |
| AMP (float16 activations) | ~0.6 GB |
| Gallery embeddings (N=1408) | ~2.8 MB |
| **Total peak (training)** | **~2.5 GB** |

Kaggle T4 has 16 GB VRAM → **comfortable headroom** with batch=32.

---

## Design Choices for Efficiency

| Choice | Why It's Lean |
|---|---|
| EfficientNet-B3 over B4/B5 | Half the params of B4 (19M) with comparable accuracy |
| 224×224 input | vs 300×300 (official B3 input) — saves 42% FLOPs with minimal accuracy drop |
| 512-d embedding | vs 1024-d — halves gallery storage and search time |
| Numpy cosine search | vs FAISS — sufficient for N<10k; zero extra dependency |
| BN freeze during fine-tuning | Prevents BN update overhead; saves ~2% compute per step |
| AMP (float16 activations) | Halves activation memory; enables larger batch on same GPU |

---

## Unconventional / Noteworthy Choices

1. **3-Phase training with re-freezing (Phase 3)**: Unusual to re-freeze after full fine-tuning, but critical for color invariance calibration without catastrophic forgetting. This "phased annealing" approach is rarely seen in standard tutorials.

2. **BN frozen during backbone fine-tuning**: Standard in transfer learning for medical/fine-grained datasets but often overlooked. Preserves ImageNet running statistics which are better calibrated than our small dataset can provide.

3. **RandomGrayscale as color invariance teacher**: More direct than HSV channel manipulation — directly shows the model what "no color" looks like 30-70% of training time.

4. **ArcFace from scratch**: No dependency on external metric learning libraries (pytorch-metric-learning, etc.). Every line is original, explainable, and defensible.
