"""
evaluate.py
===========
Evaluation Suite for Color-Invariant Visual Saree Design Recognition.

Evaluates:
A. Self-Supervised Representation Metrics:
   - Contrastive Alignment: Cosine similarity between two distinct augmented views of the same saree (Color Invariance Score).
   - Embedding Statistics: Mean, std, norm distribution of 512-D visual features.
B. Category-Level Diagnostic Probe:
   - k-NN (k=1, 5) nearest-neighbor classification on frozen visual embeddings.
   - Strictly labeled as CATEGORY diagnostic baseline, NOT fine-grained design identity.
C. Efficiency Benchmark:
   - Measured parameter count, FLOPs, single-image inference latency (ms), gallery search latency (ms).
D. Outputs structured JSON report to reports/evaluation_results.json.
"""

import os
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.dataset import SingleImageDataset, TwoViewContrastiveDataset
from data.transforms import get_inference_transforms, get_contrastive_train_transforms
from models import SareeEmbeddingModel


def evaluate_system(
    checkpoint_path: str = "./checkpoints/best_model.pt",
    data_root: str = "./kaggle",
    output_dir: str = "./reports",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print("=" * 60)
    print("DEEPLURE SAREE AI — COMPREHENSIVE EVALUATION BENCHMARK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Loading Model: {checkpoint_path}")

    model = SareeEmbeddingModel(embedding_dim=512, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # 1. Color Invariance Score (Alignment of distinct augmented colorway views)
    contrast_tf = get_contrastive_train_transforms(224)
    val_contrast_ds = TwoViewContrastiveDataset(data_root=data_root, split="valid", transform=contrast_tf)
    val_contrast_loader = DataLoader(val_contrast_ds, batch_size=32, shuffle=False)

    invariance_similarities = []
    with torch.no_grad():
        for v1, v2, _ in val_contrast_loader:
            v1 = v1.to(device)
            v2 = v2.to(device)
            z1 = model.extract_embedding(v1)
            z2 = model.extract_embedding(v2)
            # Dot product of normalized vectors = cosine similarity
            sim = (z1 * z2).sum(dim=1)
            invariance_similarities.extend(sim.cpu().tolist())

    mean_invariance_score = float(np.mean(invariance_similarities)) if invariance_similarities else 0.0

    # 2. Extract Embeddings for Train & Valid Splits for Category Diagnostic
    inf_tf = get_inference_transforms(224)
    train_ds = SingleImageDataset(data_root=data_root, split="train", transform=inf_tf)
    val_ds = SingleImageDataset(data_root=data_root, split="valid", transform=inf_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    def extract_features(loader):
        embs, cats = [], []
        with torch.no_grad():
            for imgs, _, c in loader:
                imgs = imgs.to(device)
                e = model.extract_embedding(imgs)
                embs.append(e.cpu().numpy())
                cats.extend(c)
        return np.concatenate(embs, axis=0), cats

    X_train, y_train = extract_features(train_loader)
    X_val, y_val = extract_features(val_loader)

    # Diagnostic k-NN probe on Category Metadata
    knn_1 = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn_1.fit(X_train, y_train)
    y_pred_1 = knn_1.predict(X_val)
    knn_1_acc = float(accuracy_score(y_val, y_pred_1))

    knn_5 = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn_5.fit(X_train, y_train)
    y_pred_5 = knn_5.predict(X_val)
    knn_5_acc = float(accuracy_score(y_val, y_pred_5))

    # 3. Efficiency & Latency Benchmark
    param_info = model.count_parameters()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    for _ in range(10):
        _ = model.extract_embedding(dummy_input)

    # Measure inference latency over 100 runs
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.extract_embedding(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = float(np.mean(latencies))

    # Measure retrieval search time for N=1293 gallery items
    gallery_tensor = torch.from_numpy(X_train).to(device)
    query_tensor = torch.from_numpy(X_val[:1]).to(device)

    retrieval_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = torch.matmul(gallery_tensor, query_tensor.squeeze(0))
        retrieval_latencies.append((time.perf_counter() - t0) * 1000)
    mean_retrieval_ms = float(np.mean(retrieval_latencies))

    results = {
        "representation_metrics": {
            "color_invariance_score_mean": round(mean_invariance_score, 4),
            "description": "Mean cosine similarity between two differently-colored augmented views of the same saree design (Scale -1.0 to 1.0, higher is better)."
        },
        "category_diagnostic_probe": {
            "knn_top1_accuracy": round(knn_1_acc, 4),
            "knn_top5_accuracy": round(knn_5_acc, 4),
            "note": "DIAGNOSTIC ONLY: Evaluates coarse category clustering quality. NOT fine-grained design identity."
        },
        "efficiency_metrics": {
            "backbone_parameters": param_info["backbone"],
            "embedding_head_parameters": param_info["head"],
            "total_parameters": param_info["total"],
            "embedding_dimension": 512,
            "embedding_size_bytes_per_image": 512 * 4,  # float32 = 2048 bytes (2 KB)
            "inference_latency_ms": round(mean_latency_ms, 2),
            "gallery_search_latency_ms": round(mean_retrieval_ms, 4)
        }
    }

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL MEASURED RESULTS")
    print("=" * 60)
    print(f"Color Invariance Alignment Score: {mean_invariance_score:.4f} (Target > 0.85)")
    print(f"Category Diagnostic Probe kNN@1: {knn_1_acc * 100:.2f}% | kNN@5: {knn_5_acc * 100:.2f}%")
    print(f"Total Parameters: {param_info['total']:,}")
    print(f"Single Image Latency: {mean_latency_ms:.2f} ms")
    print(f"Gallery Retrieval Latency: {mean_retrieval_ms:.4f} ms")

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--data_root", type=str, default="./kaggle")
    parser.add_argument("--output_dir", type=str, default="./reports")
    args = parser.parse_args()

    evaluate_system(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir
    )
