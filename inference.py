"""
inference.py
============
Single-image and batch inference for DeepLure Saree AI.

WHAT THIS DOES:
    Given a query image (or directory of images), it:
    1. Loads the trained model + pre-built gallery index
    2. Extracts the query embedding
    3. Searches the gallery by cosine similarity
    4. Returns top-K matches with similarity scores

    Also supports VERIFICATION mode: given two images, returns
    cosine similarity score and a SAME/DIFFERENT decision.

PIPELINE:
    query.jpg → Preprocessing → EfficientNet-B3 → 512-d embedding
        ↓
    cosine_sim(query_emb, gallery_emb)  [fast matrix multiply]
        ↓
    sorted (desc) → top-K results with paths + scores

WHY IS THIS FAST?
    - Gallery embeddings are pre-computed and stored as numpy arrays
    - Gallery search = one matrix multiply: [1, 512] @ [N, 512].T = [1, N]
    - For N=1000 gallery images: < 1ms on CPU, < 0.1ms on GPU
    - Total query latency (preprocessing + model + search) < 30ms on T4 GPU

USAGE:
    # Single query vs handloom gallery:
    python inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --gallery_dir ./gallery_index \\
        --query path/to/saree.jpg \\
        --top_k 5

    # Batch query from a directory:
    python inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --gallery_dir ./gallery_index \\
        --query_dir ./my_queries/ \\
        --top_k 3

    # Verification (pair comparison):
    python inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --gallery_dir ./gallery_index \\
        --verify img_A.jpg img_B.jpg \\
        --threshold 0.65
"""

import os
import argparse
import time
import yaml
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple

from data import get_val_transforms
from models import build_model
from gallery_builder import load_gallery_index, load_checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to the query image
        transform: Deterministic val transform

    Returns:
        tensor: [1, 3, 224, 224] — batch of 1
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)           # [3, 224, 224]
    return tensor.unsqueeze(0)        # [1, 3, 224, 224]


@torch.no_grad()
def get_query_embedding(
    model,
    image_path: str,
    transform,
    device: torch.device,
) -> np.ndarray:
    """
    Extract L2-normalized 512-d embedding for a single query image.

    Args:
        model: SareeEmbeddingModel (eval mode)
        image_path: Path to query image
        transform: Val transforms
        device: CPU or CUDA

    Returns:
        embedding: [512] float32 numpy array (unit vector)
    """
    model.eval()
    tensor = preprocess_image(image_path, transform).to(device)
    emb    = model.extract_embedding(tensor)   # [1, 512]
    return emb.squeeze(0).cpu().numpy()         # [512]


def search_gallery(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_paths: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
) -> List[dict]:
    """
    Search the gallery for the top-K most similar images.

    Search is done via dot product (= cosine similarity for unit vectors).
    Time complexity: O(N × D) where N=gallery size, D=512.
    For N=1000: ~0.5ms on CPU.

    Args:
        query_emb:     [512] query embedding (unit vector)
        gallery_emb:   [N, 512] gallery embeddings (unit vectors)
        gallery_paths: [N] file paths
        gallery_labels:[N] class labels (-1 if unknown)
        top_k:         Number of results to return

    Returns:
        List of dicts: [{"rank", "path", "score", "label"}, ...]
    """
    # Cosine similarity = dot product (both are unit vectors)
    similarities = gallery_emb @ query_emb           # [N]

    # Get top-K indices (descending)
    top_k_idx = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_k_idx, 1):
        results.append({
            "rank":  rank,
            "path":  str(gallery_paths[idx]),
            "score": float(similarities[idx]),
            "label": int(gallery_labels[idx]),
        })
    return results


def verify_pair(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    threshold: float = 0.65,
) -> dict:
    """
    Verification: decide whether two images show the same design.

    Args:
        emb_a, emb_b: [512] unit-vector embeddings for image A and B
        threshold: Cosine similarity threshold (tuned on validation set)
                   Default 0.65 is a conservative starting point;
                   optimal value is found via ROC curve during evaluation.

    Returns:
        dict: {"score", "decision", "threshold"}
            decision: "SAME_DESIGN" or "DIFFERENT_DESIGN"
    """
    score = float(np.dot(emb_a, emb_b))   # cosine sim = dot product for unit vectors
    decision = "SAME_DESIGN" if score >= threshold else "DIFFERENT_DESIGN"
    return {
        "score":     score,
        "decision":  decision,
        "threshold": threshold,
    }


def print_results(query_path: str, results: List[dict]):
    """Pretty-print identification results."""
    print(f"\n{'─'*55}")
    print(f"  Query: {os.path.basename(query_path)}")
    print(f"{'─'*55}")
    print(f"  {'Rank':<6} {'Score':<8} {'Match'}")
    print(f"  {'────':<6} {'─────':<8} {'──────────────────────────────'}")
    for r in results:
        bar = "█" * int(r["score"] * 20)
        print(f"  #{r['rank']:<5} {r['score']:.4f}  {os.path.basename(r['path'])}")
        print(f"        {bar}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Main inference orchestration
# ---------------------------------------------------------------------------

def run_identification(args, model, transform, gallery_emb, gallery_paths, gallery_labels, device):
    """Run identification (retrieval) for one or many queries."""
    if args.query:
        # Single image
        query_paths = [args.query]
    elif args.query_dir:
        # Directory of images
        from glob import glob
        query_paths = sorted(glob(os.path.join(args.query_dir, "*.jpg")))
        if not query_paths:
            query_paths = sorted(glob(os.path.join(args.query_dir, "*.png")))
        print(f"[inference] Found {len(query_paths)} query images in {args.query_dir}")
    else:
        raise ValueError("Provide --query or --query_dir for identification mode.")

    for qpath in query_paths:
        t0 = time.perf_counter()

        query_emb = get_query_embedding(model, qpath, transform, device)
        results   = search_gallery(
            query_emb, gallery_emb, gallery_paths, gallery_labels, args.top_k
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        print_results(qpath, results)
        print(f"  ⏱  Total latency: {latency_ms:.1f} ms")

    return results


def run_verification(args, model, transform, device):
    """Run verification for a pair of images."""
    if len(args.verify) != 2:
        raise ValueError("--verify requires exactly 2 image paths")

    path_a, path_b = args.verify

    t0    = time.perf_counter()
    emb_a = get_query_embedding(model, path_a, transform, device)
    emb_b = get_query_embedding(model, path_b, transform, device)
    result = verify_pair(emb_a, emb_b, threshold=args.threshold)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'─'*55}")
    print(f"  VERIFICATION RESULT")
    print(f"{'─'*55}")
    print(f"  Image A: {os.path.basename(path_a)}")
    print(f"  Image B: {os.path.basename(path_b)}")
    print(f"  Cosine Similarity: {result['score']:.4f}")
    print(f"  Threshold:         {result['threshold']:.4f}")
    print(f"  Decision:          {result['decision']}")
    print(f"  ⏱  Latency: {latency_ms:.1f} ms")
    print(f"{'─'*55}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")

    # Load model
    model, criterion = build_model(cfg)
    model, criterion = load_checkpoint(model, criterion, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    transform = get_val_transforms(cfg["data"]["image_size"])

    # Load gallery
    gallery_emb, gallery_labels, gallery_paths = load_gallery_index(args.gallery_dir)

    if args.verify:
        run_verification(args, model, transform, device)
    else:
        run_identification(args, model, transform, gallery_emb, gallery_paths, gallery_labels, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLure Saree AI — Inference")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--gallery_dir", type=str, default="./gallery_index",
                        help="Directory with pre-built gallery index")
    # Identification
    parser.add_argument("--query",       type=str, default=None,
                        help="Single query image path")
    parser.add_argument("--query_dir",   type=str, default=None,
                        help="Directory of query images")
    parser.add_argument("--top_k",       type=int, default=5)
    # Verification
    parser.add_argument("--verify",      type=str, nargs=2, default=None,
                        metavar=("IMG_A", "IMG_B"),
                        help="Two images for verification")
    parser.add_argument("--threshold",   type=float, default=0.65,
                        help="Cosine similarity threshold for verification")
    args = parser.parse_args()
    main(args)
