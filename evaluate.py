"""
evaluate.py
===========
End-to-end evaluation script for DeepLure Saree AI.

Runs BOTH evaluation tasks:
    1. Identification (retrieval): Recall@1/5/10, mAP
    2. Verification (pair similarity): ROC-AUC, EER, Accuracy

USAGE:
    # Evaluate on Kaggle test set (style-level retrieval):
    python evaluate.py \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/config.yaml \\
        --mode kaggle

    # Evaluate on Handloom gallery/query split:
    python evaluate.py \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/config.yaml \\
        --mode handloom \\
        --handloom_dir ./handloom_sarees

HOW EVALUATION WORKS:
    1. Load best checkpoint from training
    2. Extract embeddings for all images (gallery + queries)
    3. Compute pairwise cosine similarity matrix
    4. Compute Recall@K and mAP for identification
    5. Build verification pairs and compute ROC-AUC + EER

WHY TWO MODES?
    Kaggle mode: Has ground truth style labels → clean quantitative evaluation
                 Query=test split (60 images), Gallery=train+valid (1408 images)

    Handloom mode: No labels → filename-based pseudo-evaluation
                   Query=20% held out, Gallery=80%
                   Verification pairs use h_img_* vs img_* prefix as proxy labels
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    KaggleSareeDataset,
    HandloomDataset,
    VerificationPairDataset,
    get_val_transforms,
    build_handloom_split,
)
from models import build_model
from utils.metrics import (
    recall_at_k,
    mean_average_precision,
    roc_auc_eer,
    verification_accuracy,
    print_retrieval_report,
    print_verification_report,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(model, criterion, checkpoint_path: str, device: torch.device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion.load_state_dict(checkpoint["criterion_state"])
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"[evaluate.py] Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained to phase {checkpoint['phase']}, epoch {checkpoint['epoch']}, "
          f"val_acc={val_acc:.4f}")
    return model, criterion


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    batch_size: int,
    device: torch.device,
    return_labels: bool = True,
) -> tuple:
    """
    Extract L2-normalized embeddings for all images in a dataset.

    Args:
        model: SareeEmbeddingModel (eval mode)
        dataset: Dataset with __getitem__ returning (img, label) or (img, label, name)
        batch_size: Batch size for extraction
        device: CUDA or CPU
        return_labels: Whether to collect labels

    Returns:
        (embeddings, labels): numpy arrays
            embeddings: [N, 512]
            labels:     [N] (or None if return_labels=False)
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_labels     = []

    for batch in tqdm(loader, desc="  extracting", leave=False):
        # Handle both (img, label) and (img, label, name) return formats
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch

        imgs = imgs.to(device)
        emb  = model.extract_embedding(imgs)   # [B, 512], already L2-normed
        all_embeddings.append(emb.cpu().numpy())
        if return_labels:
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 512]
    labels     = np.concatenate(all_labels,     axis=0) if return_labels else None
    return embeddings, labels


# ---------------------------------------------------------------------------
# Kaggle evaluation (style-level, labeled)
# ---------------------------------------------------------------------------

def evaluate_kaggle(model, criterion, cfg, device):
    """
    Evaluate on Kaggle dataset (4 style classes).

    Identification:
        Gallery = train + valid splits
        Query   = test split
        Match   = same style class

    Verification:
        Build positive/negative pairs from test set using class labels.
    """
    print("\n[evaluate.py] === KAGGLE EVALUATION (Style-level) ===")

    val_transform = get_val_transforms(cfg["data"]["image_size"])
    data_root     = cfg["data"]["kaggle_root"]
    bs            = cfg["training"]["batch_size"]

    # Gallery: train + valid
    train_ds = KaggleSareeDataset(data_root, split="train", transform=val_transform)
    valid_ds = KaggleSareeDataset(data_root, split="valid", transform=val_transform)
    test_ds  = KaggleSareeDataset(data_root, split="test",  transform=val_transform)

    # Extract gallery embeddings (train + valid combined)
    from torch.utils.data import ConcatDataset
    gallery_ds = ConcatDataset([train_ds, valid_ds])

    print(f"  Gallery: {len(gallery_ds)} images | Query: {len(test_ds)} images")

    # We need custom extraction for ConcatDataset (no single label accessor)
    # Extract separately and concatenate
    gallery_emb_train, gallery_lbl_train = extract_embeddings(model, train_ds, bs, device)
    gallery_emb_valid, gallery_lbl_valid = extract_embeddings(model, valid_ds, bs, device)
    gallery_emb    = np.concatenate([gallery_emb_train, gallery_emb_valid])
    gallery_labels = np.concatenate([gallery_lbl_train, gallery_lbl_valid])

    query_emb, query_labels = extract_embeddings(model, test_ds, bs, device)

    # --- Identification ---
    k_vals = cfg["evaluation"]["recall_k_values"]
    recall_dict = recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k_vals)
    map_score   = mean_average_precision(query_emb, gallery_emb, query_labels, gallery_labels)
    print_retrieval_report(recall_dict, map_score, prefix="  ")

    # --- Verification (on test set pairs) ---
    verif_ds = VerificationPairDataset(test_ds, num_pairs=500, pos_ratio=0.5, seed=42)
    verif_loader = DataLoader(verif_ds, batch_size=bs, shuffle=False, num_workers=2)

    all_scores, all_is_same = [], []
    model.eval()
    for img_a, img_b, is_same in tqdm(verif_loader, desc="  verification", leave=False):
        img_a, img_b = img_a.to(device), img_b.to(device)
        emb_a = model.extract_embedding(img_a).cpu().numpy()
        emb_b = model.extract_embedding(img_b).cpu().numpy()
        # Cosine similarity for each pair
        scores = np.einsum("nd,nd->n", emb_a, emb_b)   # dot product of unit vectors
        all_scores.append(scores)
        all_is_same.append(is_same.numpy())

    scores  = np.concatenate(all_scores)
    is_same = np.concatenate(all_is_same)

    roc_auc, eer, opt_threshold = roc_auc_eer(scores, is_same)
    acc = verification_accuracy(scores, is_same, opt_threshold)
    print_verification_report(roc_auc, eer, acc, opt_threshold, prefix="  ")

    return {
        "recall": recall_dict,
        "mAP": map_score,
        "roc_auc": roc_auc,
        "eer": eer,
        "verification_acc": acc,
        "threshold": opt_threshold,
    }


# ---------------------------------------------------------------------------
# Handloom evaluation (unlabeled, filename-based split)
# ---------------------------------------------------------------------------

def evaluate_handloom(model, cfg, device, handloom_dir: str):
    """
    Evaluate on handloom corpus (no labels).

    Uses filename-based split (see data/splits.py):
        h_img_* → gallery
        img_*   → 80% gallery, 20% query

    Identification: Ranked retrieval (we report Recall@1/5/10 using
    filename prefix as proxy "same class" — limited but transparent).
    """
    print("\n[evaluate.py] === HANDLOOM EVALUATION (Proxy labels) ===")

    val_transform = get_val_transforms(cfg["data"]["image_size"])
    bs = cfg["training"]["batch_size"]
    seed = cfg["data"]["gallery_query_split_seed"]

    gallery_paths, query_paths = build_handloom_split(
        handloom_dir,
        query_fraction=cfg["data"]["query_fraction"],
        seed=seed,
        output_csv=os.path.join(cfg["paths"]["reports_dir"], "handloom_split.csv"),
    )

    # Use filename prefix as pseudo-label: h_img_* = 0, img_* = 1
    # WHY? Only for reporting purposes; these are NOT true design labels.
    def prefix_label(path):
        return 0 if os.path.basename(path).startswith("h_img_") else 1

    gallery_ds = HandloomDataset(gallery_paths, transform=val_transform)
    query_ds   = HandloomDataset(query_paths,   transform=val_transform)

    gallery_emb, _ = extract_embeddings(model, gallery_ds, bs, device, return_labels=False)
    query_emb,   _ = extract_embeddings(model, query_ds,   bs, device, return_labels=False)

    # Pseudo labels for proxy recall
    gallery_labels = np.array([prefix_label(p) for p in gallery_paths])
    query_labels   = np.array([prefix_label(p) for p in query_paths])

    k_vals = cfg["evaluation"]["recall_k_values"]
    recall_dict = recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k_vals)
    map_score   = mean_average_precision(query_emb, gallery_emb, query_labels, gallery_labels)

    print("  ⚠️  Labels are filename-prefix pseudo-labels (h_img=0, img=1) — not true design IDs")
    print_retrieval_report(recall_dict, map_score, prefix="  ")

    return {"recall": recall_dict, "mAP": map_score}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate.py] Device: {device}")

    model, criterion = build_model(cfg)
    model, criterion = load_checkpoint(model, criterion, args.checkpoint, device)
    model     = model.to(device)
    criterion = criterion.to(device)

    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)

    if args.mode == "kaggle":
        results = evaluate_kaggle(model, criterion, cfg, device)
    elif args.mode == "handloom":
        hl_dir = args.handloom_dir or cfg["data"]["handloom_root"]
        results = evaluate_handloom(model, cfg, device, hl_dir)
    else:
        print("[evaluate.py] Running BOTH modes...")
        results_kaggle   = evaluate_kaggle(model, criterion, cfg, device)
        hl_dir = args.handloom_dir or cfg["data"]["handloom_root"]
        results_handloom = evaluate_handloom(model, cfg, device, hl_dir)
        results = {"kaggle": results_kaggle, "handloom": results_handloom}

    print("\n[evaluate.py] ✅ Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLure Saree AI — Evaluation")
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--mode",          type=str, default="both",
                        choices=["kaggle", "handloom", "both"])
    parser.add_argument("--handloom_dir",  type=str, default=None)
    parser.add_argument("--device",        type=str, default="cuda")
    args = parser.parse_args()
    main(args)
