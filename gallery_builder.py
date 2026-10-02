"""
gallery_builder.py
==================
Build and save gallery embeddings for the DeepLure retrieval system.

WHAT THIS DOES:
    1. Loads all gallery images (handloom corpus OR Kaggle train+valid)
    2. Runs them through the trained model → 512-d L2-normalized embeddings
    3. Saves embeddings + metadata (paths, labels) to disk as .npy files

WHY SAVE THE GALLERY?
    At inference time (production use), you don't want to re-run all gallery
    images through the model every time a query arrives. Instead:
        - Run gallery_builder.py ONCE → saves gallery_embeddings.npy
        - inference.py loads the .npy → instant cosine search

    This is exactly how face recognition systems work:
        enroll (build gallery) → once
        identify (query)       → fast, repeated

USAGE:
    # Build gallery from handloom corpus:
    python gallery_builder.py \\
        --checkpoint checkpoints/best_model.pth \\
        --gallery_dir ./handloom_sarees \\
        --output_dir ./gallery_index

    # Build gallery from Kaggle train+valid (for style-level search):
    python gallery_builder.py \\
        --checkpoint checkpoints/best_model.pth \\
        --gallery_source kaggle \\
        --data_root ./kaggle \\
        --output_dir ./gallery_index

OUTPUT FILES:
    gallery_index/
        gallery_embeddings.npy   [N, 512] float32
        gallery_labels.npy       [N] int32 (-1 if unknown)
        gallery_paths.npy        [N] str   (absolute file paths)
        gallery_meta.yaml        metadata (model, date, num_images, embedding_dim)
"""

import os
import argparse
import yaml
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    KaggleSareeDataset,
    HandloomDataset,
    get_val_transforms,
    build_handloom_split,
)
from models import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(model, criterion, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion.load_state_dict(ckpt["criterion_state"])
    print(f"[gallery_builder] Loaded checkpoint: {path}")
    print(f"  Phase={ckpt['phase']}, Epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    return model, criterion


@torch.no_grad()
def extract_all_embeddings(
    model,
    image_paths: list,
    transform,
    batch_size: int,
    device: torch.device,
    labels: list = None,
) -> tuple:
    """
    Extract embeddings for a list of image paths.

    Args:
        model: SareeEmbeddingModel in eval mode
        image_paths: List of absolute paths to images
        transform: Validation transform (deterministic)
        batch_size: Batch size for extraction
        device: torch device
        labels: Optional list of integer labels (same length as image_paths)

    Returns:
        (embeddings, labels_arr, paths_arr):
            embeddings: [N, 512] float32 numpy array
            labels_arr: [N] int32 numpy array (-1 if no labels)
            paths_arr:  [N] object array of paths
    """
    # Build a simple dataset from paths
    dataset = HandloomDataset(
        image_paths=image_paths,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_emb = []
    model.eval()

    for batch in tqdm(loader, desc="  extracting embeddings"):
        imgs = batch[0].to(device)          # [B, 3, H, W]
        emb  = model.extract_embedding(imgs) # [B, 512]
        all_emb.append(emb.cpu().numpy())

    embeddings  = np.concatenate(all_emb, axis=0).astype(np.float32)  # [N, 512]
    labels_arr  = np.array(labels if labels else [-1] * len(image_paths), dtype=np.int32)
    paths_arr   = np.array(image_paths, dtype=object)

    print(f"  Gallery embeddings shape: {embeddings.shape}")
    print(f"  Embedding norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")

    return embeddings, labels_arr, paths_arr


def save_gallery_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    checkpoint_path: str,
    output_dir: str,
):
    """Save gallery to disk + write metadata YAML."""
    os.makedirs(output_dir, exist_ok=True)

    emb_path   = os.path.join(output_dir, "gallery_embeddings.npy")
    lbl_path   = os.path.join(output_dir, "gallery_labels.npy")
    paths_path = os.path.join(output_dir, "gallery_paths.npy")
    meta_path  = os.path.join(output_dir, "gallery_meta.yaml")

    np.save(emb_path,   embeddings)
    np.save(lbl_path,   labels)
    np.save(paths_path, paths)

    meta = {
        "num_images":    int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint":    os.path.abspath(checkpoint_path),
        "created_at":    datetime.datetime.now().isoformat(),
        "index_files": {
            "embeddings": emb_path,
            "labels":     lbl_path,
            "paths":      paths_path,
        },
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"\n[gallery_builder] Gallery index saved to: {output_dir}/")
    print(f"  {len(embeddings)} images | 512-d embeddings | "
          f"{embeddings.nbytes / 1024:.1f} KB on disk")
    print(f"  Metadata: {meta_path}")


def load_gallery_index(output_dir: str) -> tuple:
    """
    Load a previously built gallery index from disk.

    Returns:
        (embeddings, labels, paths):
            embeddings: [N, 512] float32
            labels:     [N] int32
            paths:      [N] object array of str
    """
    emb_path   = os.path.join(output_dir, "gallery_embeddings.npy")
    lbl_path   = os.path.join(output_dir, "gallery_labels.npy")
    paths_path = os.path.join(output_dir, "gallery_paths.npy")

    embeddings = np.load(emb_path)
    labels     = np.load(lbl_path)
    paths      = np.load(paths_path, allow_pickle=True)

    print(f"[gallery_builder] Loaded gallery: {len(embeddings)} images | "
          f"emb shape={embeddings.shape}")
    return embeddings, labels, paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gallery_builder] Device: {device}")

    model, criterion = build_model(cfg)
    model, criterion = load_checkpoint(model, criterion, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    val_transform = get_val_transforms(cfg["data"]["image_size"])
    bs            = cfg["training"]["batch_size"]

    if args.gallery_source == "handloom":
        # Use handloom corpus gallery split
        gallery_dir = args.gallery_dir or cfg["data"]["handloom_root"]
        gallery_paths, _ = build_handloom_split(
            gallery_dir,
            query_fraction=cfg["data"]["query_fraction"],
            seed=cfg["data"]["gallery_query_split_seed"],
        )
        labels = [-1] * len(gallery_paths)   # No ground truth for handloom

    elif args.gallery_source == "kaggle":
        # Use Kaggle train + valid as gallery
        from data import CLASS_TO_IDX
        data_root = args.data_root or cfg["data"]["kaggle_root"]
        train_ds  = KaggleSareeDataset(data_root, split="train", transform=val_transform)
        valid_ds  = KaggleSareeDataset(data_root, split="valid", transform=val_transform)
        gallery_paths = [p for p, _ in train_ds.samples] + [p for p, _ in valid_ds.samples]
        labels        = [l for _, l in train_ds.samples] + [l for _, l in valid_ds.samples]

    else:
        raise ValueError(f"Unknown gallery_source: {args.gallery_source}")

    print(f"[gallery_builder] Gallery size: {len(gallery_paths)} images")

    embeddings, labels_arr, paths_arr = extract_all_embeddings(
        model, gallery_paths, val_transform, bs, device, labels
    )

    save_gallery_index(
        embeddings, labels_arr, paths_arr,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLure — Build Gallery Index")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--config",          type=str, default="configs/config.yaml")
    parser.add_argument("--gallery_source",  type=str, default="handloom",
                        choices=["handloom", "kaggle"])
    parser.add_argument("--gallery_dir",     type=str, default=None,
                        help="Handloom directory (overrides config)")
    parser.add_argument("--data_root",       type=str, default=None,
                        help="Kaggle data root (overrides config)")
    parser.add_argument("--output_dir",      type=str, default="./gallery_index")
    args = parser.parse_args()
    main(args)
