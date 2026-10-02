"""
data/splits.py
==============
Gallery / Query split logic for the DeepLure handloom evaluation corpus.

WHY THIS MODULE EXISTS:
    The handloom corpus (165 images, no labels) serves as our evaluation arena.
    We need a reproducible, deterministic split into:
      - GALLERY: reference images that represent known designs
      - QUERY: held-out images that we search against the gallery

    The split must be:
      1. Reproducible (fixed seed=42)
      2. Stratified by filename prefix where possible (h_img vs img)
      3. Exportable as a CSV for transparency / reproducibility

DESIGN DECISIONS:
    - 80/20 gallery/query split: Standard retrieval benchmark ratio.
      More gallery images = richer search space.
      Enough query images (≈33) to get stable Recall@K estimates.
    - Seed=42: Completely arbitrary but fixed — matches config.yaml.
    - We keep h_img_* images in gallery by default (there are only 9 of them,
      likely "canonical" handloom reference images — best as gallery anchors).
"""

import os
import random
import csv
from glob import glob
from typing import Tuple, List


def build_handloom_split(
    handloom_dir: str,
    query_fraction: float = 0.20,
    seed: int = 42,
    output_csv: str = None,
) -> Tuple[List[str], List[str]]:
    """
    Split the handloom corpus into gallery and query sets.

    Strategy:
        - All h_img_* files → gallery (canonical handloom reference images)
        - img_* files → random 80/20 gallery/query split

    Args:
        handloom_dir: Path to directory containing all handloom JPG files
        query_fraction: Fraction of img_* images to use as queries (default 0.20)
        seed: Random seed for reproducibility (default 42)
        output_csv: If provided, saves the split as a CSV file for traceability

    Returns:
        (gallery_paths, query_paths): Two lists of absolute file paths
    """
    random.seed(seed)

    # Collect all images
    all_files = sorted(glob(os.path.join(handloom_dir, "*.jpg")))
    if not all_files:
        raise FileNotFoundError(
            f"No .jpg files found in {handloom_dir}. "
            "Make sure the handloom corpus is in the correct location."
        )

    # Separate by prefix
    # WHY? h_img_* likely represents "handloom" canonical samples — keep in gallery
    h_img_files = [f for f in all_files if os.path.basename(f).startswith("h_img_")]
    img_files   = [f for f in all_files if os.path.basename(f).startswith("img_")]
    other_files = [f for f in all_files if f not in h_img_files and f not in img_files]

    print(f"[splits.py] Found {len(all_files)} images total")
    print(f"  h_img_* (handloom canonical): {len(h_img_files)}")
    print(f"  img_*   (general corpus):     {len(img_files)}")
    print(f"  other:                         {len(other_files)}")

    # h_img_* → always gallery
    gallery_paths = list(h_img_files)

    # img_* → 80% gallery, 20% query
    shuffled_img = list(img_files)
    random.shuffle(shuffled_img)
    n_query = max(1, int(len(shuffled_img) * query_fraction))

    query_paths   = shuffled_img[:n_query]
    gallery_paths += shuffled_img[n_query:]

    # other → gallery by default
    gallery_paths += other_files

    print(f"  → Gallery: {len(gallery_paths)} images")
    print(f"  → Query:   {len(query_paths)} images")

    # Optionally save split to CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "role"])
            for p in gallery_paths:
                writer.writerow([p, "gallery"])
            for p in query_paths:
                writer.writerow([p, "query"])
        print(f"  → Split saved to {output_csv}")

    return gallery_paths, query_paths


def load_split_from_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Load a previously saved gallery/query split from CSV.

    Args:
        csv_path: Path to CSV file produced by build_handloom_split()

    Returns:
        (gallery_paths, query_paths)
    """
    gallery_paths, query_paths = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["role"] == "gallery":
                gallery_paths.append(row["path"])
            else:
                query_paths.append(row["path"])
    return gallery_paths, query_paths
