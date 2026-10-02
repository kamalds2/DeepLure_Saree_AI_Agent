"""
data/dataset.py
===============
Dataset classes for Color-Invariant Visual Saree Design Recognition.

STRICT DESIGN IDENTITY RULES:
- Images are recognized purely by VISUAL CONTENT (motifs, weave texture, layout).
- Filenames, folder names, and file paths are NEVER treated as design IDs.
- Category names (Banarasi, Bandhani, Ikat, Pichwai) are preserved as metadata
  for diagnostic probes only, never as fine-grained design identities.
- For self-supervised training, positive pairs are formed from two distinct augmented
  views of the SAME underlying image. Different images form the negative pairs.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class TwoViewContrastiveDataset(Dataset):
    """
    Dataset for Self-Supervised Color-Invariant Contrastive Learning (SimCLR / InfoNCE).

    For every index i, applies two distinct stochastic augmentations to the same image,
    returning (view_1, view_2).
    - view_1 and view_2 share the exact weave design, motif layout, and geometry.
    - view_1 and view_2 differ in colorway (hue/saturation/brightness jitter, grayscale) and viewpoint.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, str]] = []

        split_dir = self.data_root / split
        if split_dir.exists():
            for p in split_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append({
                        "path": str(p),
                        "category_metadata": p.parent.name
                    })

        print(f"[TwoViewContrastiveDataset] Split: {split} | Total Images: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]
        img_path = item["path"]

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        if self.transform is not None:
            view_1 = self.transform(img)
            view_2 = self.transform(img)
        else:
            raise ValueError("Contrastive dataset requires a stochastic transform pipeline.")

        return view_1, view_2, item["category_metadata"]


class SingleImageDataset(Dataset):
    """
    Standard single-image dataset for validation, gallery building, and retrieval testing.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "valid",
        transform: Optional[Callable] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, str]] = []

        split_dir = self.data_root / split
        if split_dir.exists():
            for p in split_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append({
                        "path": str(p),
                        "category_metadata": p.parent.name
                    })

        print(f"[SingleImageDataset] Split: {split} | Total Images: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        item = self.samples[idx]
        img_path = item["path"]

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        if self.transform is not None:
            tensor_img = self.transform(img)
        else:
            tensor_img = img

        return tensor_img, img_path, item["category_metadata"]
