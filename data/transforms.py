"""
data/transforms.py
==================
Realistic Color-Invariant and Spatial Augmentation Pipeline for Saree Design Retrieval.

Design Philosophy:
- Saree design identity is defined by motifs, weave structure, borders, and layout.
- Realistic color augmentation (brightness, contrast, saturation, hue jitter) teaches the
  model that color palette does not alter the fundamental textile design.
- Spatial transforms (flips, slight rotations, crops) provide scale and viewpoint robustness
  without destroying delicate repeating weave patterns.
"""

from torchvision import transforms
from typing import Tuple


def get_contrastive_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Two-view contrastive training transform.
    Applies realistic color perturbations and spatial crops to create
    two distinct visual views of the same saree design.
    """
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.30,
            saturation=0.40,
            hue=0.08
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic inference/validation transform: standard resize, center crop, normalize.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
