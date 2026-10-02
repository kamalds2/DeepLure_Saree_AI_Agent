"""
models/backbone.py
==================
EfficientNet-B3 feature extractor for DeepLure Saree AI.

WHY EFFICIENTNET-B3?
    See DeepLure_Architecture_Plan.md §2.2 for full justification.

    Short version:
      - 12M parameters vs. ResNet-50's 25M — half the size
      - ImageNet top-1: 81.6% (ResNet-50: 76.1%)
      - 1.8B FLOPs vs. ResNet-50's 4.1B — fits Kaggle free GPU comfortably
      - Compound scaling (width + depth + resolution) → better texture sensitivity
        than purely deep ResNets — critical for saree weave pattern matching
      - B3 (not B0/B1): better features; B4+ is overkill for 1,468 images

WHAT THIS MODULE DOES:
    1. Loads EfficientNet-B3 with ImageNet pretrained weights
    2. Removes the original classifier head (we replace with ArcFace + embedding)
    3. Returns [batch, 1536] feature vectors from Global Average Pooling
    4. Supports selective freezing for phase 1 (warmup) and phase 3 (fine-tuning)

FEATURE DIM: 1536
    EfficientNet-B3 last conv block outputs 1536 channels before GAP.
    This is the input to our 512-d embedding projection head.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNetB3Backbone(nn.Module):
    """
    EfficientNet-B3 feature extractor (classifier head removed).

    Outputs a [batch, 1536] feature vector after Global Average Pooling.
    The GAP layer is preserved because it spatially averages the feature map,
    making the features location-invariant — exactly what we want for
    texture pattern recognition (the weave pattern repeats across the fabric).

    Args:
        pretrained: If True, loads ImageNet weights (default: True)
                    DISCLOSURE: Uses torchvision ImageNet pretrained weights.
    """

    FEATURE_DIM = 1536  # EfficientNet-B3 output channels before classifier

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B3
        # WHY IMAGENET_1K_V1? It's the standard pretrained checkpoint.
        # Alternatives like IMAGENET_1K_V2 give marginally better accuracy
        # but are identical for our fine-tuning purposes.
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b3(weights=weights)

        # Keep everything EXCEPT the classifier head
        # base_model.features: all convolutional blocks
        # base_model.avgpool: AdaptiveAvgPool2d(1,1)  → gives [B, 1536, 1, 1]
        # base_model.classifier: [Dropout, Linear(1536, 1000)] ← REMOVE THIS
        self.features = base_model.features
        self.avgpool  = base_model.avgpool

        if pretrained:
            print(f"[backbone.py] Loaded EfficientNet-B3 with ImageNet weights")
            print(f"  Feature dim: {self.FEATURE_DIM}")
        else:
            print(f"[backbone.py] EfficientNet-B3 initialized from scratch (no pretrain)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, 224, 224]

        Returns:
            features: [B, 1536] — flattened GAP output
        """
        # Run through all EfficientNet blocks
        x = self.features(x)         # [B, 1536, H', W'] where H'=W'=7 for 224px input

        # Global Average Pooling → [B, 1536, 1, 1]
        # WHY GAP? It makes the output spatially invariant. The weave pattern
        # at different positions in the image should all contribute equally
        # to the final feature vector.
        x = self.avgpool(x)

        # Flatten → [B, 1536]
        x = torch.flatten(x, 1)
        return x

    def freeze(self):
        """
        Freeze all backbone parameters.
        Used in Phase 1 (warmup) and Phase 3 (color-invariance fine-tuning).

        WHY FREEZE? During phase 1, we don't want to disturb ImageNet pretrained
        features before the embedding head has had a chance to adapt. During phase 3,
        we want to only update the embedding head's color invariance without
        risking catastrophic forgetting of texture features.
        """
        for param in self.parameters():
            param.requires_grad = False
        print("[backbone.py] Backbone FROZEN")

    def unfreeze(self):
        """
        Unfreeze all backbone parameters.
        Used at the start of Phase 2 (full fine-tuning).
        """
        for param in self.parameters():
            param.requires_grad = True
        print("[backbone.py] Backbone UNFROZEN for full fine-tuning")

    def freeze_bn(self):
        """
        Keep BatchNorm layers frozen (in eval mode) even when unfreezing backbone.

        WHY? With small batches (32) fine-tuning BatchNorm can hurt — the running
        stats from ImageNet pretraining are better calibrated than what we'd get
        from a few thousand saree images. Keeping BN frozen is a common
        fine-tuning trick for small datasets.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        print("[backbone.py] BatchNorm layers FROZEN (eval mode)")

    def count_parameters(self) -> int:
        """Return total parameter count for efficiency report."""
        return sum(p.numel() for p in self.parameters())
