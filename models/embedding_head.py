"""
models/embedding_head.py
========================
Visual Saree Design Embedding Model (ResNet50 Backbone + 512-D L2-Normalized Head)

Architecture:
- Backbone: Pretrained ResNet50 (ImageNet weights for rich motif, texture, and edge filters).
- Feature Extraction: Global Average Pooling (2048-dim feature vector).
- Projection Head: Linear(2048, 1024) -> ReLU() -> Dropout(0.2) -> Linear(1024, 512)
- Normalization: L2 normalization to unit hypersphere S^511 (||z||_2 = 1.0).
- Distance Metric: Cosine Similarity == Dot Product on unit sphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict


class SareeEmbeddingModel(nn.Module):
    """
    Visual Saree Design Embedding Model.
    Extracts color-invariant 512-dimensional normalized embeddings from saree images.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace default classifier with Identity to access 2048-d pooled features
        feature_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Non-linear 512-D projection head
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image tensor [B, 3, 224, 224] -> L2-normalized embedding [B, 512].
        """
        features = self.backbone(x)  # [B, 2048]
        proj = self.embedding_head(features)  # [B, 512]
        embedding = F.normalize(proj, p=2, dim=1)  # L2 unit sphere projection
        return embedding

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode embedding extraction without gradients."""
        self.eval()
        return self.forward(x)

    def count_parameters(self) -> Dict[str, int]:
        """Calculates exact parameter counts for efficiency reporting."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.embedding_head.parameters())
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params,
        }
