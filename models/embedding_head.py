"""
models/embedding_head.py
========================
512-d L2-normalized embedding head for DeepLure Saree AI.

WHY THIS MODULE EXISTS:
    EfficientNet-B3 gives us a 1536-d raw feature vector. We project this down
    to a compact 512-d embedding that lives on a unit hypersphere (via L2 norm).

    The unit hypersphere is important because:
    1. ArcFace loss (our training loss) operates on angular distances on the
       unit sphere — L2 normalization is required for this.
    2. At inference, cosine similarity between two unit vectors equals their
       dot product, which is extremely fast for gallery search.
    3. Embedding size (512) controls the retrieval index size: each gallery
       image = 512 × 4 bytes = 2 KB.

ARCHITECTURE:
    1536 (from GAP) → Dropout(0.3) → Linear(512) → BN(512) → L2Normalize

DESIGN DECISIONS:

    Dropout(0.3) before linear:
        WHY? With only 1,468 training images, regularization is critical.
        Dropout prevents the head from memorizing specific images.

    BatchNorm1d(512) after linear, before L2 norm:
        WHY? Stabilizes the pre-normalization distribution.
        Without BN, some embedding dimensions collapse to near-zero, wasting
        capacity. BN ensures all 512 dimensions contribute meaningfully.
        Note: We apply BN BEFORE L2 normalization so the norm operation
        operates on a well-scaled vector.

    L2 Normalization (unit sphere projection):
        WHY? ArcFace requires features on the unit sphere.
        Also makes cosine similarity = dot product, which is numerically stable
        and cache-friendly for FAISS or torch.cdist searches.

    512 embedding dimensions (not 128, not 1024):
        WHY 512? Standard from face recognition literature (ArcFace, FaceNet).
        128-d loses too much information for fine-grained texture patterns.
        1024-d is unnecessary overkill for a 4-class domain and hurts
        gallery search speed linearly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """
    Projects [B, 1536] backbone features → [B, 512] L2-normalized embeddings.

    Args:
        in_features: Input dimension from backbone (1536 for EfficientNet-B3)
        embedding_dim: Output embedding dimension (512)
        dropout: Dropout probability before the projection (0.3)
    """

    def __init__(
        self,
        in_features: int = 1536,
        embedding_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features   = in_features
        self.embedding_dim = embedding_dim

        # Projection: 1536 → 512
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim, bias=False),  # bias=False because BN follows
            nn.BatchNorm1d(embedding_dim),
            # No activation here — we normalize next
            # WHY NO ReLU? ReLU would zero-out negative dimensions, collapsing
            # the embedding onto a positive orthant and wasting half the sphere.
        )

        # Initialize weights
        self._init_weights()
        print(f"[embedding_head.py] EmbeddingHead: {in_features} → {embedding_dim}-d (L2 norm)")

    def _init_weights(self):
        """
        Kaiming uniform init for the linear layer.
        WHY Kaiming? It accounts for the fan-in of the previous layer,
        preventing vanishing/exploding gradients at initialization.
        Standard best practice for layers followed by BN.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone feature vector [B, 1536]

        Returns:
            embedding: L2-normalized vector [B, 512] on unit hypersphere
                       Each vector has ||embedding||₂ = 1.0
        """
        x = self.projection(x)           # [B, 512] — projected + BN
        x = F.normalize(x, p=2, dim=1)  # [B, 512] — unit sphere projection
        return x

    def count_parameters(self) -> int:
        """Return parameter count for efficiency report."""
        return sum(p.numel() for p in self.parameters())


class SareeEmbeddingModel(nn.Module):
    """
    Complete embedding model: Backbone + EmbeddingHead.

    This is the main model class used by:
        - train.py (training)
        - evaluate.py (evaluation)
        - inference.py (inference / gallery building)

    Args:
        backbone: EfficientNetB3Backbone instance
        embedding_head: EmbeddingHead instance
    """

    def __init__(self, backbone, embedding_head: EmbeddingHead):
        super().__init__()
        self.backbone       = backbone
        self.embedding_head = embedding_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass: image → L2-normalized embedding.

        Args:
            x: Input images [B, 3, 224, 224]

        Returns:
            embeddings: [B, 512] unit-normalized feature vectors
        """
        features   = self.backbone(x)        # [B, 1536]
        embeddings = self.embedding_head(features)  # [B, 512]
        return embeddings

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-mode embedding extraction (no gradient tracking).

        Use this during gallery building and query evaluation to save memory.

        Args:
            x: Input images [B, 3, 224, 224]

        Returns:
            embeddings: [B, 512] unit-normalized feature vectors
        """
        self.eval()
        return self.forward(x)

    def count_parameters(self) -> dict:
        """
        Return parameter counts for efficiency report (deliverable 4).

        Returns:
            dict with 'backbone', 'head', 'total' parameter counts
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params     = sum(p.numel() for p in self.embedding_head.parameters())
        return {
            "backbone":  backbone_params,
            "head":      head_params,
            "total":     backbone_params + head_params,
        }
