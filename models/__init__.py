"""
DeepLure Saree AI Agent — Models Package (Plan v2.0)
"""

import torch
import torch.nn as nn
from .backbone import EfficientNetB3Backbone
from .embedding_head import EmbeddingHead, SareeEmbeddingModel
from .arcface import ArcFaceLoss
from .metric_losses import BatchHardTripletLoss, SupConLoss

# Aliases
SareeBackbone = EfficientNetB3Backbone
SareeMetricModel = SareeEmbeddingModel


def build_model(
    backbone_name: str = "efficientnet_b3",
    pretrained: bool = True,
    embedding_dim: int = 512,
    dropout: float = 0.3,
    num_classes: int = 4,
    arcface_margin: float = 0.5,
    arcface_scale: float = 64.0,
):
    """
    Factory function to construct backbone, embedding head, and ArcFace loss.
    """
    backbone = EfficientNetB3Backbone(pretrained=pretrained)
    in_features = getattr(backbone, "FEATURE_DIM", 1536)
    head = EmbeddingHead(in_features=in_features, embedding_dim=embedding_dim, dropout=dropout)
    model = SareeEmbeddingModel(backbone=backbone, embedding_head=head)
    criterion = ArcFaceLoss(
        in_features=embedding_dim,
        out_features=num_classes,
        s=arcface_scale,
        m=arcface_margin
    )
    return model, criterion


__all__ = [
    "EfficientNetB3Backbone",
    "SareeBackbone",
    "EmbeddingHead",
    "SareeEmbeddingModel",
    "SareeMetricModel",
    "ArcFaceLoss",
    "BatchHardTripletLoss",
    "SupConLoss",
    "build_model",
]
