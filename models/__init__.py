"""
DeepLure Saree AI Agent — Models Package
"""

from .embedding_head import SareeEmbeddingModel
from .metric_losses import ContrastiveInfoNCELoss


def build_model(
    embedding_dim: int = 512,
    pretrained: bool = True,
    temperature: float = 0.07,
):
    """Factory helper to instantiate visual embedding model and contrastive loss."""
    model = SareeEmbeddingModel(embedding_dim=embedding_dim, pretrained=pretrained)
    criterion = ContrastiveInfoNCELoss(temperature=temperature)
    return model, criterion


__all__ = [
    "SareeEmbeddingModel",
    "ContrastiveInfoNCELoss",
    "build_model",
]
