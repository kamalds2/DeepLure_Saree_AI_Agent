# models/__init__.py
from .backbone import EfficientNetB3Backbone
from .embedding_head import EmbeddingHead, SareeEmbeddingModel
from .arcface import ArcFaceLoss


def build_model(config: dict) -> tuple:
    """
    Factory function to build the complete model from config.

    Returns:
        (model, criterion): SareeEmbeddingModel and ArcFaceLoss ready for training
    """
    from data import NUM_CLASSES

    backbone = EfficientNetB3Backbone(
        pretrained=config["model"]["pretrained"]
    )
    head = EmbeddingHead(
        in_features=EfficientNetB3Backbone.FEATURE_DIM,
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"],
    )
    model     = SareeEmbeddingModel(backbone, head)
    criterion = ArcFaceLoss(
        in_features=config["model"]["embedding_dim"],
        num_classes=NUM_CLASSES,
        s=config["arcface"]["scale"],
        m=config["arcface"]["margin"],
    )

    # Print parameter count (efficiency report)
    param_counts = model.count_parameters()
    arcface_params = sum(p.numel() for p in criterion.parameters())
    print("\n[model] Parameter summary:")
    print(f"  Backbone (EfficientNet-B3): {param_counts['backbone']:,}")
    print(f"  Embedding Head:             {param_counts['head']:,}")
    print(f"  ArcFace Loss:               {arcface_params:,}")
    print(f"  TOTAL:                      {param_counts['total'] + arcface_params:,}")

    return model, criterion


__all__ = [
    "EfficientNetB3Backbone",
    "EmbeddingHead",
    "SareeEmbeddingModel",
    "ArcFaceLoss",
    "build_model",
]
