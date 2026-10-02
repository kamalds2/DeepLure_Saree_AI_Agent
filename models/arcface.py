"""
models/arcface.py
=================
ArcFace (Additive Angular Margin) Loss for DeepLure Saree AI.

WHY ARCFACE?
    See DeepLure_Architecture_Plan.md §2.3 for full justification.

    Core idea: Face recognition on a unit hypersphere.
    ArcFace adds an angular margin m between classes in the cosine similarity
    space. This makes same-class embeddings (same design) more compact and
    different-class embeddings (different designs) more separated.

    Our case: We treat the 4 Kaggle style categories as "identities".
    Even though Banarasi is broad (many different specific Banarasi designs),
    training with ArcFace teaches the model that Banarasi texture should be
    near Banarasi texture and far from Ikat texture — a useful proxy for
    the fine-grained design identity we actually care about.

MATHEMATICAL FORMULATION (from Deng et al., 2019):
    L = -log( exp(s · cos(θ_yi + m)) /
             (exp(s · cos(θ_yi + m)) + Σ_{j≠yi} exp(s · cos(θ_j))) )

    where:
        θ_yi = angle between embedding and the weight vector of its class
        m    = angular margin (0.5 radians by default)
        s    = scale factor (64 by default)

    The margin is applied DIRECTLY in the angular space (not cosine space),
    which is geometrically more uniform than CosFace (cosine space margin).

HYPERPARAMETERS:
    m = 0.5: Corresponds to ~28.6°. Standard ArcFace paper value.
             Increases this → more compact clusters, harder to train.
             Decrease this → more overlap between classes.

    s = 64: Re-scales logits after L2 normalization.
            Without s, all logits are in [-1, 1] and softmax gradient ≈ 0.
            s=64 places logits in [-64, 64] where softmax has meaningful gradient.
            Standard value from ArcFace paper.

IMPLEMENTATION NOTE:
    The weight matrix W has shape [embedding_dim, num_classes].
    Each column is the class "center" on the unit sphere.
    Both embeddings and class centers are L2-normalized, so dot products
    give cosine similarities = cos(θ).
    ArcFace then applies the margin by adding m to the target class angle.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019).

    Implements ArcFace from scratch (no third-party metric learning libraries).
    The weight matrix serves as class-center prototypes on the unit sphere.

    Args:
        in_features: Embedding dimension (512)
        num_classes: Number of training classes (4 for Kaggle dataset)
        s: Scale factor (64.0) — amplifies logits for meaningful gradients
        m: Angular margin in radians (0.5) — adds class separation
        easy_margin: If True, uses max(0, cos(θ+m)) which is more stable
                     for early training. False = stricter, standard ArcFace.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 4,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.num_classes  = num_classes
        self.s            = s
        self.m            = m
        self.easy_margin  = easy_margin

        # Class center weight matrix [in_features, num_classes]
        # Each column is a prototype direction on the unit sphere.
        # WHY nn.Parameter? So optimizer updates these class centers during training.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants (these don't change during training)
        self.cos_m = math.cos(m)   # cos(m)
        self.sin_m = math.sin(m)   # sin(m)
        # Threshold: cos(π - m). Used to avoid angle θ+m > π (wrapping around sphere)
        self.th    = math.cos(math.pi - m)
        # Fallback: cos(θ) - sin(m)*m (linear approximation near threshold)
        self.mm    = math.sin(math.pi - m) * m

        print(f"[arcface.py] ArcFaceLoss: {in_features}-d embeddings, "
              f"{num_classes} classes, m={m}, s={s}")

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: L2-normalized embeddings [B, 512] (MUST be unit vectors)
            labels: Ground-truth class indices [B], dtype=torch.long

        Returns:
            loss: Scalar cross-entropy loss with angular margin
        """
        # Step 1: Normalize class weight vectors (make them unit vectors too)
        # WHY? Both embeddings and class centers must be on the unit sphere
        # for the dot product to equal cosine similarity.
        W = F.normalize(self.weight, p=2, dim=1)   # [num_classes, 512]

        # Step 2: Compute cosine similarities between embeddings and all class centers
        # cos(θ) = embedding · W^T  [B, num_classes]
        cosine = F.linear(embeddings, W)            # [B, num_classes]
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7) # Numerical stability for acos

        # Step 3: Compute sin(θ) from cos(θ)
        # sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # Step 4: Add angular margin to target class only
        # cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m  # [B, num_classes]

        # Step 5: Handle edge case: if θ + m > π, clamp to cos(θ) - m*sin(m)
        if self.easy_margin:
            # Easy margin: only apply margin when cos(θ) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: apply fallback when cos(θ) < threshold
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Step 6: Build one-hot mask for target classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Step 7: Replace target class cosine with margin-adjusted phi
        # For non-target classes: keep original cosine (no margin applied)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Step 8: Scale by s and compute cross-entropy
        output = output * self.s  # [B, num_classes]

        loss = F.cross_entropy(output, labels)
        return loss

    def get_cosine_similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get raw cosine similarities (WITHOUT margin) for inference/evaluation.

        Args:
            embeddings: L2-normalized embeddings [B, 512]

        Returns:
            cosine: [B, num_classes] cosine similarity scores
        """
        W = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings, W)
