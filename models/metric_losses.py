"""
models/metric_losses.py
=======================
Self-Supervised Contrastive Learning Loss (SimCLR / InfoNCE) for Color-Invariant Design Embeddings.

Mechanism:
- For batch of N images, each image produces 2 augmented views (2N representations).
- Positive pair: (z_i, z_j) where view i and view j originate from the SAME original image.
- Negative pairs: all other 2(N-1) views in the batch (different saree images).
- Objective: Maximize cosine agreement between positive views of the same weave motif
  while pushing apart all distinct saree designs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveInfoNCELoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / SimCLR).
    Chen et al., ICML 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: L2-normalized embeddings for view 1 [B, D]
            z_j: L2-normalized embeddings for view 2 [B, D]
        Returns:
            scalar contrastive loss
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Concatenate both views: [2B, D]
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix: [2B, 2B]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity (diagonal)
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(diag_mask, -1e9)

        # Ground-truth positive target indices:
        # For view 1 (index 0..B-1), positive is at index + B
        # For view 2 (index B..2B-1), positive is at index - B
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])

        loss = F.cross_entropy(sim_matrix, targets)
        return loss
