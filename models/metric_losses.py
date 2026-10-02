"""
DeepLure Saree AI Agent — Metric Learning Losses (Plan v2.0)

Includes:
1. BatchHardTripletLoss — Triplet loss with online hard positive and hard negative mining.
2. SupConLoss — Supervised Contrastive Learning loss (Khosla et al., NeurIPS 2020).
3. ArcFaceLoss — Additive Angular Margin Loss (Deng et al., CVPR 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss:
    For each anchor in the batch, selects the hardest positive (furthest distance)
    and the hardest negative (closest distance).
    Loss = max(0, D(a, p_hard) - D(a, n_hard) + margin)
    """

    def __init__(self, margin: float = 0.3, distance_metric: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized feature embeddings.
            labels: (B,) class or design IDs.
        """
        # Pairwise distance matrix
        if self.distance_metric == "cosine":
            # Since embeddings are L2 normalized, cosine distance = 1 - cosine_similarity
            similarity = torch.matmul(embeddings, embeddings.T)  # (B, B) in [-1, 1]
            dist_mat = 1.0 - similarity
        else:
            # Euclidean distance squared
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Boolean masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

        pos_mask = labels_eq & (~eye_mask)  # Same class, different sample
        neg_mask = ~labels_eq               # Different class

        # Hardest positive: maximum distance among positives
        dist_pos = dist_mat.clone()
        dist_pos[~pos_mask] = -1e9
        hardest_pos_dist, _ = torch.max(dist_pos, dim=1)

        # Hardest negative: minimum distance among negatives
        dist_neg = dist_mat.clone()
        dist_neg[~neg_mask] = 1e9
        hardest_neg_dist, _ = torch.min(dist_neg, dim=1)

        # Valid anchors (must have at least one positive and one negative)
        valid_anchors = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        if not valid_anchors.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        triplet_loss = F.relu(hardest_pos_dist[valid_anchors] - hardest_neg_dist[valid_anchors] + self.margin)
        return triplet_loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon):
    Khosla et al., NeurIPS 2020. Pulls all samples of the same class together
    while pushing different classes apart on the hypersphere.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) class or design IDs.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Cosine similarity matrix scaled by temperature
        sim = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)

        # For numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-7)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        loss = -mean_log_prob_pos
        return loss.mean()
