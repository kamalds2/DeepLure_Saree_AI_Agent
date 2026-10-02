"""
utils/metrics.py
================
Evaluation metrics for DeepLure Saree AI.

METRICS IMPLEMENTED:
    1. Recall@K — Identification task (retrieval)
    2. Mean Average Precision (mAP) — Identification task (ranked retrieval)
    3. ROC-AUC — Verification task (pair classification)
    4. Equal Error Rate (EER) — Verification task (biometric standard)
    5. Accuracy at optimal threshold — Verification task

WHY THESE METRICS?
    Identification (retrieval):
        Recall@K: Directly answers "does the correct design appear in the top-K results?"
                  This is what a user cares about in a search scenario.
        mAP: More rigorous — rewards systems that rank correct results near the top.
             It's the standard metric for image retrieval benchmarks (Oxford5k, Paris).

    Verification (pair classification):
        ROC-AUC: Threshold-independent summary. Answers "how well can the model
                 distinguish same vs different design pairs across ALL thresholds?"
        EER: Equal Error Rate — the threshold where FPR = FNR. Industry standard
             for biometric verification. Lower EER = better.
        Accuracy@optimal: Practical metric — best accuracy achievable after tuning θ.

INPUTS:
    For identification:
        - query_embeddings: [Q, 512] unit vectors for query images
        - gallery_embeddings: [G, 512] unit vectors for gallery images
        - query_labels: [Q] class labels for queries
        - gallery_labels: [G] class labels for gallery images

    For verification:
        - scores: [N] cosine similarity scores for pairs
        - is_same: [N] binary labels (1=same design, 0=different)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from typing import List, Tuple


def cosine_similarity_matrix(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between all query-gallery pairs.

    Since both are L2-normalized (unit vectors), cosine_sim = dot product.
    This is done as a matrix multiplication for efficiency:
        sim_matrix = query_emb @ gallery_emb.T

    Args:
        query_emb:   [Q, D] L2-normalized query embeddings
        gallery_emb: [G, D] L2-normalized gallery embeddings

    Returns:
        sim_matrix: [Q, G] cosine similarity scores
    """
    # Ensure unit norm (defensive normalization)
    q_norm = query_emb  / (np.linalg.norm(query_emb,   axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8)
    return q_norm @ g_norm.T   # [Q, G]


def recall_at_k(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    Compute Recall@K for all K values.

    For each query:
        1. Rank gallery by cosine similarity (descending)
        2. Check if ANY of the top-K gallery items has the same label as query
        3. Recall@K = fraction of queries where this is true

    Args:
        query_emb:      [Q, D] query embeddings
        gallery_emb:    [G, D] gallery embeddings
        query_labels:   [Q] query class labels (int)
        gallery_labels: [G] gallery class labels (int)
        k_values:       List of K values to compute recall for

    Returns:
        dict: {"recall@1": 0.xx, "recall@5": 0.xx, "recall@10": 0.xx}
    """
    sim_matrix = cosine_similarity_matrix(query_emb, gallery_emb)   # [Q, G]
    n_queries  = sim_matrix.shape[0]
    results    = {}

    for k in k_values:
        hit = 0
        for q_idx in range(n_queries):
            # Get top-K gallery indices (excluding self if query is in gallery)
            sims = sim_matrix[q_idx]
            top_k_idx = np.argsort(sims)[::-1][:k]
            # Check if any top-K match the query's label
            top_k_labels = gallery_labels[top_k_idx]
            if query_labels[q_idx] in top_k_labels:
                hit += 1
        results[f"recall@{k}"] = hit / n_queries

    return results


def mean_average_precision(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute Mean Average Precision (mAP) for retrieval.

    For each query:
        1. Rank gallery by similarity
        2. Compute Average Precision (area under precision-recall curve for this query)
    mAP = mean of per-query AP scores.

    WHY mAP over Recall@K?
        mAP rewards systems that rank ALL correct items near the top, not just
        the first one. It's more sensitive to ranking quality.

    Args:
        query_emb, gallery_emb, query_labels, gallery_labels: as in recall_at_k

    Returns:
        mAP: float in [0, 1]
    """
    sim_matrix = cosine_similarity_matrix(query_emb, gallery_emb)   # [Q, G]
    n_queries  = sim_matrix.shape[0]
    ap_scores  = []

    for q_idx in range(n_queries):
        sims    = sim_matrix[q_idx]                           # [G]
        ranking = np.argsort(sims)[::-1]                      # desc order
        ranked_labels = gallery_labels[ranking]

        # Binary relevance: 1 if same class as query, 0 otherwise
        relevance = (ranked_labels == query_labels[q_idx]).astype(float)

        if relevance.sum() == 0:
            ap_scores.append(0.0)
            continue

        # Average Precision = area under P-R curve
        # sklearn's average_precision_score expects scores and binary targets
        ap = average_precision_score(relevance, sims[ranking])
        ap_scores.append(ap)

    return float(np.mean(ap_scores))


def roc_auc_eer(
    scores: np.ndarray,
    is_same: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute ROC-AUC, EER, and optimal threshold for verification.

    Args:
        scores:  [N] cosine similarity scores for each pair
        is_same: [N] binary labels (1=same design, 0=different)

    Returns:
        (roc_auc, eer, optimal_threshold): floats
    """
    # ROC-AUC
    roc_auc = roc_auc_score(is_same, scores)

    # ROC curve for EER and threshold
    fpr, tpr, thresholds = roc_curve(is_same, scores)
    fnr = 1.0 - tpr

    # EER: the threshold where FPR ≈ FNR
    # Find index where |FPR - FNR| is minimized
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Optimal threshold: maximizes (TPR - FPR) = Youden's J statistic
    # This gives the threshold that maximizes accuracy
    youden_idx      = np.argmax(tpr - fpr)
    opt_threshold   = float(thresholds[youden_idx])

    return roc_auc, eer, opt_threshold


def verification_accuracy(
    scores: np.ndarray,
    is_same: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute binary accuracy at a given cosine similarity threshold.

    Predicts SAME_DESIGN if score > threshold.

    Args:
        scores:    [N] cosine similarity scores
        is_same:   [N] binary labels (1=same, 0=different)
        threshold: Decision boundary (tuned on validation set)

    Returns:
        accuracy: float in [0, 1]
    """
    predictions = (scores >= threshold).astype(int)
    return float((predictions == is_same).mean())


def print_retrieval_report(
    recall_dict: dict,
    map_score: float,
    prefix: str = "",
):
    """Pretty-print retrieval evaluation results."""
    print(f"\n{prefix}{'─'*40}")
    print(f"{prefix}  IDENTIFICATION (Retrieval) Results")
    print(f"{prefix}{'─'*40}")
    for k, v in sorted(recall_dict.items()):
        print(f"{prefix}  {k:12s}: {v:.4f} ({v*100:.1f}%)")
    print(f"{prefix}  mAP         : {map_score:.4f} ({map_score*100:.1f}%)")
    print(f"{prefix}{'─'*40}")


def print_verification_report(
    roc_auc: float,
    eer: float,
    acc: float,
    threshold: float,
    prefix: str = "",
):
    """Pretty-print verification evaluation results."""
    print(f"\n{prefix}{'─'*40}")
    print(f"{prefix}  VERIFICATION Results")
    print(f"{prefix}{'─'*40}")
    print(f"{prefix}  ROC-AUC     : {roc_auc:.4f}")
    print(f"{prefix}  EER         : {eer:.4f} ({eer*100:.1f}%)")
    print(f"{prefix}  Accuracy    : {acc:.4f} ({acc*100:.1f}%)")
    print(f"{prefix}  Threshold   : {threshold:.4f}")
    print(f"{prefix}{'─'*40}")
