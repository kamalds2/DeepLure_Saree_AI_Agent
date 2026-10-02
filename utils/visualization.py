"""
utils/visualization.py
======================
Visualization tools for DeepLure Saree AI.

PLOTS PRODUCED:
    1. t-SNE of gallery embeddings  → shows design clusters on 2D map
    2. Similarity heatmap (Q × G)   → shows query-gallery cosine scores
    3. Top-K retrieval grid         → visual inspection of matches
    4. Training loss/accuracy curve → from training_log.csv

WHY THESE VISUALIZATIONS?
    t-SNE:
        Reduces 512-d embeddings to 2D. If training worked correctly, same-class
        images should cluster together. Color-invariance shows when images of the
        SAME design in DIFFERENT colors land in the same cluster.
        This is the primary visual proof that our approach works.

    Similarity heatmap:
        Shows the raw [Q×G] cosine similarity matrix. Good retrieval = strong
        diagonal blocks where same-class queries match same-class gallery.

    Top-K grid:
        Visual sanity check: "does the top-1 match actually look similar?"
        Essential for qualitative evaluation in the deliverable.

    Training curves:
        Shows loss decreasing across 3 phases. Phase 3 bump shows effect
        of increased grayscale augmentation.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works in Kaggle + headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. t-SNE Embedding Visualization
# ---------------------------------------------------------------------------

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "t-SNE of Saree Embeddings",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
):
    """
    Run t-SNE on embeddings and plot 2D scatter colored by class.

    WHY t-SNE over PCA?
        PCA is linear — it preserves global structure but misses local clusters.
        t-SNE is non-linear and explicitly clusters similar points.
        For visualizing "do same-design images cluster together?", t-SNE is
        the standard tool used in metric learning papers.

    WHY perplexity=30?
        Perplexity ≈ number of effective nearest neighbors.
        30 is the standard default. For our dataset size (~1468 train images),
        20-50 works well. Too low = noisy, too high = all merged.

    Args:
        embeddings:  [N, 512] L2-normalized embeddings
        labels:      [N] integer class labels
        class_names: List of class name strings
        save_path:   Where to save the PNG
        title:       Plot title
        perplexity:  t-SNE perplexity parameter
        n_iter:      t-SNE iterations
        seed:        Random seed for reproducibility
    """
    from sklearn.manifold import TSNE

    print(f"[visualization] Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        metric="cosine",    # Use cosine distance (matches our embedding space)
        init="pca",         # PCA init is more stable than random
    )
    coords_2d = tsne.fit_transform(embeddings)   # [N, 2]

    # Color palette (one color per class)
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels[labels >= 0]))

    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=colors[lbl % len(colors)],
            marker=markers[lbl % len(markers)],
            label=name,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="Style Class", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add annotation explaining color invariance
    ax.text(
        0.02, 0.98,
        "Well-separated clusters → good color-invariant texture discrimination",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", color="gray",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualization] t-SNE saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Similarity Heatmap
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    max_per_class: int = 10,
):
    """
    Plot a cosine similarity heatmap between queries and gallery (subsampled).

    Args:
        query_emb, gallery_emb: [Q, 512] and [G, 512] embeddings
        query_labels, gallery_labels: [Q] and [G] integer labels
        class_names: Class name list
        save_path: Output PNG path
        max_per_class: Max images per class to show (keeps heatmap readable)
    """
    # Subsample: take up to max_per_class per class for both query and gallery
    def subsample(emb, labels, max_n):
        idxs = []
        for cls in sorted(set(labels[labels >= 0])):
            cls_idx = np.where(labels == cls)[0]
            chosen  = cls_idx[:max_n]
            idxs.extend(chosen.tolist())
        return emb[idxs], labels[idxs], idxs

    q_emb_s, q_lbl_s, _  = subsample(query_emb,   query_labels,   max_per_class)
    g_emb_s, g_lbl_s, _  = subsample(gallery_emb, gallery_labels, max_per_class)

    sim_matrix = q_emb_s @ g_emb_s.T   # [Q', G']

    # Sort both axes by label so same-class items are adjacent
    q_order = np.argsort(q_lbl_s)
    g_order = np.argsort(g_lbl_s)
    sim_sorted = sim_matrix[q_order][:, g_order]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sim_sorted, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xlabel("Gallery Images (sorted by class)")
    ax.set_ylabel("Query Images (sorted by class)")
    ax.set_title("Query × Gallery Cosine Similarity Heatmap\n"
                 "(bright diagonal blocks = correct retrieval)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualization] Heatmap saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Top-K Retrieval Grid
# ---------------------------------------------------------------------------

def plot_topk_grid(
    query_path: str,
    results: List[dict],
    save_path: str,
    title: str = "Top-K Retrieval Results",
):
    """
    Show query image alongside its top-K gallery matches in a grid.

    Args:
        query_path: Path to query image
        results: List of dicts from inference.search_gallery()
                 Each: {"rank", "path", "score", "label"}
        save_path: Output PNG path
        title: Plot title
    """
    top_k = len(results)
    fig   = plt.figure(figsize=(3 * (top_k + 1), 4))
    gs    = gridspec.GridSpec(1, top_k + 1, figure=fig)

    # Query image
    ax_q  = fig.add_subplot(gs[0])
    img_q = Image.open(query_path).convert("RGB")
    ax_q.imshow(img_q)
    ax_q.set_title("QUERY", fontsize=10, fontweight="bold", color="#E74C3C")
    ax_q.axis("off")

    # Gallery matches
    for r in results:
        ax   = fig.add_subplot(gs[r["rank"]])
        try:
            img  = Image.open(r["path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
        color = "#2ECC71" if r["score"] > 0.7 else "#F39C12" if r["score"] > 0.5 else "#E74C3C"
        ax.set_title(
            f"#{r['rank']} | {r['score']:.3f}",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[visualization] Top-K grid saved: {save_path}")


# ---------------------------------------------------------------------------
# 4. Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    log_csv_path: str,
    save_path: str,
):
    """
    Plot training and validation loss + accuracy curves from training_log.csv.

    Args:
        log_csv_path: Path to CSV file produced by utils/logger.py
        save_path: Output PNG path
    """
    # Read CSV
    train_loss, val_loss, train_acc, val_acc, epochs_all = [], [], [], [], []
    phase_boundaries = []
    prev_phase = None

    with open(log_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        epoch_counter = 0
        for row in reader:
            phase = int(row["phase"])
            if prev_phase is not None and phase != prev_phase:
                phase_boundaries.append(epoch_counter)
            prev_phase = phase
            if row["split"] == "train":
                train_loss.append(float(row["loss"]))
                train_acc.append(float(row["accuracy"]))
                epoch_counter += 1
            elif row["split"] == "val":
                val_loss.append(float(row["loss"]))
                val_acc.append(float(row["accuracy"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(train_loss, label="Train Loss",      color="#E74C3C", linewidth=2)
    ax1.plot(val_loss,   label="Val Loss",        color="#3498DB", linewidth=2, linestyle="--")
    for pb in phase_boundaries:
        ax1.axvline(pb, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Epoch (all 3 phases)")
    ax1.set_ylabel("ArcFace Loss")
    ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(train_acc, label="Train Acc",        color="#E74C3C", linewidth=2)
    ax2.plot(val_acc,   label="Val Acc",          color="#3498DB", linewidth=2, linestyle="--")
    for pb in phase_boundaries:
        ax2.axvline(pb, color="gray", linestyle=":", alpha=0.5, label="Phase boundary" if pb == phase_boundaries[0] else "")
    ax2.set_xlabel("Epoch (all 3 phases)")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("Training Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate phases
    if phase_boundaries:
        mid_p1 = phase_boundaries[0] / 2
        mid_p2 = (phase_boundaries[0] + (phase_boundaries[1] if len(phase_boundaries) > 1 else len(train_loss))) / 2
        ax1.text(mid_p1, ax1.get_ylim()[1] * 0.95, "Phase 1\n(warmup)", ha="center", fontsize=8, color="gray")
        ax1.text(mid_p2, ax1.get_ylim()[1] * 0.95, "Phase 2\n(fine-tune)", ha="center", fontsize=8, color="gray")

    plt.suptitle("DeepLure Training Curves (3-Phase)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualization] Training curves saved: {save_path}")
