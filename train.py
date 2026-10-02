"""
train.py
========
3-Phase Training Loop for DeepLure Saree AI.

TRAINING PHASES (from DeepLure_Architecture_Plan.md §4.1):
    Phase 1 — Head Warmup (5 epochs):
        Freeze backbone, train only embedding head + ArcFace.
        WHY? Let the new projection head adapt before disturbing pretrained weights.

    Phase 2 — Full Fine-tuning (25 epochs):
        Unfreeze all layers with discriminative LR (backbone=1e-5, head=1e-4).
        WHY discriminative LR? Backbone already has good texture features from
        ImageNet. We nudge it gently toward saree-specific patterns.

    Phase 3 — Color Invariance (5 epochs):
        Re-freeze backbone, increase RandomGrayscale to 70%.
        WHY? Final calibration for color invariance without destroying texture features.

TOTAL: 35 epochs. Checkpoint saved at best validation accuracy (phase 2).

USAGE:
    python train.py --config configs/config.yaml --data_root ./kaggle

    # With custom output dir:
    python train.py --config configs/config.yaml \
                    --data_root ./kaggle \
                    --checkpoint_dir ./checkpoints \
                    --device cuda
"""

import os
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Local imports
from data import (
    KaggleSareeDataset,
    get_train_transforms,
    get_val_transforms,
    get_phase3_transforms,
    NUM_CLASSES,
)
from models import build_model
from utils.logger import TrainingLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy from raw logits."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current LR from first param group (for logging)."""
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(
    model, criterion, optimizer, epoch, phase, val_acc, path
):
    """Save a checkpoint with all training state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "phase":      phase,
        "val_acc":    val_acc,
        "model_state":     model.state_dict(),
        "criterion_state": criterion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Checkpoint saved: {path} (val_acc={val_acc:.4f})")


# ---------------------------------------------------------------------------
# One-epoch training / validation functions
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler=None,     # torch.cuda.amp.GradScaler for mixed precision
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, avg_accuracy): epoch-level metrics
    """
    model.train()
    criterion.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision (AMP) — halves memory use on GPU
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                loss = criterion(embeddings, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss       = criterion(embeddings, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    grad_clip
                )
            optimizer.step()

        # Compute accuracy using cosine similarity (no margin at inference)
        with torch.no_grad():
            cosine_logits = criterion.get_cosine_similarities(embeddings)
        acc = compute_accuracy(cosine_logits, labels)

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += int(acc * bs)
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one validation epoch.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    criterion.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        # Use raw cosine similarity for accuracy (no margin at eval)
        cosine_logits = criterion.get_cosine_similarities(embeddings)
        acc = compute_accuracy(cosine_logits, labels)

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += int(acc * bs)
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase(
    phase: int,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_path: str,
    best_val_acc: float,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    """
    Run a full training phase (N epochs). Returns updated best_val_acc.
    """
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    print(f"\n{'='*60}")
    print(f" PHASE {phase} — {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, criterion, train_loader, optimizer, device, grad_clip, scaler
        )
        logger.log(phase, epoch, "train", train_loss, train_acc, get_lr(optimizer))

        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        logger.log(phase, epoch, "val", val_loss, val_acc, get_lr(optimizer))

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save best checkpoint (phase 2 training = main fine-tuning)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, criterion, optimizer,
                epoch, phase, val_acc, checkpoint_path
            )

    return best_val_acc


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Device: {device}")

    # Create output dirs
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    checkpoint_best = os.path.join(
        cfg["paths"]["checkpoint_dir"], "best_model.pth"
    )

    # Logger
    logger = TrainingLogger(log_dir=cfg["paths"]["logs_dir"])

    # ------------------------------------------------------------------
    # Phase 1 + 2 DataLoaders (standard transforms)
    # ------------------------------------------------------------------
    aug_cfg = cfg["augmentation"]
    train_transform = get_train_transforms(
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
        grayscale_prob=aug_cfg["random_grayscale_prob"],
        color_jitter_brightness=aug_cfg["color_jitter"]["brightness"],
        color_jitter_contrast=aug_cfg["color_jitter"]["contrast"],
        color_jitter_saturation=aug_cfg["color_jitter"]["saturation"],
        color_jitter_hue=aug_cfg["color_jitter"]["hue"],
        rotation_degrees=aug_cfg["random_rotation_degrees"],
    )
    val_transform = get_val_transforms(image_size=cfg["data"]["image_size"])

    data_root = args.data_root or cfg["data"]["kaggle_root"]

    train_dataset = KaggleSareeDataset(data_root, split="train", transform=train_transform)
    val_dataset   = KaggleSareeDataset(data_root, split="valid", transform=val_transform)

    # Balanced sampler (handle Banarasi class imbalance)
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(
        train_dataset, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model, criterion = build_model(cfg)
    model     = model.to(device)
    criterion = criterion.to(device)

    best_val_acc = 0.0

    # ==================================================================
    # PHASE 1: Head Warmup — backbone frozen
    # ==================================================================
    p1_cfg = cfg["training"]["phase1"]
    model.backbone.freeze()

    optimizer_p1 = torch.optim.AdamW([
        {"params": model.embedding_head.parameters(), "lr": p1_cfg["lr_head"]},
        {"params": criterion.parameters(),            "lr": p1_cfg["lr_head"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=p1_cfg["epochs"]
    )

    best_val_acc = run_phase(
        phase=1,
        model=model, criterion=criterion,
        optimizer=optimizer_p1, scheduler=scheduler_p1,
        train_loader=train_loader, val_loader=val_loader,
        epochs=p1_cfg["epochs"],
        device=device, logger=logger,
        checkpoint_path=checkpoint_best,
        best_val_acc=best_val_acc,
        grad_clip=cfg["training"]["gradient_clip"],
    )

    # ==================================================================
    # PHASE 2: Full Fine-tuning — discriminative LR
    # ==================================================================
    p2_cfg = cfg["training"]["phase2"]
    model.backbone.unfreeze()
    model.backbone.freeze_bn()  # Keep BN frozen (small dataset safety)

    optimizer_p2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(),       "lr": p2_cfg["lr_backbone"]},
        {"params": model.embedding_head.parameters(), "lr": p2_cfg["lr_head"]},
        {"params": criterion.parameters(),            "lr": p2_cfg["lr_arcface"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=p2_cfg["epochs"]
    )

    best_val_acc = run_phase(
        phase=2,
        model=model, criterion=criterion,
        optimizer=optimizer_p2, scheduler=scheduler_p2,
        train_loader=train_loader, val_loader=val_loader,
        epochs=p2_cfg["epochs"],
        device=device, logger=logger,
        checkpoint_path=checkpoint_best,
        best_val_acc=best_val_acc,
        grad_clip=cfg["training"]["gradient_clip"],
    )

    # ==================================================================
    # PHASE 3: Color Invariance Fine-tuning — increased grayscale prob
    # ==================================================================
    p3_cfg = cfg["training"]["phase3"]
    model.backbone.freeze()

    # Rebuild train_loader with phase-3 transforms (higher grayscale prob)
    phase3_transform = get_phase3_transforms(
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )
    train_dataset_p3 = KaggleSareeDataset(data_root, split="train", transform=phase3_transform)
    sample_weights_p3 = train_dataset_p3.get_class_weights()
    sampler_p3 = WeightedRandomSampler(
        weights=sample_weights_p3, num_samples=len(sample_weights_p3), replacement=True
    )
    train_loader_p3 = DataLoader(
        train_dataset_p3, batch_size=bs, sampler=sampler_p3,
        num_workers=nw, pin_memory=True, drop_last=True,
    )

    optimizer_p3 = torch.optim.AdamW([
        {"params": model.embedding_head.parameters(), "lr": p3_cfg["lr_head"]},
        {"params": criterion.parameters(),            "lr": p3_cfg["lr_head"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p3, T_max=p3_cfg["epochs"]
    )

    best_val_acc = run_phase(
        phase=3,
        model=model, criterion=criterion,
        optimizer=optimizer_p3, scheduler=scheduler_p3,
        train_loader=train_loader_p3, val_loader=val_loader,
        epochs=p3_cfg["epochs"],
        device=device, logger=logger,
        checkpoint_path=checkpoint_best,
        best_val_acc=best_val_acc,
        grad_clip=cfg["training"]["gradient_clip"],
    )

    # Done
    logger.summary()
    print(f"\n[train.py] ✅ Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"  Checkpoint: {checkpoint_best}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLure Saree AI — Training")
    parser.add_argument("--config",         type=str, default="configs/config.yaml")
    parser.add_argument("--data_root",      type=str, default=None,
                        help="Override kaggle data root from config")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint dir from config")
    parser.add_argument("--device",         type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args)
