"""
utils/logger.py
===============
Training logger for DeepLure Saree AI — tracks loss, accuracy, LR per epoch.

WHY A CUSTOM LOGGER?
    We need to log training/val metrics per epoch across 3 phases (35 epochs total).
    A simple CSV logger lets us plot training curves afterward and check
    that training is progressing (loss decreasing, accuracy increasing).
    No external dependency on tensorboard or wandb — keeps the Kaggle notebook clean.
"""

import os
import csv
import time
from typing import Dict, Any


class TrainingLogger:
    """
    Lightweight CSV-based training logger.

    Writes one row per epoch with: phase, epoch, loss, accuracy, lr, elapsed_time.

    Args:
        log_dir: Directory where CSV log file will be saved
        filename: CSV filename (default: training_log.csv)
    """

    def __init__(self, log_dir: str = "./logs", filename: str = "training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.start_time = time.time()
        self.epoch_data = []

        # Write header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "phase", "epoch", "split",
                "loss", "accuracy", "lr", "elapsed_sec"
            ])
        print(f"[logger.py] Training log: {self.log_path}")

    def log(
        self,
        phase: int,
        epoch: int,
        split: str,
        loss: float,
        accuracy: float,
        lr: float,
    ):
        """
        Log one epoch's metrics.

        Args:
            phase: Training phase (1, 2, or 3)
            epoch: Epoch number (within the phase)
            split: "train" or "val"
            loss: Average loss for this epoch
            accuracy: Top-1 accuracy for this epoch
            lr: Current learning rate (backbone lr, or head lr if frozen)
        """
        elapsed = time.time() - self.start_time
        row = {
            "phase": phase,
            "epoch": epoch,
            "split": split,
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 4),
            "lr": f"{lr:.2e}",
            "elapsed_sec": round(elapsed, 1),
        }
        self.epoch_data.append(row)

        # Append to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(row.values()))

        # Print to console
        print(
            f"  Phase {phase} | Epoch {epoch:>3} | {split:<5} | "
            f"Loss: {loss:.4f} | Acc: {accuracy:.3f} | LR: {lr:.2e}"
        )

    def summary(self):
        """Print best val accuracy and final training loss."""
        val_rows = [r for r in self.epoch_data if r["split"] == "val"]
        if val_rows:
            best = max(val_rows, key=lambda r: r["accuracy"])
            print(f"\n[logger.py] Best val accuracy: {best['accuracy']:.4f} "
                  f"(Phase {best['phase']}, Epoch {best['epoch']})")
        print(f"[logger.py] Log saved to: {self.log_path}")
