"""
train.py
========
Self-Supervised Color-Invariant Contrastive Training Engine for DeepLure Saree Design Recognition.

Strict Rules:
- NO filename-based identity grouping.
- NO treating coarse category folders (Banarasi/Bandhani/Ikat/Pichwai) as design IDs.
- Positive pairs: Two distinct augmented views of the same original saree image (color/spatial perturbations).
- Negative pairs: All other saree images in the batch.
"""

import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TwoViewContrastiveDataset
from data.transforms import get_contrastive_train_transforms
from models import SareeEmbeddingModel, ContrastiveInfoNCELoss


def train(
    data_root: str = "./kaggle",
    output_dir: str = "./checkpoints",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    temperature: float = 0.07,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print("=" * 60)
    print("DEEPLURE SAREE AI — COLOR-INVARIANT CONTRASTIVE TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data Root: {data_root}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    # Transforms & Datasets
    transform = get_contrastive_train_transforms(image_size=224)
    train_dataset = TwoViewContrastiveDataset(data_root=data_root, split="train", transform=transform)
    val_dataset = TwoViewContrastiveDataset(data_root=data_root, split="valid", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=(device.type == "cuda")
    )

    # Model & Loss
    model = SareeEmbeddingModel(embedding_dim=512, pretrained=True).to(device)
    criterion = ContrastiveInfoNCELoss(temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        # --- TRAINING ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        start_time = time.time()

        for view_1, view_2, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            optimizer.zero_grad()
            z_1 = model(view_1)
            z_2 = model(view_2)

            loss = criterion(z_1, z_2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, train_batches)

        # --- VALIDATION ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for view_1, view_2, _ in val_loader:
                view_1 = view_1.to(device)
                view_2 = view_2.to(device)

                z_1 = model(view_1)
                z_2 = model(view_2)
                v_loss = criterion(z_1, z_2)

                val_loss_sum += v_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(1, val_batches)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "embedding_dim": 512,
            }, best_model_path)
            print(f"  ★ Best checkpoint saved: {best_model_path} (Val Loss: {best_val_loss:.4f})")

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nTraining complete. History saved to {history_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./kaggle")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
