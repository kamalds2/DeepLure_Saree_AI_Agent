"""
gallery_builder.py
==================
Builds normalized 512-D visual embedding index for the Saree Gallery.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import SingleImageDataset
from data.transforms import get_inference_transforms
from models import SareeEmbeddingModel


def build_gallery(
    checkpoint_path: str,
    data_root: str = "./kaggle",
    split: str = "train",
    output_dir: str = "./gallery_index",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    print("=" * 60)
    print("BUILDING VISUAL GALLERY EMBEDDING INDEX")
    print("=" * 60)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = SareeEmbeddingModel(embedding_dim=512, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    transform = get_inference_transforms(image_size=224)
    dataset = SingleImageDataset(data_root=data_root, split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    all_embeddings = []
    all_paths = []
    all_categories = []

    with torch.no_grad():
        for imgs, paths, categories in tqdm(loader, desc="Extracting Gallery Embeddings"):
            imgs = imgs.to(device)
            embeddings = model.extract_embedding(imgs)  # [B, 512] L2 normalized
            all_embeddings.append(embeddings.cpu())
            all_paths.extend(paths)
            all_categories.extend(categories)

    gallery_tensor = torch.cat(all_embeddings, dim=0)  # [N, 512]
    embeddings_file = os.path.join(output_dir, "gallery_embeddings.pt")
    metadata_file = os.path.join(output_dir, "gallery_metadata.json")

    torch.save(gallery_tensor, embeddings_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(all_paths),
            "embedding_dim": 512,
            "paths": all_paths,
            "categories": all_categories,
        }, f, indent=2)

    print(f"\nGallery successfully built!")
    print(f"Total images indexed: {gallery_tensor.shape[0]}")
    print(f"Embedding matrix shape: {gallery_tensor.shape}")
    print(f"Saved: {embeddings_file} and {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--data_root", type=str, default="./kaggle")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./gallery_index")
    args = parser.parse_args()

    build_gallery(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir
    )
