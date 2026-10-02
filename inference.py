"""
inference.py
============
Single-Image Visual Design Retrieval and Top-K Similarity Search.
"""

import os
import json
import argparse
from PIL import Image
import torch
import matplotlib.pyplot as plt

from data.transforms import get_inference_transforms
from models import SareeEmbeddingModel


def search_query(
    query_image_path: str,
    checkpoint_path: str = "./checkpoints/best_model.pt",
    gallery_dir: str = "./gallery_index",
    top_k: int = 5,
    save_plot_path: str = "./reports/retrieval_demo.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device)
    print("=" * 60)
    print("DEEPLURE VISUAL SAREE DESIGN RETRIEVAL DEMO")
    print("=" * 60)
    print(f"Query Image: {query_image_path}")

    # Load gallery index
    gallery_emb_file = os.path.join(gallery_dir, "gallery_embeddings.pt")
    gallery_meta_file = os.path.join(gallery_dir, "gallery_metadata.json")

    if not os.path.exists(gallery_emb_file) or not os.path.exists(gallery_meta_file):
        raise FileNotFoundError(f"Gallery index not found in {gallery_dir}. Run gallery_builder.py first.")

    gallery_embeddings = torch.load(gallery_emb_file, map_location=device)  # [N, 512]
    with open(gallery_meta_file, "r", encoding="utf-8") as f:
        gallery_metadata = json.load(f)

    # Load model
    model = SareeEmbeddingModel(embedding_dim=512, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # Preprocess query image
    transform = get_inference_transforms(image_size=224)
    with Image.open(query_image_path) as raw_img:
        query_img_rgb = raw_img.convert("RGB")
    query_tensor = transform(query_img_rgb).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Extract query embedding
    with torch.no_grad():
        query_emb = model.extract_embedding(query_tensor)  # [1, 512]

    # Compute Cosine Similarities (Dot product on unit sphere)
    # query_emb: [1, 512], gallery_embeddings: [N, 512] -> [N]
    similarities = torch.matmul(gallery_embeddings, query_emb.squeeze(0))  # [N]
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    print(f"\nTop-{top_k} Visually Retrieved Designs:")
    retrieved_results = []
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
        matched_path = gallery_metadata["paths"][idx]
        matched_cat = gallery_metadata["categories"][idx]
        print(f"  Rank #{rank}: Cosine Similarity = {score:.4f} | Category: {matched_cat} | Path: {matched_path}")
        retrieved_results.append({
            "rank": rank,
            "score": score,
            "path": matched_path,
            "category": matched_cat
        })

    # Visualization plot
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))

    # Query Image
    axes[0].imshow(query_img_rgb)
    axes[0].set_title("QUERY IMAGE\n(Visual Pattern Input)", fontsize=11, fontweight="bold", color="darkblue")
    axes[0].axis("off")

    # Retrieved Gallery Images
    for rank, res in enumerate(retrieved_results, start=1):
        ax = axes[rank]
        if os.path.exists(res["path"]):
            with Image.open(res["path"]) as r_img:
                ax.imshow(r_img.convert("RGB"))
        ax.set_title(f"Rank #{rank}\nSim: {res['score']:.4f}\n({res['category']})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nRetrieval visualization saved to: {save_plot_path}")
    return retrieved_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--gallery_dir", type=str, default="./gallery_index")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_plot", type=str, default="./reports/retrieval_demo.png")
    args = parser.parse_args()

    search_query(
        query_image_path=args.query,
        checkpoint_path=args.checkpoint,
        gallery_dir=args.gallery_dir,
        top_k=args.top_k,
        save_plot_path=args.output_plot
    )
