"""
DeepLure Saree AI Agent — Visual Ground Truth Protocol & Benchmark Manager

Core Rule:
- Saree design identity is determined SOLELY by visual patterns (motifs, layout, weave, structure).
- Filenames, folder names, and image paths MUST NEVER define design identity.
- Coarse category folders (Banarasi, Bandhani, Ikat, Pichwai) are auxiliary attributes only.
- Verification pairs for evaluation are strictly verified visual matches (same motif/composition across different color palettes) vs hard negatives.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


class VisualGroundTruthManager:
    """
    Manages verified visual design pairs and benchmark annotations for evaluation.
    """

    def __init__(self, benchmark_file: Optional[str] = None):
        self.benchmark_file = benchmark_file
        self.verified_pairs: List[Dict[str, Any]] = []
        if benchmark_file and Path(benchmark_file).exists():
            self.load_benchmark(benchmark_file)

    def load_benchmark(self, path: str):
        """Loads human-verified visual pairs from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.verified_pairs = data.get("pairs", [])

    def save_benchmark(self, path: str, pairs: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        """Saves verified visual pairs to standard JSON benchmark file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "description": "DeepLure Visual Saree Design Benchmark (Human Verified)",
            "rule": "Identity is established strictly by visual motifs/layout, independent of color palette or filename.",
            "total_pairs": len(pairs),
            "metadata": metadata or {},
            "pairs": pairs
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def create_color_invariance_eval_pair(
        img_path: str,
        transformed_img_path: Optional[str] = None,
        notes: str = "Color-jittered / Grayscale pair for color-invariance evaluation"
    ) -> Dict[str, Any]:
        """
        Creates a color-invariance test pair representing the same exact physical weave design
        with completely different colorway / palette.
        """
        return {
            "image_1": str(img_path),
            "image_2": str(transformed_img_path or img_path),
            "is_same_design": 1,
            "verification_type": "color_invariant_same_design",
            "notes": notes
        }

    @staticmethod
    def create_hard_negative_pair(
        img_path_1: str,
        img_path_2: str,
        category: str,
        notes: str = "Different design motifs within the same broad style category"
    ) -> Dict[str, Any]:
        """
        Creates a hard negative evaluation pair (two distinct saree designs from the same
        broad category like Banarasi, testing fine-grained motif separation).
        """
        return {
            "image_1": str(img_path_1),
            "image_2": str(img_path_2),
            "is_same_design": 0,
            "verification_type": "hard_negative_same_category",
            "category": category,
            "notes": notes
        }
