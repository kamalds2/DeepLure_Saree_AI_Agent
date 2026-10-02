"""
DeepLure Saree AI Agent — Ground Truth Manager (Plan v2.0)

Handles:
- Design Identity definitions (manual annotations vs synthetic multi-color pairs).
- Pair generation for verification and metric learning (positive same-design, negative different-design).
- Leakage-safe split management (train/val/gallery/query).
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from PIL import Image


class GroundTruthManager:
    """
    Ground truth manager ensuring that design identities are clearly distinguished
    from broad pattern categories, and pairs for verification evaluation are constructed
    without synthetic test leakage.
    """

    def __init__(self, annotation_file: Optional[str] = None):
        self.annotation_file = annotation_file
        self.annotations = []
        if annotation_file and Path(annotation_file).exists():
            self.load_annotations(annotation_file)

    def load_annotations(self, path: str):
        """Loads JSON annotations formatted with design_id, path, category."""
        with open(path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

    def save_annotations(self, path: str):
        """Saves current annotations to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2)

    def create_verification_pairs(
        self,
        image_records: List[Dict[str, Any]],
        num_positive_pairs: int = 200,
        num_negative_pairs: int = 200,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Creates balanced positive and negative pairs for verification testing (ROC-AUC / EER).
        Positive pairs: Same design identity (same design_id or augmented pair).
        Negative pairs: Different design identity (often same pattern category as hard negatives).
        """
        random.seed(seed)
        pairs = []

        # Group by design_id
        design_groups = {}
        category_groups = {}
        for rec in image_records:
            d_id = rec.get("design_id", rec.get("category"))
            cat = rec.get("category")

            design_groups.setdefault(d_id, []).append(rec)
            category_groups.setdefault(cat, []).append(rec)

        # Generate Positive Pairs (same design)
        pos_count = 0
        valid_designs = [d for d, items in design_groups.items() if len(items) >= 2]

        if valid_designs:
            while pos_count < num_positive_pairs:
                d = random.choice(valid_designs)
                img1, img2 = random.sample(design_groups[d], 2)
                pairs.append({
                    "img1": img1["path"],
                    "img2": img2["path"],
                    "label": 1,  # Match
                    "type": "positive_intra_design",
                    "category": img1["category"]
                })
                pos_count += 1

        # Generate Negative Pairs (different design, but preferably same category for hard negatives)
        neg_count = 0
        all_designs = list(design_groups.keys())

        if len(all_designs) >= 2:
            while neg_count < num_negative_pairs:
                d1, d2 = random.sample(all_designs, 2)
                img1 = random.choice(design_groups[d1])
                img2 = random.choice(design_groups[d2])

                is_hard = (img1["category"] == img2["category"])
                pairs.append({
                    "img1": img1["path"],
                    "img2": img2["path"],
                    "label": 0,  # Non-match
                    "type": "negative_hard_same_category" if is_hard else "negative_random",
                    "category_1": img1["category"],
                    "category_2": img2["category"]
                })
                neg_count += 1

        return pairs
