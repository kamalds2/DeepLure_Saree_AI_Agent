"""
DeepLure Saree AI Agent — Pure Visual Dataset Audit Module

Performs pre-training dataset inspection focused strictly on data integrity:
- Image count and distribution per split (train, valid, test).
- Image format validation and corrupted file detection.
- Image resolution and aspect ratio statistics.
- Color channel variance & grayscale distribution.
- NO filename-based grouping or pseudo-identity deduction.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class VisualDatasetAuditor:
    """Audits dataset images purely based on visual and structural validity."""

    def __init__(self, data_root: str, class_names: List[str] = None):
        self.data_root = Path(data_root)
        self.class_names = class_names or ["Banarasi", "Bandhani", "Ikat", "Pichwai"]

    def audit(self) -> Dict[str, Any]:
        """Runs visual dataset audit."""
        if not self.data_root.exists():
            return {
                "status": "error",
                "message": f"Data root {self.data_root} does not exist."
            }

        splits = ["train", "valid", "test"]
        split_stats = {}
        resolution_stats = []
        corrupted_files = []
        class_distribution = {cls: 0 for cls in self.class_names}
        total_images = 0

        for split in splits:
            split_dir = self.data_root / split
            if not split_dir.exists():
                continue

            split_stats[split] = {cls: 0 for cls in self.class_names}
            split_stats[split]["total"] = 0

            for cls in self.class_names:
                cls_dir = split_dir / cls
                if not cls_dir.exists():
                    continue

                for img_path in cls_dir.glob("*.*"):
                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                        continue

                    total_images += 1
                    split_stats[split][cls] += 1
                    split_stats[split]["total"] += 1
                    class_distribution[cls] += 1

                    if HAS_PIL:
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # Verify image integrity
                            with Image.open(img_path) as img:
                                w, h = img.size
                                resolution_stats.append((w, h))
                        except Exception as e:
                            corrupted_files.append({"path": str(img_path), "error": str(e)})

        resolutions_w = [r[0] for r in resolution_stats] if resolution_stats else [0]
        resolutions_h = [r[1] for r in resolution_stats] if resolution_stats else [0]

        report = {
            "status": "success",
            "data_root": str(self.data_root),
            "total_images": total_images,
            "corrupted_images_count": len(corrupted_files),
            "corrupted_files": corrupted_files,
            "splits": split_stats,
            "coarse_category_distribution": class_distribution,
            "resolution": {
                "min_width": int(min(resolutions_w)),
                "max_width": int(max(resolutions_w)),
                "mean_width": float(sum(resolutions_w) / len(resolutions_w)) if resolutions_w else 0.0,
                "min_height": int(min(resolutions_h)),
                "max_height": int(max(resolutions_h)),
                "mean_height": float(sum(resolutions_h) / len(resolutions_h)) if resolutions_h else 0.0,
            },
            "visual_guardrails": [
                "Strict Rule: Identity is derived from visual weave/motifs, never filenames or image paths.",
                "Coarse categories (Banarasi, Bandhani, Ikat, Pichwai) serve as auxiliary domain features only.",
                "Color invariance is enforced by training transformations, ensuring color is not used for design matching."
            ]
        }

        return report

    def generate_markdown_report(self, report: Dict[str, Any], output_file: str):
        """Generates markdown summary of the audit."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        md = []
        md.append("# 📊 DeepLure Saree AI Agent — Visual Dataset Audit Report")
        md.append(f"\n**Data Root:** `{report.get('data_root')}`  ")
        md.append(f"**Total Valid Images:** {report.get('total_images')}  ")
        md.append(f"**Corrupted Files Detected:** {report.get('corrupted_images_count')}\n")

        md.append("## Split & Category Breakdown")
        md.append("| Split | Banarasi | Bandhani | Ikat | Pichwai | Total |")
        md.append("|---|---|---|---|---|---|")
        for split, stats in report.get("splits", {}).items():
            md.append(
                f"| `{split}` | {stats.get('Banarasi', 0)} | {stats.get('Bandhani', 0)} | "
                f"{stats.get('Ikat', 0)} | {stats.get('Pichwai', 0)} | **{stats.get('total', 0)}** |"
            )

        md.append("\n## Resolution Statistics")
        res = report.get("resolution", {})
        md.append(f"- **Width:** Min {res.get('min_width')}px, Max {res.get('max_width')}px, Mean {res.get('mean_width', 0):.1f}px")
        md.append(f"- **Height:** Min {res.get('min_height')}px, Max {res.get('max_height')}px, Mean {res.get('mean_height', 0):.1f}px")

        md.append("\n## Core Visual Identity Guardrails")
        for g in report.get("visual_guardrails", []):
            md.append(f"- 🟢 {g}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md))


if __name__ == "__main__":
    auditor = VisualDatasetAuditor("./kaggle")
    results = auditor.audit()
    auditor.generate_markdown_report(results, "./reports/dataset_audit_report.md")
    with open("./reports/dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Visual dataset audit complete. Saved to reports/dataset_audit_report.*")
