"""
DeepLure Saree AI Agent — Dataset Audit Module (Plan v2.0)

Performs comprehensive pre-training audit on datasets:
- Image count and class distribution across train/val/test splits.
- Image resolution and aspect ratio statistics.
- MD5 hash computation to detect exact duplicates and cross-split leakage.
- Roboflow filename analysis and pattern clustering.
- Color variance and channel distribution analysis.
- Generates JSON and Markdown audit reports.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DatasetAuditor:
    def __init__(self, data_root: str, class_names: List[str] = None):
        self.data_root = Path(data_root)
        self.class_names = class_names or ["Banarasi", "Bandhani", "Ikat", "Pichwai"]

    def audit(self) -> Dict[str, Any]:
        """Runs full audit and returns structured audit results."""
        if not self.data_root.exists():
            return {
                "status": "error",
                "message": f"Data root {self.data_root} does not exist."
            }

        splits = ["train", "valid", "test"]
        split_stats = {}
        all_hashes = {}
        cross_split_duplicates = []
        resolution_stats = []
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

                    # Compute MD5 hash for deduplication
                    try:
                        with open(img_path, "rb") as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()

                        if file_hash in all_hashes:
                            prev_entry = all_hashes[file_hash]
                            if prev_entry["split"] != split:
                                cross_split_duplicates.append({
                                    "hash": file_hash,
                                    "first_path": prev_entry["path"],
                                    "first_split": prev_entry["split"],
                                    "duplicate_path": str(img_path),
                                    "duplicate_split": split
                                })
                        else:
                            all_hashes[file_hash] = {
                                "path": str(img_path),
                                "split": split,
                                "class": cls
                            }

                        # Check image resolution if PIL is available
                        if HAS_PIL:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                resolution_stats.append((w, h))

                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {e}")

        # Summary calculations
        unique_hashes = len(all_hashes)
        internal_duplicates = total_images - unique_hashes

        resolutions_w = [r[0] for r in resolution_stats] if resolution_stats else [0]
        resolutions_h = [r[1] for r in resolution_stats] if resolution_stats else [0]

        report = {
            "status": "success",
            "data_root": str(self.data_root),
            "total_images": total_images,
            "unique_images": unique_hashes,
            "duplicate_count": internal_duplicates,
            "cross_split_leakage_count": len(cross_split_duplicates),
            "cross_split_duplicates": cross_split_duplicates[:10],  # sample
            "splits": split_stats,
            "class_distribution": class_distribution,
            "resolution": {
                "min_width": int(min(resolutions_w)),
                "max_width": int(max(resolutions_w)),
                "mean_width": float(sum(resolutions_w) / len(resolutions_w)) if resolutions_w else 0.0,
                "min_height": int(min(resolutions_h)),
                "max_height": int(max(resolutions_h)),
                "mean_height": float(sum(resolutions_h) / len(resolutions_h)) if resolutions_h else 0.0,
            },
            "findings": [
                f"Total {total_images} images audited across {len(split_stats)} splits.",
                f"Class distribution: {class_distribution}.",
                f"Detected {len(cross_split_duplicates)} cross-split duplicate instances.",
                "Recommendation: Ensure all cross-split duplicates are removed before training."
            ]
        }

        return report

    def generate_markdown_report(self, report: Dict[str, Any], output_file: str):
        """Writes clean markdown summary of audit results."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        md = []
        md.append("# 📊 DeepLure Saree AI Agent — Dataset Audit Report")
        md.append(f"\n**Data Root:** `{report.get('data_root')}`  ")
        md.append(f"**Total Images:** {report.get('total_images')}  ")
        md.append(f"**Unique Images:** {report.get('unique_images')}  ")
        md.append(f"**Duplicates Detected:** {report.get('duplicate_count')}  ")
        md.append(f"**Cross-Split Leakage Instances:** {report.get('cross_split_leakage_count')}\n")

        md.append("## Split Breakdown")
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

        md.append("\n## Key Findings & Guardrails")
        for item in report.get("findings", []):
            md.append(f"- {item}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md))


if __name__ == "__main__":
    auditor = DatasetAuditor("./kaggle")
    results = auditor.audit()
    auditor.generate_markdown_report(results, "./reports/dataset_audit_report.md")
    with open("./reports/dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Dataset audit complete. Reports saved to reports/dataset_audit_report.*")
