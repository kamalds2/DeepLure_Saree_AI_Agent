"""
data/dataset.py
===============
PyTorch Dataset classes for the DeepLure Saree AI Agent.

TWO DATASETS:
    1. KaggleSareeDataset  — Indian Saree Patterns (Kaggle/Roboflow, MIT License)
       - 4 style classes: Banarasi, Bandhani, Ikat, Pichwai
       - Used for: ArcFace pretraining (style-level metric learning)
       - Structure: kaggle/{train|valid|test}/{ClassName}/*.jpg

    2. HandloomDataset     — DeepLure proprietary corpus
       - No class labels (flat directory of JPG files)
       - Used for: Gallery + Query evaluation
       - Structure: handloom_sarees/*.jpg

DESIGN DECISIONS:

    KaggleSareeDataset:
        - We use torchvision ImageFolder semantics but explicitly control class mapping
          so Banarasi=0, Bandhani=1, Ikat=2, Pichwai=3 across ALL splits.
        - WHY? ImageFolder assigns labels alphabetically, which by luck gives us
          Banarasi→0, Bandhani→1, Ikat→2, Pichwai→3. We make this explicit.
        - Balanced sampler is handled externally in train.py using WeightedRandomSampler.

    HandloomDataset:
        - Returns label=-1 (unknown) for all images since there are no labels.
        - Also returns the filename stem so we can group by filename at eval time.
        - The filename IS the identity for retrieval purposes (evaluation is done
          externally with proper pair construction).
"""

import os
from glob import glob
from typing import List, Optional, Tuple, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Class name → integer label mapping (consistent across all splits)
# ---------------------------------------------------------------------------
CLASS_TO_IDX = {
    "Banarasi": 0,
    "Bandhani": 1,
    "Ikat":     2,
    "Pichwai":  3,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)


class KaggleSareeDataset(Dataset):
    """
    Dataset for the Kaggle Indian Saree Patterns corpus.

    Directory structure expected:
        root/
          train/
            Banarasi/  *.jpg
            Bandhani/  *.jpg
            Ikat/      *.jpg
            Pichwai/   *.jpg
          valid/ ...
          test/  ...

    Args:
        root: Path to kaggle/ directory
        split: One of "train", "valid", "test"
        transform: torchvision transforms to apply (from transforms.py)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "valid", "test"), (
            f"split must be 'train', 'valid', or 'test', got '{split}'"
        )
        self.root = root
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []  # (path, label)
        self._load_samples()

    def _load_samples(self):
        """Scan the directory and build (path, label) pairs."""
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected structure: {self.root}/{self.split}/{{ClassName}}/*.jpg"
            )

        for class_name, label in CLASS_TO_IDX.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                # Gracefully skip missing classes (test set might be sparse)
                print(f"  [KaggleSareeDataset] Warning: class dir not found: {class_dir}")
                continue
            files = sorted(glob(os.path.join(class_dir, "*.jpg")))
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {os.path.join(self.root, self.split)}. "
                "Check that the Kaggle dataset is extracted correctly."
            )

        # Print class distribution (important for detecting imbalance)
        class_counts = [0] * NUM_CLASSES
        for _, lbl in self.samples:
            class_counts[lbl] += 1
        print(f"[KaggleSareeDataset] {self.split}: {len(self.samples)} images")
        for cls, count in zip(CLASS_TO_IDX.keys(), class_counts):
            print(f"  {cls}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label): image as torch.Tensor [3, H, W], label as int
        """
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        WHY? Banarasi has 432 images vs Bandhani/Pichwai with 279 each.
        Balanced sampling prevents ArcFace from biasing toward the majority class.
        Each sample weight = 1 / count_of_its_class.

        Returns:
            weights: Tensor of shape [len(dataset)] for WeightedRandomSampler
        """
        class_counts = [0] * NUM_CLASSES
        for _, lbl in self.samples:
            class_counts[lbl] += 1

        # weight per class = 1/count (rarer class = higher weight)
        class_weights = [
            1.0 / c if c > 0 else 0.0
            for c in class_counts
        ]
        # Assign per-sample weight
        sample_weights = torch.tensor(
            [class_weights[lbl] for _, lbl in self.samples],
            dtype=torch.float32,
        )
        return sample_weights


class HandloomDataset(Dataset):
    """
    Dataset for the DeepLure proprietary handloom corpus.

    All images are in a single flat directory with no class labels.
    Labels are set to -1 (unknown). Filenames are returned so that
    evaluation scripts can group images by filename-based pseudo-identity.

    ⚠️  PROPRIETARY DATA — Do not redistribute. Delete after exercise.

    Args:
        image_paths: List of absolute image paths (from splits.py)
        transform: torchvision transforms to apply
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
    ):
        if not image_paths:
            raise ValueError("image_paths is empty — no handloom images provided.")
        self.image_paths = image_paths
        self.transform = transform
        print(f"[HandloomDataset] Loaded {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            (image_tensor, label, filename_stem):
                - image_tensor: torch.Tensor [3, H, W]
                - label: always -1 (no ground truth labels)
                - filename_stem: e.g. "img_106019" (without .jpg extension)
        """
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename_stem = os.path.splitext(os.path.basename(path))[0]
        return image, -1, filename_stem


class VerificationPairDataset(Dataset):
    """
    Dataset for verification evaluation — yields pairs (img_A, img_B, is_same).

    WHY A SEPARATE DATASET FOR PAIRS?
        Verification is a binary classification task on image pairs.
        We build this from the Kaggle dataset using style class labels as
        ground truth: same class = same design (positive pair),
        different class = different design (negative pair).

        This is an approximation — two Banarasi images may have different
        specific designs. But at the style/category level, it provides a
        usable verification benchmark.

    Args:
        dataset: A KaggleSareeDataset instance
        num_pairs: Total number of pairs to generate
        pos_ratio: Fraction of pairs that are positive (same class)
        seed: Random seed
    """

    def __init__(
        self,
        dataset: KaggleSareeDataset,
        num_pairs: int = 2000,
        pos_ratio: float = 0.5,
        seed: int = 42,
    ):
        import random
        random.seed(seed)

        self.dataset = dataset
        self.pairs: List[Tuple[int, int, int]] = []  # (idx_a, idx_b, is_same)

        # Group indices by class
        class_indices = {c: [] for c in range(NUM_CLASSES)}
        for idx, (_, lbl) in enumerate(dataset.samples):
            class_indices[lbl].append(idx)

        n_pos = int(num_pairs * pos_ratio)
        n_neg = num_pairs - n_pos

        # Build positive pairs (same class)
        for _ in range(n_pos):
            cls = random.randint(0, NUM_CLASSES - 1)
            if len(class_indices[cls]) < 2:
                continue
            a, b = random.sample(class_indices[cls], 2)
            self.pairs.append((a, b, 1))

        # Build negative pairs (different classes)
        for _ in range(n_neg):
            cls_a, cls_b = random.sample(range(NUM_CLASSES), 2)
            a = random.choice(class_indices[cls_a])
            b = random.choice(class_indices[cls_b])
            self.pairs.append((a, b, 0))

        random.shuffle(self.pairs)
        n_pos_actual = sum(1 for _, _, s in self.pairs if s == 1)
        n_neg_actual = sum(1 for _, _, s in self.pairs if s == 0)
        print(f"[VerificationPairDataset] {len(self.pairs)} pairs "
              f"(pos={n_pos_actual}, neg={n_neg_actual})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            (img_a, img_b, is_same):
                - img_a, img_b: torch.Tensor [3, H, W]
                - is_same: 1 if same class, 0 if different class
        """
        idx_a, idx_b, is_same = self.pairs[idx]
        img_a, _ = self.dataset[idx_a]
        img_b, _ = self.dataset[idx_b]
        return img_a, img_b, is_same
