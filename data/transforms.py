"""
data/transforms.py
==================
Color-invariant augmentation pipeline for DeepLure Saree AI.

WHY THIS MODULE EXISTS:
    The core challenge is COLOR INVARIANCE — the same saree weave pattern
    rendered in different colorways must produce similar embeddings. The most
    direct way to enforce this at training time is to make color an unreliable
    feature by aggressively randomizing it. This module centralizes all
    augmentation logic so every experiment uses the same pipeline.

DESIGN DECISIONS:
    1. ColorJitter(hue=0.3, sat=0.8): Forces the network away from color cues.
       If hue and saturation change drastically every batch, the network learns
       to rely on luminance/texture structure instead.

    2. RandomGrayscale(p): If 30% of the time the image is grayscale, the
       network literally cannot use color for those samples. This is the
       strongest color-invariance teacher during training.

    3. RandomVerticalFlip: Saree weave patterns are often symmetric top-to-bottom
       (especially border designs). Being invariant to vertical orientation helps.

    4. RandomRotation(15°): Fabric is sometimes photographed at slight angles.
       Beyond 15° we start destroying the periodic weave pattern structure, so
       we cap there.

    5. Normalize with ImageNet stats: We use EfficientNet-B3 pretrained on
       ImageNet — it expects images normalized to ImageNet mean/std.
"""

import torchvision.transforms as T


# ---------------------------------------------------------------------------
# ImageNet normalization constants (must match pretrained EfficientNet-B3)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    grayscale_prob: float = 0.3,
    color_jitter_brightness: float = 0.4,
    color_jitter_contrast: float = 0.4,
    color_jitter_saturation: float = 0.8,
    color_jitter_hue: float = 0.3,
    rotation_degrees: float = 15.0,
) -> T.Compose:
    """
    Build training transforms with color-invariant augmentation.

    Args:
        image_size: Final crop size (224 for EfficientNet-B3 compatibility)
        resize_size: Resize before crop (256 gives a bit of crop randomness)
        grayscale_prob: Probability of converting to grayscale (phase 3 bumps to 0.7)
        color_jitter_*: ColorJitter parameters — see docstring above
        rotation_degrees: Max rotation in degrees

    Returns:
        torchvision.transforms.Compose pipeline

    NOTE:
        The order of transforms matters:
        - Geometric transforms (flip, rotate, crop) come BEFORE color transforms
          because random crop + flip needs the full image.
        - ColorJitter before Grayscale: jitter in color space, then optionally flatten.
        - ToTensor + Normalize always last.
    """
    return T.Compose([
        # --- Geometric (spatial structure) ---
        T.Resize(resize_size),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),     # Sarees are often vertically symmetric
        T.RandomRotation(degrees=rotation_degrees),

        # --- Color (invariance) ---
        # WHY ColorJitter before Grayscale?
        # Apply color distortion first so the grayscale conversion (if triggered)
        # operates on an already-distorted color image. This prevents the model
        # from detecting whether an image was jitter-converted to grayscale
        # vs. natively grayscale, creating a stronger invariance signal.
        T.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        ),
        T.RandomGrayscale(p=grayscale_prob),  # Output is 3-ch (duplicated L channel)

        # --- Tensor conversion + normalization ---
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Deterministic validation / inference transforms (no randomness).

    WHY center crop instead of random crop?
        At inference time we want a deterministic result for the same image.
        Center crop is the standard for ImageNet-pretrained models.

    Args:
        image_size: Target size (224)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return T.Compose([
        T.Resize(image_size + 32),   # Slight oversize then center crop
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_phase3_transforms(
    image_size: int = 224,
    resize_size: int = 256,
) -> T.Compose:
    """
    Phase 3 transforms: maximum color invariance enforcement.

    WHY a separate phase 3 pipeline?
        Phase 3 is a dedicated color-invariance fine-tuning phase where we
        increase RandomGrayscale probability from 0.3 to 0.7. 70% of training
        images will be grayscale — the model is forced to rely almost entirely
        on texture/luminance. This acts as a final "color-blindness" calibration.

    Args:
        image_size: 224
        resize_size: 256

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return T.Compose([
        T.Resize(resize_size),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),           # Gentler rotation in phase 3
        T.ColorJitter(                          # Full jitter still applied
            brightness=0.3,
            contrast=0.3,
            saturation=0.9,                     # Even more aggressive saturation
            hue=0.4,                            # Even more aggressive hue
        ),
        T.RandomGrayscale(p=0.7),              # KEY CHANGE: 0.3 → 0.7
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
