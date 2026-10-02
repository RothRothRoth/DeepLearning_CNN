"""Image transformations and data augmentation for PyTorch traffic sign models.

Ensures:
- 224x224 input resolution.
- Standard ImageNet normalization.
- Rotation and color jitter for training.
- No horizontal flipping to preserve directional sign semantics.
- Deterministic transforms for validation and testing.
"""

from typing import Tuple
import torch
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Returns training data transformations with mild domain-safe augmentations.

    Note: RandomHorizontalFlip is deliberately omitted to preserve
    directional traffic sign semantics (e.g. Left Bend vs. Right Bend).
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Returns deterministic transformations for validation and test evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization on a tensor image [C, H, W] or batch [B, C, H, W].

    Clamps values to [0.0, 1.0] for safe visualization with matplotlib or PIL.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, device=tensor.device)

    if tensor.dim() == 4:
        # Batch of images: [B, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.dim() == 3:
        # Single image: [C, H, W]
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    denorm = tensor * std + mean
    return torch.clamp(denorm, 0.0, 1.0)
