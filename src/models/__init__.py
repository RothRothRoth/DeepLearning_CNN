"""Model definitions: SmallCNN (from scratch) and ResNet-18 (pretrained)."""

from src.models.cnn import SmallCNN, count_parameters
from src.models.resnet import (
    FrozenBackboneResNet,
    create_resnet18_finetuned,
    create_resnet18_frozen,
    freeze_batchnorm_stats,
    get_trainable_parameters,
)

__all__ = [
    "FrozenBackboneResNet",
    "SmallCNN",
    "count_parameters",
    "create_resnet18_finetuned",
    "create_resnet18_frozen",
    "freeze_batchnorm_stats",
    "get_trainable_parameters",
]
