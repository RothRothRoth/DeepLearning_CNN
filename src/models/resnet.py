"""Approaches 2 and 3: ImageNet-pretrained ResNet-18 (torchvision).

- create_resnet18_frozen:    backbone frozen, only the new 26-class head trains
                             (transfer learning as a fixed feature extractor).
- create_resnet18_finetuned: same pretrained start, every layer trainable
                             (full fine-tuning).

Both load weights through the torchvision weights API
(ResNet18_Weights.IMAGENET1K_V1), not the deprecated `pretrained=True`.
The project's ImageNet normalisation in src/config.py matches these weights.
"""

from typing import List, Type

from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet

from src.config import NUM_CLASSES

# Fixed pretrained checkpoint so both approaches start from identical weights.
RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


class FrozenBackboneResNet(ResNet):
    """torchvision ResNet whose frozen BatchNorm layers never leave eval mode.

    requires_grad=False alone stops weight updates, but BatchNorm layers in
    train() mode still overwrite their running mean/variance with statistics
    of the current batch. Overriding train() keeps every BatchNorm layer whose
    parameters are frozen in eval mode, so the pretrained backbone stays fully
    fixed under any generic training loop that calls model.train().

    The module structure and state_dict keys are identical to torchvision's
    ResNet, so checkpoints are interchangeable.
    """

    def train(self, mode: bool = True) -> "FrozenBackboneResNet":
        super().train(mode)
        if mode:
            freeze_batchnorm_stats(self)
        return self


def _pretrained_resnet18_with_new_head(
    num_classes: int, model_cls: Type[ResNet] = ResNet
) -> ResNet:
    """Loads ImageNet ResNet-18 and replaces its 1000-way fc with a fresh head.

    Mirrors torchvision.models.resnet18(weights=...): builds
    ResNet(BasicBlock, [2, 2, 2, 2]) with 1000 outputs and loads the
    hash-checked pretrained state_dict. `model_cls` lets the frozen approach
    use FrozenBackboneResNet with exactly the same weights and layout.
    """
    model = model_cls(
        BasicBlock, [2, 2, 2, 2], num_classes=len(RESNET18_WEIGHTS.meta["categories"])
    )
    model.load_state_dict(RESNET18_WEIGHTS.get_state_dict(progress=True, check_hash=True))
    # New classifier: 512 pooled features -> num_classes logits (randomly initialised).
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_resnet18_frozen(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Pretrained ResNet-18 with a fully fixed backbone and a trainable classifier.

    Trainable:  model.fc (weight + bias) only.
    Frozen:     every other parameter (conv1, bn1, layer1-4), and the
                backbone BatchNorm running statistics, which stay in eval
                mode even after model.train() (see FrozenBackboneResNet).
    """
    model = _pretrained_resnet18_with_new_head(num_classes, FrozenBackboneResNet)

    # Freeze the whole pretrained backbone...
    for param in model.parameters():
        param.requires_grad = False
    # ...then explicitly unfreeze only the new classifier.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def create_resnet18_finetuned(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Pretrained ResNet-18 with a new classifier and ALL parameters trainable."""
    model = _pretrained_resnet18_with_new_head(num_classes)
    for param in model.parameters():
        param.requires_grad = True
    return model


def freeze_batchnorm_stats(model: nn.Module) -> nn.Module:
    """Puts every BatchNorm layer whose parameters are frozen into eval mode.

    Frozen BatchNorm layers then keep their ImageNet running mean/variance
    instead of re-estimating them from training batches. Called automatically
    by FrozenBackboneResNet.train().
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in m.parameters()
        ):
            m.eval()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Returns the total (or trainable-only) number of parameters in a model."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Returns parameters with requires_grad=True (pass these to the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]
