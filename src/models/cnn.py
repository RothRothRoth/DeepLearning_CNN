"""Approach 1: a small convolutional neural network trained from scratch.

SmallCNN is a deliberately compact baseline for 26-class Cambodian road sign
classification. It uses no pretrained weights and no torchvision backbones,
so every parameter is learned from the CamTSD training split alone.

Expected input:  [batch, 3, 224, 224] (normalized RGB crops)
Output:          [batch, num_classes] raw logits (no softmax; use CrossEntropyLoss)
"""

from typing import Tuple

import torch
from torch import nn

from src.config import IMAGE_SIZE, NUM_CLASSES


class ConvBlock(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.

    - Conv2d (3x3, padding=1) learns local spatial features while preserving
      height/width. Bias is omitted because BatchNorm adds its own shift.
    - BatchNorm2d stabilises activations so the network trains reliably from
      random initialisation.
    - ReLU adds non-linearity.
    - MaxPool2d (2x2) halves the spatial resolution, growing the receptive
      field and reducing compute for the next block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class SmallCNN(nn.Module):
    """Three-block CNN from scratch for road sign classification.

    Spatial flow for a 224x224 input:
        [B, 3, 224, 224]
        -> block1 -> [B, 32, 112, 112]
        -> block2 -> [B, 64, 56, 56]
        -> block3 -> [B, 128, 28, 28]
        -> global average pool -> [B, 128, 1, 1]
        -> flatten -> [B, 128]
        -> FC + ReLU + Dropout -> [B, hidden_dim]
        -> classifier -> [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        channels: Tuple[int, int, int] = (32, 64, 128),
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels

        # Feature extractor: three conv blocks, each doubling channels and
        # halving resolution (low-level edges -> shapes -> sign-level patterns).
        self.features = nn.Sequential(
            ConvBlock(in_channels, c1),  # Block 1: colours, edges
            ConvBlock(c1, c2),  # Block 2: corners, simple shapes
            ConvBlock(c2, c3),  # Block 3: sign parts, symbols, digits
        )

        # Global average pooling collapses each feature map to a single value,
        # making the head independent of input size and avoiding a huge
        # flattened FC layer (the main source of parameters in naive CNNs).
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Classification head.
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, c3, 1, 1] -> [B, c3]
            nn.Linear(c3, hidden_dim),  # combine pooled features
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # regularisation against overfitting
            nn.Linear(hidden_dim, num_classes),  # one logit per class
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv/linear layers (suited to ReLU), unit BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Returns the total (or trainable-only) number of parameters in a model."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


if __name__ == "__main__":
    model = SmallCNN()
    print(model)
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    print("Output shape:", tuple(model(dummy).shape))
    print("Total parameters:", count_parameters(model))
    print("Trainable parameters:", count_parameters(model, trainable_only=True))
