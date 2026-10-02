"""Architecture sanity checks for SmallCNN (no training, no data required).

Run with either:
    python -m pytest tests/test_cnn.py -v
    python -m tests.test_cnn
"""

import torch

from src.config import IMAGE_SIZE, NUM_CLASSES
from src.models.cnn import SmallCNN, count_parameters

BATCH_SIZE = 2


def _forward() -> torch.Tensor:
    torch.manual_seed(0)
    model = SmallCNN()
    model.eval()
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        return model(x)


def test_instantiation():
    model = SmallCNN()
    assert isinstance(model, torch.nn.Module)


def test_output_shape():
    assert NUM_CLASSES == 26
    assert tuple(_forward().shape) == (BATCH_SIZE, NUM_CLASSES)


def test_no_nan():
    out = _forward()
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()


def test_configurable_num_classes():
    model = SmallCNN(num_classes=10).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert tuple(out.shape) == (BATCH_SIZE, 10)


def test_parameter_counts():
    model = SmallCNN()
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    assert total > 0
    assert trainable == total
    # ResNet-18 has ~11.2M parameters; the from-scratch baseline must be far smaller.
    assert total < 1_000_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    model = SmallCNN()
    print(f"Total parameters:     {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
