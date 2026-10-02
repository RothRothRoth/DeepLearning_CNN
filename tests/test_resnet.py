"""Architecture sanity checks for the ResNet-18 models (dummy tensors only).

The first run downloads the ImageNet ResNet-18 weights (~45 MB) into the
torch hub cache. Run with either:
    python -m pytest tests/test_resnet.py -v
    python -m tests.test_resnet
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet

from src.config import IMAGE_SIZE, NUM_CLASSES
from src.models import (
    count_parameters,
    create_resnet18_finetuned,
    create_resnet18_frozen,
    get_trainable_parameters,
)
from src.models.resnet import RESNET18_WEIGHTS, FrozenBackboneResNet
from src.training import get_device, train_one_epoch

BATCH_SIZE = 2
RESNET18_IMAGENET_PARAMS = 11_689_512  # torchvision ResNet-18 with 1000-way fc
FC_PARAMS = 512 * NUM_CLASSES + NUM_CLASSES

_models = {}


def _model(kind: str) -> nn.Module:
    """Builds each model once per test run (weights load from the local cache)."""
    if kind not in _models:
        factory = create_resnet18_frozen if kind == "frozen" else create_resnet18_finetuned
        _models[kind] = factory()
    return _models[kind]


def _forward(model: nn.Module) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        return model(x)


def test_import():
    assert callable(create_resnet18_frozen)
    assert callable(create_resnet18_finetuned)
    assert RESNET18_WEIGHTS.name == "IMAGENET1K_V1"


def test_output_shape_and_finite():
    assert NUM_CLASSES == 26
    for kind in ("frozen", "finetuned"):
        out = _forward(_model(kind))
        assert tuple(out.shape) == (BATCH_SIZE, NUM_CLASSES), kind
        assert torch.isfinite(out).all(), f"{kind} output has NaN/Inf"


def test_classifier_layer():
    for kind in ("frozen", "finetuned"):
        fc = _model(kind).fc
        assert isinstance(fc, nn.Linear)
        assert fc.in_features == 512
        assert fc.out_features == NUM_CLASSES


def test_pretrained_weights_loaded():
    # Both approaches must start from identical ImageNet backbone weights.
    frozen = _model("frozen").state_dict()
    finetuned = _model("finetuned").state_dict()
    for k, v in frozen.items():
        if not k.startswith("fc."):
            assert torch.equal(v, finetuned[k]), f"backbone mismatch at {k}"
    # A freshly initialised conv1 would not have these ImageNet statistics.
    assert abs(frozen["bn1.running_mean"].abs().mean().item()) > 0


def test_frozen_backbone():
    model = _model("frozen")
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"
    assert count_parameters(model, trainable_only=True) == FC_PARAMS
    trainable = get_trainable_parameters(model)
    assert len(trainable) == 2  # fc.weight, fc.bias


def test_frozen_gradients_reach_only_fc():
    model = _model("frozen")
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    nn.CrossEntropyLoss()(model(x), y).backward()
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            assert p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None, f"{name} received a gradient"
    model.zero_grad(set_to_none=True)
    model.eval()


def _bn_stats(model: nn.Module) -> dict:
    return {
        k: v.clone()
        for k, v in model.state_dict().items()
        if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def _dummy_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # Shifted/scaled inputs so batch statistics differ clearly from ImageNet's.
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g) * 3 + 2
    y = torch.randint(0, NUM_CLASSES, (4,), generator=g)
    return x, y


def test_frozen_bn_stats_fixed_in_train_mode():
    model = create_resnet18_frozen()  # fresh model: this test runs forward passes
    assert isinstance(model, FrozenBackboneResNet)
    model.train()

    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 20 and all(not m.training for m in bns)
    assert model.training and model.fc.training

    before = _bn_stats(model)
    x, _ = _dummy_batch()
    with torch.no_grad():
        model(x)
    after = _bn_stats(model)
    for k in before:
        assert torch.equal(before[k], after[k]), f"{k} changed in train mode"

    # Still frozen backbone, trainable classifier.
    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("fc."), name
    # eval() still works normally.
    model.eval()
    assert not model.training and not model.fc.training


def test_frozen_model_with_shared_training_engine():
    model = create_resnet18_frozen()
    device = get_device()
    model.to(device)
    before_bn = _bn_stats(model)
    before_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    x, y = _dummy_batch()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    optimizer = torch.optim.SGD(get_trainable_parameters(model), lr=0.1)
    train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, device)

    for k, v in _bn_stats(model).items():
        assert torch.equal(before_bn[k], v), f"{k} changed during training"
    for name, p in model.named_parameters():
        changed = not torch.equal(before_params[name], p.detach())
        if name.startswith("fc."):
            assert changed, f"{name} was not updated"
        else:
            assert not changed, f"{name} was updated"


def test_finetuned_bn_stats_update_in_train_mode():
    model = create_resnet18_finetuned()
    assert type(model) is ResNet  # plain torchvision ResNet, normal train() behaviour
    model.train()
    assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    before = _bn_stats(model)
    x, _ = _dummy_batch()
    with torch.no_grad():
        model(x)
    after = _bn_stats(model)
    assert not torch.equal(before["bn1.running_mean"], after["bn1.running_mean"])


def test_matches_torchvision_resnet18():
    reference = resnet18(weights=RESNET18_WEIGHTS).state_dict()
    for kind in ("frozen", "finetuned"):
        state = _model(kind).state_dict()
        assert state.keys() == reference.keys(), kind
        for k, v in reference.items():
            if not k.startswith("fc."):
                assert torch.equal(state[k], v), f"{kind}: {k} differs from torchvision"


def test_finetuned_all_trainable():
    model = _model("finetuned")
    assert all(p.requires_grad for p in model.parameters())
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_parameter_counts():
    expected_total = RESNET18_IMAGENET_PARAMS - (512 * 1000 + 1000) + FC_PARAMS
    for kind in ("frozen", "finetuned"):
        assert count_parameters(_model(kind)) == expected_total, kind


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    for kind in ("frozen", "finetuned"):
        m = _model(kind)
        total = count_parameters(m)
        trainable = count_parameters(m, trainable_only=True)
        print(
            f"ResNet-18 {kind:9s}: total {total:,} | trainable {trainable:,} "
            f"| frozen {total - trainable:,}"
        )
