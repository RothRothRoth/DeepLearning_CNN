"""Sanity checks for the shared training loop and checkpointing.

Uses SmallCNN with small random dummy tensors only; the real dataset is never
touched. Run with either:
    python -m pytest tests/test_training.py -v
    python -m tests.test_training
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_SEED, NUM_CLASSES
from src.models.cnn import SmallCNN
from src.seed import set_seed
from src.training import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    train_model,
    train_one_epoch,
    validate,
)

IMG_SIZE = 64  # SmallCNN uses global pooling, so a small size keeps the test fast.
HISTORY_KEYS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"]


def _dummy_loader(n: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _setup():
    set_seed(DEFAULT_SEED)
    device = get_device()
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, device


def test_train_one_epoch():
    model, optimizer, device = _setup()
    before = [p.detach().clone() for p in model.parameters()]
    loss, acc = train_one_epoch(
        model, _dummy_loader(32, 0), torch.nn.CrossEntropyLoss(), optimizer, device
    )
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
    assert 0.0 <= acc <= 1.0
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))


def test_validate():
    model, _, device = _setup()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss, acc = validate(model, _dummy_loader(16, 1), torch.nn.CrossEntropyLoss(), device)
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
    assert 0.0 <= acc <= 1.0
    # Validation must not change weights or BatchNorm running stats.
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v), f"validate() modified {k}"


def test_train_model_history_and_best_restore():
    model, optimizer, device = _setup()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    epochs = 3
    history = train_model(
        model,
        _dummy_loader(32, 0),
        _dummy_loader(16, 1),
        optimizer,
        num_epochs=epochs,
        device=device,
        scheduler=scheduler,
        monitor="val_loss",
        model_name="small_cnn",
        seed=DEFAULT_SEED,
        verbose=False,
    )
    for key in HISTORY_KEYS:
        assert len(history[key]) == epochs, key
    assert history["lr"] == [1e-3, 5e-4, 2.5e-4]
    assert history["total_time"] >= sum(history["epoch_time"]) * 0.99
    assert history["best_epoch"] in history["epoch"]
    assert history["best_metric"] == min(history["val_loss"])
    # Restored model must reproduce the best validation loss.
    val_loss, _ = validate(model, _dummy_loader(16, 1), torch.nn.CrossEntropyLoss(), device)
    assert abs(val_loss - history["best_metric"]) < 1e-5


def test_checkpoint_save_load_restore():
    model, optimizer, device = _setup()
    history = train_model(
        model, _dummy_loader(32, 0), _dummy_loader(16, 1), optimizer,
        num_epochs=1, device=device, verbose=False,
    )
    hparams = {"lr": 1e-3, "batch_size": 8, "optimizer": "Adam", "epochs": 1}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "small_cnn.pt"
        save_checkpoint(
            path, model, optimizer, epoch=1, best_metric=history["best_metric"],
            history=history, model_name="small_cnn", num_classes=NUM_CLASSES,
            seed=DEFAULT_SEED, hyperparameters=hparams,
        )
        assert path.is_file()

        # Fresh model with different random weights.
        set_seed(DEFAULT_SEED + 1)
        restored = SmallCNN().to(device)
        restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(model.state_dict().values(), restored.state_dict().values())
        )

        ckpt = load_checkpoint(path, restored, restored_opt, map_location=device)

    for k, v in model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), f"mismatch in {k}"
    assert restored_opt.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert ckpt["model_name"] == "small_cnn"
    assert ckpt["num_classes"] == NUM_CLASSES
    assert ckpt["seed"] == DEFAULT_SEED
    assert ckpt["epoch"] == 1
    assert ckpt["hyperparameters"] == hparams
    assert ckpt["best_metric"] == history["best_metric"]
    assert ckpt["history"]["val_acc"] == history["val_acc"]


def test_train_model_writes_best_checkpoint():
    model, optimizer, device = _setup()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        history = train_model(
            model, _dummy_loader(32, 0), _dummy_loader(16, 1), optimizer,
            num_epochs=2, device=device, checkpoint_path=path,
            model_name="small_cnn", seed=DEFAULT_SEED, verbose=False,
        )
        ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == history["best_epoch"]
    assert ckpt["num_classes"] == NUM_CLASSES
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), ckpt["model_state_dict"][k]), f"mismatch in {k}"


if __name__ == "__main__":
    print(f"Device: {get_device()}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
