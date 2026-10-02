"""Reusable PyTorch training loop shared by all models (SmallCNN, ResNet, ...).

Design rules:
- train_model only accepts a train loader and a validation loader. There is
  no parameter for a test loader, so the test split cannot influence training,
  model selection, or early stopping.
- Model selection uses the validation set only; the best weights are restored
  into the model before train_model returns.
- Loss is CrossEntropyLoss on raw logits.
"""

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpoint import _head_out_features, save_checkpoint

# Metrics where a larger value is better; everything else is minimised.
_MAXIMIZE_METRICS = {"val_acc"}
_SUPPORTED_MONITORS = {"val_acc", "val_loss"}


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Returns CUDA if available (and preferred), otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accepts (images, labels) or (images, labels, meta) batches."""
    return batch[0], batch[1]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one optimisation pass over `loader`.

    Returns:
        (mean loss per sample, accuracy in [0, 1])
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Training loader produced no samples.")
    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates `model` on `loader` without updating weights.

    Returns:
        (mean loss per sample, accuracy in [0, 1])
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Validation loader produced no samples.")
    return running_loss / total, correct / total


def _is_improvement(value: float, best: Optional[float], monitor: str) -> bool:
    if best is None:
        return True
    if monitor in _MAXIMIZE_METRICS:
        return value > best
    return value < best


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: Optional[torch.device] = None,
    scheduler: Optional[Any] = None,
    criterion: Optional[nn.Module] = None,
    monitor: str = "val_acc",
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Trains `model` and restores the weights from the best validation epoch.

    Args:
        model: Network to train (moved to `device`).
        train_loader: Training split loader.
        val_loader: Validation split loader (used for model selection).
        optimizer: Optimizer over model parameters.
        num_epochs: Maximum number of epochs.
        device: Target device; defaults to CUDA if available else CPU.
        scheduler: Optional LR scheduler, stepped once per epoch.
            ReduceLROnPlateau is stepped with the monitored metric.
        criterion: Loss function; defaults to nn.CrossEntropyLoss().
        monitor: "val_acc" (maximised) or "val_loss" (minimised).
        early_stopping_patience: Stop after this many epochs without
            improvement. None disables early stopping.
        checkpoint_path: If given, the best checkpoint is saved here each time
            the monitored metric improves.
        model_name, num_classes, seed, hyperparameters: Stored in the
            checkpoint and history for reproducibility.
        verbose: Print one line per epoch.

    Returns:
        History dict with per-epoch lists (epoch, train_loss, val_loss,
        train_acc, val_acc, lr, epoch_time) plus best_epoch, best_metric,
        best_metric_name, total_time, epochs_completed, stopped_early, device.
    """
    if monitor not in _SUPPORTED_MONITORS:
        raise ValueError(f"monitor must be one of {_SUPPORTED_MONITORS}, got {monitor!r}")
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")

    device = device or get_device()
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "best_epoch": None,
        "best_metric": None,
        "best_metric_name": monitor,
        "total_time": 0.0,
        "epochs_completed": 0,
        "stopped_early": False,
        "device": str(device),
        "model_name": model_name,
    }

    best_state = copy.deepcopy(model.state_dict())
    best_metric: Optional[float] = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        # LR used during this epoch (before the scheduler steps).
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        metric = val_acc if monitor == "val_acc" else val_loss

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            else:
                scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        history["epochs_completed"] = epoch

        improved = _is_improvement(metric, best_metric, monitor)
        if improved:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_metric"] = metric
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                history["total_time"] = time.perf_counter() - start_time
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=metric,
                    history=history,
                    model_name=model_name,
                    num_classes=num_classes if num_classes is not None else _head_out_features(model),
                    seed=seed,
                    hyperparameters=hyperparameters,
                    best_metric_name=monitor,
                    scheduler=scheduler,
                )
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"lr {current_lr:.2e} | {epoch_time:.1f}s"
                + (" *" if improved else "")
            )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            history["stopped_early"] = True
            if verbose:
                print(f"Early stopping: no improvement for {early_stopping_patience} epochs.")
            break

    history["total_time"] = time.perf_counter() - start_time

    # Restore the weights from the best validation epoch.
    model.load_state_dict(best_state)

    if verbose:
        print(
            f"Best {monitor} = {history['best_metric']:.4f} at epoch {history['best_epoch']} "
            f"| total time {history['total_time']:.1f}s"
        )
    return history
