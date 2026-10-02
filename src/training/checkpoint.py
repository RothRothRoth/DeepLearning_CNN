"""Checkpoint save/load utilities shared by all models (SmallCNN, ResNet, ...).

Checkpoints are plain dictionaries written with torch.save. Everything except
the state_dicts is stored as Python primitives (str, int, float, bool, list,
dict, None), so checkpoints load with torch.load(weights_only=True) and never
execute arbitrary pickled code.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION: int = 1


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    history: Dict[str, Any],
    model_name: str,
    num_classes: int,
    seed: Optional[int],
    hyperparameters: Optional[Dict[str, Any]] = None,
    best_metric_name: str = "val_acc",
    scheduler: Optional[Any] = None,
) -> Path:
    """Saves a training checkpoint to `path` (parent directories are created).

    Args:
        path: Destination file, e.g. models/small_cnn_best.pt.
        model: Model whose weights are saved.
        optimizer: Optimizer whose state is saved (None to skip).
        epoch: Epoch number (1-based) these weights correspond to.
        best_metric: Best validation metric value reached so far.
        history: Training history dict returned by train_model.
        model_name: Identifier such as "small_cnn" or "resnet18".
        num_classes: Size of the classification head.
        seed: Random seed used for the run.
        hyperparameters: Primitive-valued dict (lr, batch_size, epochs, ...).
        best_metric_name: Name of the monitored metric ("val_acc" / "val_loss").
        scheduler: Optional LR scheduler whose state is saved.

    Returns:
        The resolved checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name,
        "num_classes": int(num_classes),
        "seed": seed,
        "epoch": int(epoch),
        "best_metric_name": best_metric_name,
        "best_metric": float(best_metric),
        "hyperparameters": dict(hyperparameters or {}),
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Loads a checkpoint and optionally restores model/optimizer/scheduler state.

    Args:
        path: Checkpoint file written by save_checkpoint.
        model: If given, its weights are restored in place.
        optimizer: If given, its state is restored in place.
        scheduler: If given, its state is restored in place.
        map_location: Device to map tensors onto (default CPU, safe everywhere).
        strict: Passed to model.load_state_dict.

    Returns:
        The full checkpoint dictionary (metadata, history, state_dicts).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    if model is not None:
        expected = checkpoint.get("num_classes")
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        out_features = _head_out_features(model)
        if expected is not None and out_features is not None and out_features != expected:
            raise ValueError(
                f"Checkpoint has {expected} classes but model head has {out_features}."
            )

    if optimizer is not None:
        if checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("Checkpoint does not contain an optimizer state.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        if checkpoint.get("scheduler_state_dict") is None:
            raise ValueError("Checkpoint does not contain a scheduler state.")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def _head_out_features(model: nn.Module) -> Optional[int]:
    """Returns out_features of the model's last Linear layer, if any."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last.out_features if last is not None else None
