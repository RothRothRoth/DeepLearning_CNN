"""Reproducibility utilities for Python, NumPy, and PyTorch."""

import os
import random
import numpy as np
import torch

from src.config import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures cuDNN for deterministic execution.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoaders to ensure per-worker reproducibility.

    Args:
        worker_id: DataLoader worker index.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
