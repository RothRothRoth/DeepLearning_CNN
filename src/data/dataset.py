"""PyTorch Dataset and DataLoader implementations for Cambodian Road Signs."""

from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_SEED,
    NUM_CLASSES,
    NUM_WORKERS,
    ProjectPaths,
    get_paths,
)
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.seed import seed_worker


def resolve_crop_path(processed_root: Union[str, Path], crop_path: str) -> Path:
    """Resolves a manifest crop_path against the processed-crops root.

    Manifests store portable POSIX paths relative to dataset/processed/, e.g.
    "train/KM_POST/example.jpg". This joins them onto `processed_root` with
    pathlib, so the same manifest works on Windows, Linux and Colab.
    """
    crop_path = str(crop_path)
    posix = PurePosixPath(crop_path)
    if "\\" in crop_path or posix.is_absolute() or ".." in posix.parts or ":" in crop_path:
        raise ValueError(
            f"Non-portable crop_path {crop_path!r}: expected a POSIX path relative to "
            "dataset/processed/, e.g. 'train/KM_POST/example.jpg'."
        )
    return Path(processed_root).joinpath(*posix.parts)


class TrafficSignDataset(Dataset):
    """Custom PyTorch Dataset for Cambodian Traffic Sign crops."""

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        processed_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        """Initializes the dataset.

        Args:
            data_source: pandas DataFrame or path to split CSV manifest.
            processed_root: dataset/processed/ directory that manifest crop_path
                values are relative to. Defaults to get_paths().processed_dir.
            transform: PyTorch image transformation callable.
            return_meta: If True, returns additional metadata dict in __getitem__.
        """
        if isinstance(data_source, (str, Path)):
            self.df = pd.read_csv(data_source).reset_index(drop=True)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        self.processed_root = (
            Path(processed_root).resolve() if processed_root else get_paths().processed_dir
        )
        self.transform = transform
        self.return_meta = return_meta

        # Canonical class-to-index mapping
        self.class_to_idx: Dict[str, int] = CLASS_TO_IDX
        self.classes = CLASS_NAMES

        # Validate class labels
        for cls_name in self.df["class_name"].unique():
            if cls_name not in self.class_to_idx:
                raise ValueError(f"Unrecognized class '{cls_name}' found in dataset manifest.")

        # Validate every crop path up front so non-portable manifests fail early.
        if "crop_path" not in self.df.columns:
            raise KeyError("Dataset manifest has no 'crop_path' column.")
        for crop_path in self.df["crop_path"]:
            resolve_crop_path(self.processed_root, crop_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict]]:
        row = self.df.iloc[index]

        img_path = resolve_crop_path(self.processed_root, row["crop_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Cropped image not found at: {img_path}")

        # Open image as 3-channel RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        class_name = str(row["class_name"])
        label = self.class_to_idx[class_name]

        if self.return_meta:
            meta = {
                "parent_filename": row.get("parent_filename", ""),
                "crop_filename": row.get("crop_filename", img_path.name),
                "class_name": class_name,
                "split": row.get("split", ""),
                "img_path": str(img_path),
            }
            return image, label, meta

        return image, label


def get_dataloaders(
    paths: Optional[ProjectPaths] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds reproducible PyTorch DataLoaders for train, validation, and test splits.

    Args:
        paths: ProjectPaths instance.
        batch_size: Mini-batch size.
        num_workers: Multiprocessing data loading workers.
        seed: Random seed for DataLoader worker initialization.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if paths is None:
        paths = get_paths()

    train_csv = paths.splits_dir / "train.csv"
    val_csv = paths.splits_dir / "val.csv"
    test_csv = paths.splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"Split manifests not found in {paths.splits_dir}. "
            "Please run prepare_dataset pipeline first."
        )

    # Datasets
    train_dataset = TrafficSignDataset(
        data_source=train_csv,
        processed_root=paths.processed_dir,
        transform=get_train_transforms(),
    )
    val_dataset = TrafficSignDataset(
        data_source=val_csv,
        processed_root=paths.processed_dir,
        transform=get_eval_transforms(),
    )
    test_dataset = TrafficSignDataset(
        data_source=test_csv,
        processed_root=paths.processed_dir,
        transform=get_eval_transforms(),
    )

    # Deterministic PyTorch DataLoader generator
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader
