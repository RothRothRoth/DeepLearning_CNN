"""Configuration settings for Cambodian Road Sign Recognition (CamTSD).

Defines canonical class names, dataset paths, training parameters,
and normalization statistics for PyTorch workflows.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

# Reproducibility
DEFAULT_SEED: int = 42

# Image and Batch Configurations
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
NUM_WORKERS: int = 2

# ImageNet Normalization Constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Split Ratios
TRAIN_SPLIT_RATIO: float = 0.70
VAL_SPLIT_RATIO: float = 0.15
TEST_SPLIT_RATIO: float = 0.15

# 26 Canonical Selected Classes (Sorted alphabetically)
CLASS_NAMES: List[str] = [
    "30_SPEED_LIMIT",
    "40_SPEED_LIMIT",
    "60_SPEED_LIMIT",
    "80_SPEED_LIMIT",
    "CARRIAGE_WAY_NARROWS",
    "CHILDREN_CROSSING",
    "CROSS_ROAD",
    "DIRECTION",
    "END_SPEED_LIMIT",
    "HOSPITAL",
    "KEEP_RIGHT",
    "KM_POST",
    "LEFT_BEND",
    "NO_ENTRY",
    "NO_HORN",
    "NO_PARKING",
    "NO_UTURN",
    "PEDESTRAIN_CROSSING",
    "PEDESTRAIN_CR_AREA",
    "PRIORITY_ROAD",
    "RIGHT_BEND",
    "ROAD_JUNCTION_ON_THE_LEFT",
    "ROAD_JUNCTION_ON_THE_RIGHT",
    "SLOW_DOWN",
    "STAGGERED_JUNCTION_RL",
    "U_TURN",
]

NUM_CLASSES: int = len(CLASS_NAMES)
assert NUM_CLASSES == 26, f"Expected 26 classes, got {NUM_CLASSES}"

# Mapping from raw JSON annotation names in CAM_TSR_v3_json.json to canonical class names
RAW_TO_CANONICAL: Dict[str, str] = {
    "30 SPEED LIMIT": "30_SPEED_LIMIT",
    "40 SPEED LIMIT": "40_SPEED_LIMIT",
    "60 SPEED LIMIT": "60_SPEED_LIMIT",
    "80 SPEED LIMIT": "80_SPEED_LIMIT",
    "CARRIAGE WAY NARROWS": "CARRIAGE_WAY_NARROWS",
    "CHILDREN CROSSING": "CHILDREN_CROSSING",
    "CROSS ROAD": "CROSS_ROAD",
    "DIRECTION": "DIRECTION",
    "END SPEED LIMIT": "END_SPEED_LIMIT",
    "HOSPITAL": "HOSPITAL",
    "KEEP RIGHT": "KEEP_RIGHT",
    "KM POST": "KM_POST",
    "LEFT BEND": "LEFT_BEND",
    "NO ENTRY": "NO_ENTRY",
    "NO HORN": "NO_HORN",
    "NO PARKING": "NO_PARKING",
    "NO UTURN": "NO_UTURN",
    "PEDESTRAIN CROSSING": "PEDESTRAIN_CROSSING",
    "PEDESTRAIN CR AREA": "PEDESTRAIN_CR_AREA",
    "PRIORITY ROAD": "PRIORITY_ROAD",
    "RIGHT BEND": "RIGHT_BEND",
    "ROAD JUNCTION ON THE LEFT": "ROAD_JUNCTION_ON_THE_LEFT",
    "ROAD JUNCTION ON THE RIGHT": "ROAD_JUNCTION_ON_THE_RIGHT",
    "SLOW DOWN": "SLOW_DOWN",
    "STAGGERED JUNCTION, RL": "STAGGERED_JUNCTION_RL",
    "STAGGERED JUNCTION RL": "STAGGERED_JUNCTION_RL",
    "U TURN": "U_TURN",
}

# Numerical class index mappings
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class ProjectPaths:
    """Encapsulates all workspace and dataset paths with dynamic environment support."""

    root_dir: Path
    dataset_dir: Path
    raw_dir: Path
    raw_images_dir: Path
    raw_annotations_path: Path
    processed_dir: Path
    splits_dir: Path
    models_dir: Path
    results_dir: Path


def get_paths(custom_base: str = None) -> ProjectPaths:
    """Resolves project paths dynamically for local or Google Colab environments.

    Priority:
    1. custom_base argument if provided.
    2. ROAD_SIGN_PROJECT_DIR environment variable if set.
    3. Google Colab / Google Drive path if /content/drive/MyDrive/Road_Sign_Recognition_CNN exists.
    4. Current repository root.
    """
    if custom_base:
        root = Path(custom_base).resolve()
    elif os.environ.get("ROAD_SIGN_PROJECT_DIR"):
        root = Path(os.environ["ROAD_SIGN_PROJECT_DIR"]).resolve()
    elif os.path.exists("/content/drive/MyDrive/Road_Sign_Recognition_CNN"):
        root = Path("/content/drive/MyDrive/Road_Sign_Recognition_CNN").resolve()
    else:
        # Default to repository root
        root = Path(__file__).resolve().parent.parent

    dataset_dir = root / "dataset"
    raw_dir = dataset_dir / "raw"
    raw_images_dir = raw_dir / "images"
    raw_annotations_path = raw_dir / "data" / "CAM_TSR_v3_json.json"

    # Processed crops separated by split to prevent namespace collisions
    processed_dir = dataset_dir / "processed"
    splits_dir = dataset_dir / "splits"
    models_dir = root / "models"
    results_dir = root / "results"

    return ProjectPaths(
        root_dir=root,
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        raw_images_dir=raw_images_dir,
        raw_annotations_path=raw_annotations_path,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        models_dir=models_dir,
        results_dir=results_dir,
    )
