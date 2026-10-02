"""Dataset preparation pipeline for Cambodian Traffic Signs Dataset (CamTSD).

Performs:
1. Parsing raw JSON annotations (CAM_TSR_v3_json.json).
2. Filtering for the 26 target traffic sign classes (excluding UNDEFINED).
3. Leak-free source-image level splitting (70% train, 15% validation, 15% test).
4. Cropping bounding boxes from source images into structured split directories.
5. Saving audit manifests in dataset/splits/.
6. Comprehensive integrity validation.
"""

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_SEED,
    NUM_CLASSES,
    ProjectPaths,
    RAW_TO_CANONICAL,
    TEST_SPLIT_RATIO,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    get_paths,
)


def load_raw_annotations(json_path: Path) -> Dict[str, Any]:
    """Loads VIA JSON annotations file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_and_filter_records(
    annotations: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Extracts valid 26-class bounding boxes from raw annotations.

    Excludes UNDEFINED annotations and non-target classes.
    """
    records: List[Dict[str, Any]] = []
    class_counter: Counter = Counter()

    for image_key, data in annotations.items():
        filename = data.get("filename", "")
        if not filename:
            continue

        regions = data.get("regions", [])
        for region_idx, region in enumerate(regions):
            region_attrs = region.get("region_attributes", {})
            raw_name = region_attrs.get("Name", "").strip()

            # Skip UNDEFINED or empty
            if not raw_name or raw_name == "UNDEFINED":
                continue

            # Check if class is in 26 target classes
            canonical_name = RAW_TO_CANONICAL.get(raw_name)
            if canonical_name is None or canonical_name not in CLASS_TO_IDX:
                continue

            shape_attrs = region.get("shape_attributes", {})
            shape_name = shape_attrs.get("name", "")
            if shape_name != "rect":
                # Handle bounding box coordinates
                continue

            x = int(shape_attrs.get("x", 0))
            y = int(shape_attrs.get("y", 0))
            width = int(shape_attrs.get("width", 0))
            height = int(shape_attrs.get("height", 0))

            if width <= 0 or height <= 0:
                continue

            record = {
                "parent_filename": filename,
                "region_index": region_idx,
                "raw_name": raw_name,
                "class_name": canonical_name,
                "class_idx": CLASS_TO_IDX[canonical_name],
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": width,
                "bbox_h": height,
            }
            records.append(record)
            class_counter[canonical_name] += 1

    return records, dict(class_counter)


def split_source_images(
    records: List[Dict[str, Any]],
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
) -> Dict[str, str]:
    """Splits unique source image filenames into train, val, and test partitions.

    Guarantees zero parent-image data leakage across splits.

    Returns:
        Mapping of {parent_filename: split_name ('train', 'val', 'test')}
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    ), "Split ratios must sum to 1.0"

    # Identify unique parent images containing target classes
    df = pd.DataFrame(records)
    unique_parents = sorted(df["parent_filename"].unique())

    # Create primary class per source image for stratified splitting
    parent_primary_class = (
        df.groupby("parent_filename")["class_name"].agg(lambda x: x.mode()[0]).to_dict()
    )
    parent_labels = [parent_primary_class[p] for p in unique_parents]

    # Handle rare classes that have only 1 image for stratification
    label_counts = Counter(parent_labels)
    can_stratify = all(count >= 2 for count in label_counts.values())

    # Stage 1: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_parents, temp_parents = train_test_split(
        unique_parents,
        test_size=temp_ratio,
        random_state=seed,
        stratify=parent_labels if can_stratify else None,
    )

    # Stage 2: Val vs Test (50/50 of Temp)
    temp_labels = (
        [parent_primary_class[p] for p in temp_parents] if can_stratify else None
    )
    temp_label_counts = Counter(temp_labels) if temp_labels else {}
    can_stratify_temp = (
        can_stratify and all(count >= 2 for count in temp_label_counts.values())
    )

    val_parents, test_parents = train_test_split(
        temp_parents,
        test_size=test_ratio / temp_ratio,
        random_state=seed,
        stratify=temp_labels if can_stratify_temp else None,
    )

    split_map: Dict[str, str] = {}
    for p in train_parents:
        split_map[p] = "train"
    for p in val_parents:
        split_map[p] = "val"
    for p in test_parents:
        split_map[p] = "test"

    return split_map


def crop_and_save_dataset(
    raw_images_dir: Path,
    records: List[Dict[str, Any]],
    split_map: Dict[str, str],
    processed_dir: Path,
    splits_dir: Path,
) -> pd.DataFrame:
    """Crops sign regions from source images and saves them into split/class folders.

    Layout:
      dataset/processed/<split>/<class_name>/<crop_id>.jpg
      dataset/splits/train.csv, val.csv, test.csv, train.txt, val.txt, test.txt

    Manifest crop_path values are portable POSIX paths relative to
    `processed_dir` (e.g. "train/KM_POST/<crop_id>.jpg"), with no
    machine-specific absolute paths, so manifests work on any OS.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_records: List[Dict[str, Any]] = []

    # Cache loaded parent images to avoid repeated I/O
    for rec in records:
        parent_fn = rec["parent_filename"]
        split = split_map.get(parent_fn)
        if not split:
            continue

        class_name = rec["class_name"]
        reg_idx = rec["region_index"]

        # Target directory: dataset/processed/<split>/<class_name>/
        target_dir = processed_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Unique crop filename incorporating parent and region
        crop_name = (
            f"{Path(parent_fn).stem}_crop{reg_idx:02d}_{rec['class_idx']:02d}.jpg"
        )
        crop_path = target_dir / crop_name

        # Perform image crop if source exists
        source_image_path = raw_images_dir / parent_fn
        if source_image_path.exists() and not crop_path.exists():
            image = Image.open(source_image_path).convert("RGB")
            x, y, w, h = (
                rec["bbox_x"],
                rec["bbox_y"],
                rec["bbox_w"],
                rec["bbox_h"],
            )

            # Clamp coordinates to image boundaries
            img_w, img_h = image.size
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            if x2 > x1 and y2 > y1:
                cropped = image.crop((x1, y1, x2, y2))
                cropped.save(crop_path, "JPEG", quality=95)

        processed_records.append(
            {
                "crop_path": crop_path.relative_to(processed_dir).as_posix(),
                "crop_filename": crop_name,
                "parent_filename": parent_fn,
                "split": split,
                "class_name": class_name,
                "class_idx": rec["class_idx"],
                "bbox_x": rec["bbox_x"],
                "bbox_y": rec["bbox_y"],
                "bbox_w": rec["bbox_w"],
                "bbox_h": rec["bbox_h"],
            }
        )

    all_df = pd.DataFrame(processed_records)

    # Save manifests per split
    for split_name in ["train", "val", "test"]:
        split_df = all_df[all_df["split"] == split_name].reset_index(drop=True)
        split_df.to_csv(splits_dir / f"{split_name}.csv", index=False, lineterminator="\n")

        # Also save simple line-by-line path list
        with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8", newline="\n") as f:
            for path_str in split_df["crop_path"]:
                f.write(f"{path_str}\n")

    return all_df


def validate_dataset_integrity(
    splits_dir: Path,
    processed_dir: Path,
) -> Dict[str, Any]:
    """Audits the prepared dataset against all 5 critical lecturer & anti-leakage requirements.

    Checks:
    1. Parent image overlap between train, val, and test is ZERO.
    2. Exactly 26 classes are present and match canonical names.
    3. UNDEFINED is absent.
    4. No accidental 'splits' class directory exists in class hierarchy.
    5. Train, val, and test splits are cleanly separated.
    6. Manifest crop paths are portable (relative POSIX, no absolute paths).
    """
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError("Split CSV manifests missing from splits directory.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_parents: Set[str] = set(train_df["parent_filename"])
    val_parents: Set[str] = set(val_df["parent_filename"])
    test_parents: Set[str] = set(test_df["parent_filename"])

    # Check 1: Zero parent overlap
    train_val_overlap = train_parents.intersection(val_parents)
    train_test_overlap = train_parents.intersection(test_parents)
    val_test_overlap = val_parents.intersection(test_parents)
    zero_overlap = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    # Check 2: Exactly 26 classes
    all_classes = set(train_df["class_name"]).union(
        set(val_df["class_name"]), set(test_df["class_name"])
    )
    exact_26_classes = all_classes == set(CLASS_NAMES) and len(all_classes) == 26

    # Check 3: UNDEFINED is absent
    undefined_absent = "UNDEFINED" not in all_classes

    # Check 4: No 'splits' directory treated as class
    no_splits_class = "splits" not in all_classes and "SPLITS" not in all_classes

    # Check 5: Separation of paths (crop_path is relative to processed_dir)
    train_paths_ok = all(p.startswith("train/") for p in train_df["crop_path"])
    val_paths_ok = all(p.startswith("val/") for p in val_df["crop_path"])
    test_paths_ok = all(p.startswith("test/") for p in test_df["crop_path"])

    # Check 6: Portable manifests (no backslashes, drive letters, absolute paths)
    all_dfs = (train_df, val_df, test_df)
    portable_paths = all(
        "\\" not in p and ":" not in p and not p.startswith("/")
        for df in all_dfs
        for p in df["crop_path"]
    ) and not any("absolute_crop_path" in df.columns for df in all_dfs)

    report = {
        "source_images_train": len(train_parents),
        "source_images_val": len(val_parents),
        "source_images_test": len(test_parents),
        "source_images_total": len(
            train_parents.union(val_parents).union(test_parents)
        ),
        "crops_train": len(train_df),
        "crops_val": len(val_df),
        "crops_test": len(test_df),
        "crops_total": len(train_df) + len(val_df) + len(test_df),
        "class_count": len(all_classes),
        "zero_parent_overlap": zero_overlap,
        "exact_26_classes": exact_26_classes,
        "undefined_absent": undefined_absent,
        "no_splits_class": no_splits_class,
        "path_separation_ok": train_paths_ok and val_paths_ok and test_paths_ok,
        "portable_paths": portable_paths,
    }

    # Save summary metadata
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    return report


def run_preparation_pipeline(
    paths: ProjectPaths = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the end-to-end dataset preparation and verification pipeline."""
    if paths is None:
        paths = get_paths()

    annotations = load_raw_annotations(paths.raw_annotations_path)
    records, raw_counts = parse_and_filter_records(annotations)
    split_map = split_source_images(records, seed=seed)

    all_df = crop_and_save_dataset(
        raw_images_dir=paths.raw_images_dir,
        records=records,
        split_map=split_map,
        processed_dir=paths.processed_dir,
        splits_dir=paths.splits_dir,
    )

    validation_report = validate_dataset_integrity(
        splits_dir=paths.splits_dir,
        processed_dir=paths.processed_dir,
    )

    return all_df, validation_report
