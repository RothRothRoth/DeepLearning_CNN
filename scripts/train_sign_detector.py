"""Train and evaluate a 1-class YOLO traffic-sign detector on the local GPU (local version of notebook 09).

Two-stage pipeline: the detector (trained here) finds every sign in a street photo, then the UNCHANGED final
fine-tuned ResNet-18 classifies each detected box into one of the 26 classes.

The detector uses the SAME train / val / test split by source photo as the classifier (dataset/splits/*.csv).
Raw photos that contain no sign of the 26 classes are not in the manifests and go to train only, so the test
photos are exactly the classifier's test photos. Nothing in src/, results/, dataset/ or the classifier changes.

Usage (from the project root):
    python scripts/train_sign_detector.py --smoke          # 1-epoch check on a small subset
    python scripts/train_sign_detector.py                  # full training + test evaluation
    python scripts/train_sign_detector.py --resume         # continue an interrupted full run
    python scripts/train_sign_detector.py --eval-only      # re-run the test evaluation for a finished run
"""

import argparse
import json
import os
import pathlib
import random
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from src.data.transforms import get_eval_transforms  # noqa: E402
from src.models import create_resnet18_finetuned  # noqa: E402
from src.training import load_checkpoint  # noqa: E402


def _import_ultralytics_safely():
    """Imports ultralytics despite a Windows quirk.

    With Microsoft Store Python, site-packages lives under D:\\WpSystem. At import time Ultralytics looks for a
    .git folder in every parent folder, and checking the protected D:\\WpSystem\\.git raises WinError 1337.
    During the import only, such an OSError is treated as "does not exist"; normal behaviour is restored after.
    This runs at module level, so Windows DataLoader worker processes (which re-import this script before
    loading any Ultralytics objects) get the same treatment.
    """
    original_exists = pathlib.Path.exists

    def safe_exists(self, *args, **kwargs):
        try:
            return original_exists(self, *args, **kwargs)
        except OSError:
            return False

    pathlib.Path.exists = safe_exists
    try:
        import ultralytics  # noqa: F401
        import ultralytics.data.build  # noqa: F401  (modules the DataLoader workers need)
    finally:
        pathlib.Path.exists = original_exists


_import_ultralytics_safely()


def import_ultralytics():
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR

    SETTINGS.update({"sync": False})  # do not send anonymous usage statistics to Ultralytics
    # Plot labels need Arial.ttf; use the copy that ships with Windows instead of downloading it.
    windows_arial = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
    if not (USER_CONFIG_DIR / "Arial.ttf").exists() and windows_arial.is_file():
        shutil.copy(windows_arial, USER_CONFIG_DIR / "Arial.ttf")
    return ultralytics, YOLO


def build_dataset(raw_dir, splits_dir, out_dir, max_width=1920, subset=None, seed=42):
    """Writes the YOLO dataset (images resized to max_width, labels copied) using the classifier's split."""
    splits = {s: pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")}
    parent_split = {}
    for s, df in splits.items():
        for p in df["parent_filename"].unique():
            assert parent_split.setdefault(p, s) == s, f"{p} appears in two splits"
    images = sorted(p.name for p in (raw_dir / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    missing = [p for p in parent_split if not (raw_dir / "images" / p).is_file()]
    assert not missing, f"manifest photos missing: {missing[:5]}"
    if subset:  # smoke test: a few photos per split
        rng = random.Random(seed)
        by_split = {s: [n for n in images if parent_split.get(n, "train") == s] for s in ("train", "val", "test")}
        images = sorted(sum((rng.sample(v, min(subset, len(v))) for v in by_split.values()), []))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = {s: {"images": 0, "boxes": 0, "no_label_file": 0} for s in ("train", "val", "test")}
    t0 = time.time()
    for i, name in enumerate(images, 1):
        split = parent_split.get(name, "train")
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        img = Image.open(raw_dir / "images" / name).convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)))
        img.save(out_dir / "images" / split / name, quality=95)
        label = raw_dir / "labels" / (Path(name).stem + ".txt")
        if label.is_file():
            shutil.copy(label, out_dir / "labels" / split / label.name)
            stats[split]["boxes"] += sum(1 for line in label.read_text().splitlines() if line.strip())
        else:
            stats[split]["no_label_file"] += 1  # treated by YOLO as a photo without signs
        stats[split]["images"] += 1
        if i % 500 == 0:
            print(f"  built {i}/{len(images)} photos ({time.time() - t0:.0f}s)", flush=True)
    if not subset:
        test_parents = {p for p, s in parent_split.items() if s == "test"}
        assert test_parents == {p.name for p in (out_dir / "images" / "test").iterdir()}, "test split mismatch"
        print("Test photos are exactly the classifier's test photos.")
    (out_dir / "data.yaml").write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\n"
                                       "test: images/test\nnames:\n  0: Traffic Sign\n")
    print(f"YOLO dataset built in {time.time() - t0:.0f}s: {json.dumps(stats)}")
    return stats, parent_split


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def end_to_end(detector, classifier_ckpt, raw_dir, splits_dir, imgsz, conf, device, photos=None):
    """Detect -> crop from the full-resolution photo -> classify with the unchanged ResNet-18, on test photos.

    Every 26-class test sign (test.csv) is matched to the unused detection with the highest IoU (>= 0.5)."""
    torch_device = f"cuda:{device}" if isinstance(device, int) else device  # Ultralytics uses 0, torch "cuda:0"
    classifier = create_resnet18_finetuned(num_classes=NUM_CLASSES)
    load_checkpoint(classifier_ckpt, model=classifier, map_location=torch_device)
    classifier.to(torch_device).eval()
    tf = get_eval_transforms(IMAGE_SIZE)
    test_df = pd.read_csv(splits_dir / "test.csv")
    if photos is not None:
        test_df = test_df[test_df["parent_filename"].isin(photos)]
    rows, unmatched, n_dets = [], 0, 0
    for parent, gt in test_df.groupby("parent_filename"):
        photo = Image.open(raw_dir / "images" / parent).convert("RGB")
        boxes = detector.predict(photo, imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes
        dets = [list(map(float, b)) for b in boxes.xyxy.cpu()]
        n_dets += len(dets)
        preds = []
        with torch.no_grad():
            for x1, y1, x2, y2 in dets:
                p = torch.softmax(classifier(tf(photo.crop((int(x1), int(y1), int(x2), int(y2))))
                                             .unsqueeze(0).to(torch_device)), 1)[0].cpu()
                preds.append((CLASS_NAMES[int(p.argmax())], float(p.max())))
        used = set()
        for _, g in gt.iterrows():
            gbox = (g["bbox_x"], g["bbox_y"], g["bbox_x"] + g["bbox_w"], g["bbox_y"] + g["bbox_h"])
            best, best_iou = None, 0.5
            for i, dbox in enumerate(dets):
                v = iou(gbox, dbox)
                if i not in used and v >= best_iou:
                    best, best_iou = i, v
            if best is not None:
                used.add(best)
            name, pconf = preds[best] if best is not None else (None, None)
            rows.append({"parent": parent, "crop_path": g["crop_path"], "true": g["class_name"],
                         "detected": best is not None, "iou": round(best_iou, 4) if best is not None else 0.0,
                         "pred": name, "conf": pconf, "correct": name == g["class_name"]})
        unmatched += len(dets) - len(used)
    e2e = pd.DataFrame(rows)
    summary = {
        "test_photos": int(test_df["parent_filename"].nunique()),
        "test_signs_26_classes": len(e2e),
        "detections_total": n_dets,
        "detection_recall_26_classes": float(e2e["detected"].mean()),
        "end_to_end_accuracy": float(e2e["correct"].mean()),
        "classifier_accuracy_on_detected_signs": float(e2e.loc[e2e["detected"], "correct"].mean()),
        "unmatched_detections": int(unmatched),
        "detector_conf_threshold": conf,
        "note": "unmatched detections are false positives OR real signs outside the 26 classes (e.g. UNDEFINED)",
    }
    return e2e, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset" / "raw")
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits")
    ap.add_argument("--work-dir", type=Path, default=Path(r"D:\DL_detector"))
    ap.add_argument("--classifier", type=Path, default=Path(r"D:\DL_checkpoint\resnet18_finetuned_tuned_best.pt"))
    ap.add_argument("--export-weights", type=Path, default=Path(r"D:\DL_checkpoint\sign_detector_best.pt"))
    ap.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained starting weights")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold for the pipeline test")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 40 photos per split, separate output folder")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the YOLO dataset even if it exists")
    ap.add_argument("--resume", action="store_true", help="resume the interrupted full run")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate the finished run")
    args = ap.parse_args()

    ultralytics, YOLO = import_ultralytics()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"device {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    if device != 0:
        sys.exit("No CUDA GPU found: training on CPU would take days.")
    for label, p in {"raw photos": args.raw_dir / "images", "raw YOLO labels": args.raw_dir / "labels",
                     "split manifests": args.splits_dir, "classifier checkpoint": args.classifier}.items():
        if not p.exists():
            sys.exit(f"Missing {label}: {p}")

    tag = "smoke" if args.smoke else f"yolov8s_{args.imgsz}_seed{args.seed}"
    data_dir = args.work_dir / ("yolo_signs_smoke" if args.smoke else "yolo_signs")
    run_dir = args.work_dir / "runs" / tag
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)  # Ultralytics downloads starting weights / fonts into the working directory

    if args.rebuild or args.smoke or not (data_dir / "data.yaml").exists():
        stats, _ = build_dataset(args.raw_dir, args.splits_dir, data_dir, subset=40 if args.smoke else None,
                                 seed=args.seed)
    else:
        stats = "reused existing dataset at " + str(data_dir)
        print(f"Reusing YOLO dataset at {data_dir} (use --rebuild to rebuild)")

    best = run_dir / "weights" / "best.pt"
    minutes = None
    if not args.eval_only:
        t0 = time.time()
        if args.resume:
            YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
        else:
            if run_dir.exists() and not args.smoke:
                sys.exit(f"{run_dir} exists. Use --resume to continue it, or --eval-only to evaluate it.")
            YOLO(args.model).train(data=str(data_dir / "data.yaml"), imgsz=args.imgsz,
                                   epochs=1 if args.smoke else args.epochs, batch=args.batch, seed=args.seed,
                                   deterministic=True, patience=15, workers=args.workers, device=device,
                                   project=str(run_dir.parent), name=tag, exist_ok=args.smoke, plots=True)
        minutes = round((time.time() - t0) / 60, 1)
        print(f"Training finished in {minutes} min. Best weights: {best}")

    detector = YOLO(str(best))
    m = detector.val(data=str(data_dir / "data.yaml"), split="test", imgsz=args.imgsz, batch=args.batch,
                     device=device, project=str(run_dir), name="test_eval", exist_ok=True, plots=True)
    det = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
           "mAP50_95": float(m.box.map)}
    print("Detector on test photos:", json.dumps(det))

    photos = {p.name for p in (data_dir / "images" / "test").iterdir()} if args.smoke else None
    e2e, summary = end_to_end(detector, args.classifier, args.raw_dir, args.splits_dir, args.imgsz, args.conf,
                              device, photos)
    print("End-to-end pipeline on test photos:", json.dumps(summary, indent=2))

    e2e.to_csv(run_dir / "end_to_end_test_predictions.csv", index=False)
    report = {
        "smoke_test": args.smoke,
        "detector": {"start_weights": args.model, "imgsz": args.imgsz, "epochs_requested": args.epochs,
                     "batch": args.batch, "seed": args.seed, "ultralytics_version": ultralytics.__version__,
                     "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                     "train_minutes_this_session": minutes, "best_weights": str(best)},
        "dataset": {"split_source": "dataset/splits/*.csv (same split as the classifier)", "build": stats},
        "detector_test_metrics": det,
        "end_to_end_test": summary,
        "classifier_checkpoint": str(args.classifier),
    }
    (run_dir / "detector_and_pipeline_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics to {run_dir}")

    if not args.smoke:
        if args.export_weights.exists():
            print(f"NOT overwriting existing {args.export_weights}; copy {best} there yourself if wanted.")
        else:
            shutil.copy(best, args.export_weights)
            print(f"Copied detector weights to {args.export_weights}")


if __name__ == "__main__":
    main()
