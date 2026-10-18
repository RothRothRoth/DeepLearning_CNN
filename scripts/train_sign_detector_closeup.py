"""Retrain the 1-class YOLO sign detector so it also finds LARGE / close-up signs (inference app unchanged).

Why: the first detector (scripts/train_sign_detector.py) only saw dashcam photos, where a sign is usually
under 1% of the frame. On close-ups (a sign filling 30-60% of the photo) it found ~13% of signs and often
boxed only the inside of a sign, so the classifier never saw the whole sign.

What this script adds to the SAME YOLO dataset (same train / val / test split by source photo):
  1. zoom close-ups : for every train / val photo with a sign, a crop around one sign so that it fills
                      25-70% of the frame; every box fully (or >= 60%) inside is kept, clipped to the crop.
  2. composites      : 1-4 annotated sign crops of the same split pasted large onto plain / coloured
                      backgrounds (like a sign photographed close or a printed sign on a wall).
Test photos are NEVER used for training. Training starts from the first detector's weights.

Evaluation (test split only), old vs new detector, each with the UNCHANGED ResNet-18 classifier:
  - normal test photos   : the same end-to-end test as the first detector (463 signs, 379 photos)
  - close-up test        : zoom close-ups made from TEST photos (fixed seed)
  - composite test       : composites made from TEST sign crops (fixed seed)

Usage (from the project root):
    python scripts/train_sign_detector_closeup.py              # build data, train, evaluate
    python scripts/train_sign_detector_closeup.py --eval-only  # evaluate a finished run
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sign_detector as base  # noqa: E402  (also applies the Windows-safe Ultralytics import)

PROJECT_ROOT = base.PROJECT_ROOT
CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES = base.CLASS_NAMES, base.IMAGE_SIZE, base.NUM_CLASSES


# ----------------------------------------------------------------------------------------------- data
def read_boxes(label_path, w, h):
    """YOLO label file -> [(x1, y1, x2, y2)] in pixels."""
    if not label_path.is_file():
        return []
    out = []
    for line in label_path.read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def write_labels(path, boxes, w, h):
    path.write_text("".join(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} "
                            f"{(y2 - y1) / h:.6f}\n" for x1, y1, x2, y2 in boxes))


def zoom_closeup(img, boxes, target, rng, out_width=None, min_visible=0.6):
    """Crop around boxes[target] so it fills 25-70% of the width; returns (image, kept boxes, kept indices)."""
    x1, y1, x2, y2 = boxes[target]
    frac = rng.uniform(0.25, 0.7)
    aspect = rng.choice([0.75, 1.0, 1.33, 1.78])                       # crop height / width
    cw = max(x2 - x1, (y2 - y1) / aspect) / frac
    ch = cw * aspect
    cx = (x1 + x2) / 2 + rng.uniform(-0.25, 0.25) * cw
    cy = (y1 + y2) / 2 + rng.uniform(-0.25, 0.25) * ch
    left, top = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    right, bottom = int(min(img.width, left + cw)), int(min(img.height, top + ch))
    crop = img.crop((left, top, right, bottom))
    s = (out_width or rng.randint(800, 1600)) / crop.width
    crop = crop.resize((round(crop.width * s), round(crop.height * s)))
    kept, idx = [], []
    for i, (a, b, c, d) in enumerate(boxes):
        ca, cb, cc, cd = max(a, left), max(b, top), min(c, right), min(d, bottom)
        if cc <= ca or cd <= cb:
            continue
        if (cc - ca) * (cd - cb) >= min_visible * (c - a) * (d - b):
            kept.append(((ca - left) * s, (cb - top) * s, (cc - left) * s, (cd - top) * s))
            idx.append(i)
    return crop, kept, idx


def background(w, h, rng):
    kind = rng.random()
    if kind < 0.45:                                                     # white / light grey (print, clip art)
        v = rng.randint(215, 255)
        return Image.new("RGB", (w, h), (v, v, v))
    if kind < 0.75:                                                     # any solid colour (wall, sky)
        return Image.new("RGB", (w, h), tuple(rng.randint(0, 255) for _ in range(3)))
    small = Image.new("RGB", (4, 4))                                    # smooth random gradient
    small.putdata([tuple(rng.randint(0, 255) for _ in range(3)) for _ in range(16)])
    return small.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(8))


def composite(crops, rng):
    """1-4 sign crops pasted large and non-overlapping; returns (image, boxes, class names)."""
    w = rng.randint(800, 1600)
    h = round(w * rng.choice([0.56, 0.75, 1.0, 1.33]))
    img = background(w, h, rng)
    boxes, names = [], []
    for _ in range(rng.randint(1, 4)):
        path, cname = rng.choice(crops)
        sign = Image.open(path).convert("RGB")
        side = rng.uniform(0.18, 0.6) * min(w, h)
        s = side / max(sign.size)
        sign = sign.resize((max(8, round(sign.width * s)), max(8, round(sign.height * s))))
        for _ in range(30):
            x, y = rng.randint(0, w - sign.width), rng.randint(0, h - sign.height)
            b = (x, y, x + sign.width, y + sign.height)
            if all(base.iou(b, o) == 0 for o in boxes):
                img.paste(sign, (x, y))
                boxes.append(b)
                names.append(cname)
                break
    return img, boxes, names


def processed_crops(split):
    root = PROJECT_ROOT / "dataset" / "processed" / split
    return [(p, p.parent.name) for p in sorted(root.glob("*/*.jpg"))]


def build_closeup_dataset(base_dir, out_dir, seed, n_composites):
    """Copies the first detector's YOLO dataset and adds close-ups + composites to train and val only."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.time()
    stats = {}
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
            for f in (base_dir / kind / split).iterdir():
                try:
                    os.link(f, out_dir / kind / split / f.name)          # hard link: no extra disk space
                except OSError:
                    shutil.copy(f, out_dir / kind / split / f.name)
        stats[split] = {"original_photos": len(list((base_dir / "images" / split).iterdir()))}
    for split in ("train", "val"):
        rng = random.Random(f"{seed}-{split}")
        n_zoom = 0
        for img_path in sorted((base_dir / "images" / split).iterdir()):
            img = Image.open(img_path).convert("RGB")
            boxes = read_boxes(base_dir / "labels" / split / (img_path.stem + ".txt"), img.width, img.height)
            if not boxes:
                continue
            crop, kept, _ = zoom_closeup(img, boxes, rng.randrange(len(boxes)), rng)
            name = f"zoom_{img_path.stem}"
            crop.save(out_dir / "images" / split / f"{name}.jpg", quality=92)
            write_labels(out_dir / "labels" / split / f"{name}.txt", kept, crop.width, crop.height)
            n_zoom += 1
        crops = processed_crops(split)
        n_comp = n_composites if split == "train" else n_composites // 5
        for i in range(n_comp):
            img, boxes, _ = composite(crops, rng)
            img.save(out_dir / "images" / split / f"comp_{i:05d}.jpg", quality=92)
            write_labels(out_dir / "labels" / split / f"comp_{i:05d}.txt", boxes, img.width, img.height)
        stats[split].update({"zoom_closeups": n_zoom, "composites": n_comp})
        print(f"  {split}: +{n_zoom} zoom close-ups, +{n_comp} composites ({time.time() - t0:.0f}s)", flush=True)
    stats["test"]["note"] = "test photos copied unchanged; no synthetic images in test"
    (out_dir / "data.yaml").write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\n"
                                       "test: images/test\nnames:\n  0: Traffic Sign\n")
    print(f"Close-up YOLO dataset built in {time.time() - t0:.0f}s: {json.dumps(stats)}")
    return stats


# ----------------------------------------------------------------------------------------- evaluation
def closeup_test_sets(raw_dir, splits_dir, seed=0, n_zoom=200, n_comp=200):
    """Fixed close-up test items from TEST photos / TEST crops: [(image, [(box, class_name or None)])]."""
    rng = random.Random(seed)
    df = pd.read_csv(splits_dir / "test.csv")
    by_photo = {p: g for p, g in df.groupby("parent_filename")}
    rows = list(df.itertuples())
    rng.shuffle(rows)
    zoom = []
    for r in rows[:n_zoom]:
        photo = Image.open(raw_dir / "images" / r.parent_filename).convert("RGB")
        g = by_photo[r.parent_filename]
        boxes = [(b.bbox_x, b.bbox_y, b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h) for b in g.itertuples()]
        names = list(g["class_name"])
        target = boxes.index((r.bbox_x, r.bbox_y, r.bbox_x + r.bbox_w, r.bbox_y + r.bbox_h))
        crop, kept, idx = zoom_closeup(photo, boxes, target, rng, out_width=1000, min_visible=1.0)
        zoom.append((crop, [(b, names[i]) for b, i in zip(kept, idx)]))
    crops = processed_crops("test")
    comp = []
    for _ in range(n_comp):
        img, boxes, names = composite(crops, rng)
        comp.append((img, list(zip(boxes, names))))
    return {"closeup_zoom_test": zoom, "composite_test": comp}


def score(detector, classifier, tf, items, imgsz, conf, device):
    torch_device = f"cuda:{device}" if isinstance(device, int) else device
    n = found = named = extra = 0
    for img, gts in items:
        dets = [tuple(map(float, b)) for b in
                detector.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes.xyxy.cpu()]
        used = set()
        for box, cname in gts:
            n += 1
            best, best_iou = None, 0.5
            for i, d in enumerate(dets):
                v = base.iou(box, d)
                if i not in used and v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                continue
            used.add(best)
            found += 1
            with torch.no_grad():
                x = tf(img.crop(tuple(int(v) for v in dets[best]))).unsqueeze(0).to(torch_device)
                named += CLASS_NAMES[int(classifier(x).argmax())] == cname
        extra += len(dets) - len(used)
    return {"signs": n, "found": found, "found_and_named": named, "extra_boxes": extra,
            "found_pct": round(100 * found / n, 2), "found_and_named_pct": round(100 * named / n, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset" / "raw")
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits")
    ap.add_argument("--work-dir", type=Path, default=Path(r"D:\DL_detector"))
    ap.add_argument("--classifier", type=Path, default=Path(r"D:\DL_checkpoint\resnet18_finetuned_tuned_best.pt"))
    ap.add_argument("--start-weights", type=Path, default=Path(r"D:\DL_checkpoint\sign_detector_best.pt"),
                    help="the first detector; also the 'old' detector in the comparison")
    ap.add_argument("--export-weights", type=Path, default=Path(r"D:\DL_checkpoint\sign_detector_closeup_best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--composites", type=int, default=1500, help="composites added to train (val gets 1/5)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from its last.pt")
    args = ap.parse_args()

    ultralytics, YOLO = base.import_ultralytics()
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found.")
    device = 0
    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    base_data = args.work_dir / "yolo_signs"
    data_dir = args.work_dir / "yolo_signs_closeup"
    tag = f"yolov8s_{args.imgsz}_closeup_seed{args.seed}"
    run_dir = args.work_dir / "runs" / tag
    for label, p in {"first YOLO dataset": base_data / "data.yaml", "start weights": args.start_weights,
                     "classifier": args.classifier}.items():
        if not p.exists():
            sys.exit(f"Missing {label}: {p}")
    os.chdir(args.work_dir)

    stats, minutes = "reused", None
    if args.resume:
        t0 = time.time()
        # Keeps the run's own settings; only workers / batch can be lowered (e.g. after running out of memory).
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True, workers=args.workers, batch=args.batch)
        minutes = round((time.time() - t0) / 60, 1)
        print(f"Resumed training finished in {minutes} min (this session only).")
    elif not args.eval_only:
        if run_dir.exists():
            sys.exit(f"{run_dir} exists; use --eval-only, or move it away first.")
        stats = build_closeup_dataset(base_data, data_dir, args.seed, args.composites)
        t0 = time.time()
        YOLO(str(args.start_weights)).train(data=str(data_dir / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs,
                                            batch=args.batch, seed=args.seed, deterministic=True, patience=10,
                                            workers=args.workers, device=device, project=str(run_dir.parent),
                                            name=tag, plots=True)
        minutes = round((time.time() - t0) / 60, 1)
        print(f"Training finished in {minutes} min.")

    new = YOLO(str(run_dir / "weights" / "best.pt"))
    old = YOLO(str(args.start_weights))
    m = new.val(data=str(base_data / "data.yaml"), split="test", imgsz=args.imgsz, batch=args.batch, device=device,
                project=str(run_dir), name="test_eval", exist_ok=True, plots=True)
    det = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
           "mAP50_95": float(m.box.map)}
    print("New detector, normal test photos:", json.dumps(det))
    e2e, summary = base.end_to_end(new, args.classifier, args.raw_dir, args.splits_dir, args.imgsz, args.conf, device)
    e2e.to_csv(run_dir / "end_to_end_test_predictions.csv", index=False)
    print("New detector, normal end-to-end:", json.dumps(summary))

    classifier = base.create_resnet18_finetuned(num_classes=NUM_CLASSES)
    base.load_checkpoint(args.classifier, model=classifier, map_location="cuda:0")
    classifier.to("cuda:0").eval()
    tf = base.get_eval_transforms(IMAGE_SIZE)
    sets = closeup_test_sets(args.raw_dir, args.splits_dir)
    comparison = {}
    for name, items in sets.items():
        comparison[name] = {"old_detector": score(old, classifier, tf, items, args.imgsz, args.conf, device),
                            "new_detector": score(new, classifier, tf, items, args.imgsz, args.conf, device)}
        print(name, json.dumps(comparison[name]))

    report = {
        "detector": {"start_weights": str(args.start_weights), "imgsz": args.imgsz, "epochs_requested": args.epochs,
                     "batch": args.batch, "seed": args.seed, "ultralytics_version": ultralytics.__version__,
                     "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                     "train_minutes_this_session": minutes, "best_weights": str(run_dir / "weights" / "best.pt")},
        "dataset": {"build": stats, "note": "test split = original test photos only"},
        "detector_test_metrics_normal_photos": det,
        "end_to_end_test_normal_photos": summary,
        "closeup_comparison_test_split": comparison,
        "classifier_checkpoint": str(args.classifier),
    }
    (run_dir / "closeup_detector_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics to {run_dir}")
    if args.export_weights.exists():
        print(f"NOT overwriting existing {args.export_weights}")
    else:
        shutil.copy(run_dir / "weights" / "best.pt", args.export_weights)
        print(f"Copied new detector weights to {args.export_weights}")


if __name__ == "__main__":
    main()
