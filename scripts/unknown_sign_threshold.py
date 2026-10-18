"""Choose and test an "unknown sign" confidence cut-off for the UNCHANGED final ResNet-18 (inference only).

The classifier knows 26 classes and always picks one. Rule: if its top-1 softmax confidence is below a cut-off t,
the web app shows "Unknown sign" instead of a class.

Data (annotated signs, cropped from the original photos with the annotation box):
  known   = the 26-class signs of a split (dataset/splits/<split>.csv)
  unknown = every other annotated sign in the same photos (STOP, ROAD WORK AHEAD, PARKING, ..., UNDEFINED);
            the classifier was never trained on any of them.

Selection (fixed before looking at test):
  known  : validation 26-class signs (454)   -- train crops are memorised, so their confidence is not typical
  unknown: train + validation unknown signs  -- never seen by the classifier, so train ones are fair here
  t in 0.30 ... 0.99 (step 0.01); pick the t with the highest balanced accuracy
  (known signs kept + unknown signs flagged) / 2; ties -> the lowest t.
Test (once): test 26-class signs (463) and test unknown signs (30).

usage: python scripts/unknown_sign_threshold.py --checkpoint path/to/resnet18_finetuned_tuned_best.pt
writes results/unknown_sign/: threshold.json, selection_sweep.csv, test_once.json, test_predictions.csv,
                              test_unknown_examples.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_NAMES, CLASS_TO_IDX, IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from src.data.prepare_dataset import RAW_TO_CANONICAL  # noqa: E402
from src.data.transforms import get_eval_transforms  # noqa: E402
from src.models import create_resnet18_finetuned  # noqa: E402
from src.training import get_device, load_checkpoint  # noqa: E402


def collect(split, annotations, splits_dir):
    """[(photo, (x, y, w, h), true_class or None, raw_name)] for every annotated sign in the split's photos."""
    known = pd.read_csv(splits_dir / f"{split}.csv")
    photos = set(known["parent_filename"])
    items = [(r.parent_filename, (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h), r.class_name, r.class_name)
             for r in known.itertuples()]
    for d in annotations.values():
        if d.get("filename") not in photos:
            continue
        for reg in d.get("regions", []):
            name = reg.get("region_attributes", {}).get("Name", "").strip()
            canon = RAW_TO_CANONICAL.get(name)
            shape = reg.get("shape_attributes", {})
            if (canon is not None and canon in CLASS_TO_IDX) or shape.get("name") != "rect":
                continue
            w, h = int(shape.get("width", 0)), int(shape.get("height", 0))
            if w > 0 and h > 0:
                items.append((d["filename"], (int(shape["x"]), int(shape["y"]), w, h), None, name or "(empty)"))
    return items


@torch.no_grad()
def predict(items, model, tf, device, raw_dir):
    out, cache = [], {}
    for photo, (x, y, w, h), true, raw in items:
        if photo not in cache:
            cache.clear()
            cache[photo] = Image.open(raw_dir / photo).convert("RGB")
        crop = cache[photo].crop((x, y, x + w, y + h))
        p = torch.softmax(model(tf(crop).unsqueeze(0).to(device)), 1)[0].cpu()
        out.append({"photo": photo, "box": f"{x},{y},{w},{h}", "true": true, "raw_name": raw,
                    "pred": CLASS_NAMES[int(p.argmax())], "conf": float(p.max())})
    return pd.DataFrame(out)


def rates(df, t):
    known, unknown = df[df["true"].notna()], df[df["true"].isna()]
    kept = known["conf"] >= t
    return {
        "known_signs": len(known), "unknown_signs": len(unknown),
        "known_kept": float(kept.mean()),
        "known_kept_and_correct": float((kept & (known["pred"] == known["true"])).mean()),
        "known_correct_without_cutoff": float((known["pred"] == known["true"]).mean()),
        "unknown_flagged": float((unknown["conf"] < t).mean()),
        "balanced_accuracy": float((kept.mean() + (unknown["conf"] < t).mean()) / 2),
    }


def montage(df, t, path):
    """Test unknown signs with their best 26-class guess, flagged (green) or not (red)."""
    rows = df[df["true"].isna()].sort_values("conf").reset_index(drop=True)
    tile, cols = 220, 6
    font = ImageFont.truetype("arial.ttf", 16) if Path("C:/Windows/Fonts/arial.ttf").exists() else None
    sheet = Image.new("RGB", (cols * tile, -(-len(rows) // cols) * (tile + 44)), "white")
    for i, r in rows.iterrows():
        x, y, w, h = map(int, r["box"].split(","))
        crop = Image.open(PROJECT_ROOT / "dataset" / "raw" / "images" / r["photo"]).convert("RGB")
        crop = crop.crop((x, y, x + w, y + h))
        crop.thumbnail((tile - 10, tile - 10))
        ox, oy = (i % cols) * tile, (i // cols) * (tile + 44)
        sheet.paste(crop, (ox + (tile - crop.width) // 2, oy + 5))
        flagged = r["conf"] < t
        d = ImageDraw.Draw(sheet)
        d.rectangle([ox + 2, oy + 2, ox + tile - 3, oy + tile + 40], outline="#238B5A" if flagged else "#C0392B",
                    width=4)
        d.text((ox + 8, oy + tile - 2), r["raw_name"].title()[:22], fill="black", font=font)
        d.text((ox + 8, oy + tile + 18), "flagged unknown" if flagged else
               f"called {r['pred'].replace('_', ' ').title()[:13]} {100 * r['conf']:.0f}%",
               fill="#238B5A" if flagged else "#C0392B", font=font)
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset" / "raw")
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "unknown_sign")
    args = ap.parse_args()

    device = get_device()
    model = create_resnet18_finetuned(num_classes=NUM_CLASSES)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.to(device).eval()
    tf = get_eval_transforms(IMAGE_SIZE)
    ann = json.loads((args.raw_dir / "data" / "CAM_TSR_v3_json.json").read_text(encoding="utf-8"))
    images = args.raw_dir / "images"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val = predict(collect("val", ann, args.splits_dir), model, tf, device, images)
    train_unknown = predict([i for i in collect("train", ann, args.splits_dir) if i[2] is None], model, tf, device,
                            images)
    selection = pd.concat([val, train_unknown], ignore_index=True)
    sweep = pd.DataFrame([{"threshold": round(t / 100, 2), **rates(selection, t / 100)} for t in range(30, 100)])
    sweep.to_csv(args.out_dir / "selection_sweep.csv", index=False)
    best = sweep.sort_values(["balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]
    t = float(best["threshold"])
    choice = {"threshold": t, "rule": "max balanced accuracy on selection set; ties -> lowest threshold",
              "selection_set": "validation 26-class signs + train and validation non-26-class signs",
              "selection_rates": rates(selection, t), "rates_without_cutoff_on_selection": rates(selection, 0.0)}
    (args.out_dir / "threshold.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print("Chosen on selection set:", json.dumps(choice, indent=2))

    test = predict(collect("test", ann, args.splits_dir), model, tf, device, images)
    test.to_csv(args.out_dir / "test_predictions.csv", index=False)
    no_undef = test[test["raw_name"] != "UNDEFINED"]
    report = {"threshold": t, "test": rates(test, t), "test_excluding_UNDEFINED_unknowns": rates(no_undef, t),
              "test_without_cutoff": rates(test, 0.0)}
    (args.out_dir / "test_once.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Test, evaluated once:", json.dumps(report, indent=2))
    montage(test, t, args.out_dir / "test_unknown_examples.jpg")


if __name__ == "__main__":
    main()
