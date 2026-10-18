"""Leakage check for the grouped folds: sign crop + surrounding scene similarity across train / val / test.

For every crop, a context window (3x the sign box, square, clipped to the photo) is embedded with ImageNet
ResNet-18 features. Calibration:
  same physical sign   = same class, same GoPro recording and chapter, frames < 2 s apart
  reference (unlinked) = same class, different drive clusters
Flag threshold (fixed rule) = 10th percentile of the same-sign similarities.
For each outer fold, every val / test crop whose most similar SAME-CLASS training crop is above the threshold is
reported as a POSSIBLE match (it can also be a different sign in a similar-looking place). A contact sheet of the
strongest test matches is saved for visual inspection.

Outputs: D:\\DL_phase3\\leakage\\  (features, contact sheet) and results/grouped_evaluation/leakage_check.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GX = re.compile(r"^GX(\d{2})(\d{4})\.MP4_snapshot_(\d+)\.(\d+)\.(\d+)", re.I)


@torch.no_grad()
def context_features(crops, raw_images, device, scale=3.0):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats, batch, cache = [], [], {}
    for i, r in enumerate(crops.itertuples()):
        if r.parent_filename not in cache:
            cache = {r.parent_filename: Image.open(raw_images / r.parent_filename).convert("RGB")}
        photo = cache[r.parent_filename]
        side = scale * max(r.bbox_w, r.bbox_h)
        cx, cy = r.bbox_x + r.bbox_w / 2, r.bbox_y + r.bbox_h / 2
        box = (max(0, cx - side / 2), max(0, cy - side / 2), min(photo.width, cx + side / 2),
               min(photo.height, cy + side / 2))
        batch.append(tf(photo.crop(tuple(int(v) for v in box))))
        if len(batch) == 64 or i == len(crops) - 1:
            feats.append(torch.nn.functional.normalize(net(torch.stack(batch).to(device)), dim=1).cpu())
            batch = []
    return torch.cat(feats).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grouped-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits_grouped")
    ap.add_argument("--raw-images", type=Path, default=PROJECT_ROOT / "dataset" / "raw" / "images")
    ap.add_argument("--work-dir", type=Path, default=Path(r"D:\DL_phase3\leakage"))
    ap.add_argument("--report", type=Path, default=PROJECT_ROOT / "results" / "grouped_evaluation" / "leakage_check.json")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    crops = pd.concat([pd.read_csv(args.grouped_dir / "fold_0" / f"{r}.csv") for r in ("train", "val", "test")],
                      ignore_index=True).sort_values("parent_filename").reset_index(drop=True)
    feat_file = args.work_dir / "context_features.npy"
    order_file = args.work_dir / "context_features_order.json"
    if feat_file.exists() and json.loads(order_file.read_text()) == crops.crop_path.tolist():
        F = np.load(feat_file)
    else:
        F = context_features(crops, args.raw_images, "cuda" if torch.cuda.is_available() else "cpu")
        np.save(feat_file, F)
        order_file.write_text(json.dumps(crops.crop_path.tolist()))
    S = F @ F.T

    # ---- calibration ----
    m = crops.parent_filename.str.extract(GX)
    crops["rec"] = m[1]
    crops["chap"] = m[0]
    crops["t"] = m[2].astype(float) * 60 + m[3].astype(float) + m[4].astype(float) / 1000
    same, ref = [], []
    for cls, g in crops.groupby("class_name"):
        idx = g.index.values
        for a_i, a in enumerate(idx):
            for b in idx[a_i + 1:]:
                ra, rb = crops.loc[a], crops.loc[b]
                if pd.notna(ra.rec) and ra.rec == rb.rec and ra.chap == rb.chap and abs(ra.t - rb.t) < 2:
                    same.append(S[a, b])
                elif ra.cluster != rb.cluster:
                    ref.append(S[a, b])
    same, ref = np.array(same), np.array(ref)
    thr = float(np.percentile(same, 10))
    calib = {"same_sign_pairs": len(same), "same_sign_median": float(np.median(same)),
             "reference_pairs": len(ref), "reference_median": float(np.median(ref)),
             "reference_p99": float(np.percentile(ref, 99)), "reference_p999": float(np.percentile(ref, 99.9)),
             "threshold_p10_same_sign": thr,
             "reference_pairs_above_threshold_pct": float((ref >= thr).mean() * 100)}
    print(json.dumps(calib, indent=1))

    # ---- per outer fold ----
    folds = json.loads((args.grouped_dir / "summary.json").read_text())["folds"]
    key = dict(zip(crops.crop_path, crops.index))
    report = {"design": __doc__, "calibration": calib, "folds": []}
    strongest = []
    for f in folds:
        k = f["outer_fold"]
        parts = {r: pd.read_csv(args.grouped_dir / f"fold_{k}" / f"{r}.csv") for r in ("train", "val", "test")}
        tr = parts["train"]
        res = {"outer_fold": k}
        for role in ("val", "test"):
            flagged = []
            for r in parts[role].itertuples():
                cand = tr[tr.class_name == r.class_name].crop_path.map(key).values
                if len(cand) == 0:
                    continue
                sims = S[key[r.crop_path], cand]
                j = int(sims.argmax())
                if sims[j] >= thr:
                    flagged.append((r.crop_path, tr.crop_path.iloc[0] if False else crops.crop_path[cand[j]],
                                    float(sims[j])))
            res[f"{role}_crops"] = len(parts[role])
            res[f"{role}_possible_matches"] = len(flagged)
            res[f"{role}_possible_match_pct"] = round(100 * len(flagged) / len(parts[role]), 2)
            if role == "test":
                strongest += [(k, *x) for x in flagged]
                res["test_flagged_examples"] = sorted(flagged, key=lambda x: -x[2])[:10]
        report["folds"].append(res)
        print(f"fold {k}: val possible matches {res['val_possible_matches']}/{res['val_crops']}, "
              f"test possible matches {res['test_possible_matches']}/{res['test_crops']}")
    args.report.write_text(json.dumps(report, indent=1))

    # ---- contact sheet of the 24 strongest test matches (test context | train context) ----
    strongest.sort(key=lambda x: -x[3])
    tiles = []
    for k, a, b, s in strongest[:24]:
        pair = []
        for cp in (a, b):
            r = crops.loc[key[cp]]
            photo = Image.open(args.raw_images / r.parent_filename).convert("RGB")
            side = 3 * max(r.bbox_w, r.bbox_h)
            cx, cy = r.bbox_x + r.bbox_w / 2, r.bbox_y + r.bbox_h / 2
            pair.append(photo.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2),
                                    int(cy + side / 2))).resize((140, 140)))
        tiles.append((pair, f"f{k} {s:.3f}"))
    sheet = Image.new("RGB", (4 * 300, (len(tiles) // 4 + 1) * 165), "white")
    d = ImageDraw.Draw(sheet)
    for i, ((ta, tb), cap) in enumerate(tiles):
        x, y = (i % 4) * 300, (i // 4) * 165
        sheet.paste(ta, (x, y))
        sheet.paste(tb, (x + 145, y))
        d.text((x + 4, y + 145), cap + "  test | train", fill="black")
    sheet.save(args.work_dir / "strongest_test_matches.jpg", quality=85)
    print("saved", args.report, "and", args.work_dir / "strongest_test_matches.jpg")


if __name__ == "__main__":
    main()
