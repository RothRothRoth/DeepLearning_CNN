"""Read-only analysis of the ORIGINAL random split (dataset/splits): temporal near-duplicates of test crops.

Definition (time-based, from file names only):
  A test crop counts as a "near-duplicate" if it comes from a GoPro video frame
  (GX<chapter><recording>.MP4_snapshot_<min>.<sec>.<ms>) and the TRAINING split contains a crop of the SAME CLASS
  from the SAME video file whose frame time is at most --gap seconds away (default 2 s).
  Such a pair most likely shows the same physical sign a moment earlier or later. This does not prove that the
  model memorised it; it only means the two images are not independent.
NR*_*.jpg and photo_* crops have no frame time in their names, so they are reported as "timing unknown".

Outputs (new files): results/grouped_evaluation/original_split_near_duplicates.csv (one row per test crop)
                     results/grouped_evaluation/original_split_near_duplicates_summary.json
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GX = re.compile(r"(GX\d{6}\.MP4)_snapshot_(\d+)\.(\d+)\.(\d+)", re.I)


def frame(path):
    m = GX.search(path)
    return (m.group(1).upper(), int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000) if m else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits")
    ap.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results" / "hyperparameter_tuning" /
                    "resnet18_finetuned_lr_wd_seed42" / "final_tuned_model" / "test_predictions.csv")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "grouped_evaluation" /
                    "original_split_near_duplicates.csv")
    ap.add_argument("--gap", type=float, default=2.0)
    args = ap.parse_args()

    train = pd.read_csv(args.splits_dir / "train.csv")
    test = pd.read_csv(args.splits_dir / "test.csv")
    for d in (train, test):
        d[["video_file", "frame_time_s"]] = d.crop_path.apply(lambda p: pd.Series(frame(p)))
    rows = []
    for r in test.itertuples():
        is_video = pd.notna(r.video_file)
        src = "gopro_video_frame" if is_video else ("nr_photo" if Path(r.crop_path).name.upper().startswith("NR")
                                                    else "phone_photo")
        near, gap = None, None
        if is_video:
            same = train[(train.video_file == r.video_file) & (train.class_name == r.class_name)]
            if len(same):
                i = (same.frame_time_s - r.frame_time_s).abs().idxmin()
                near, gap = same.at[i, "crop_path"], round(abs(same.at[i, "frame_time_s"] - r.frame_time_s), 3)
        rows.append({"crop_path": r.crop_path, "class_name": r.class_name, "source_type": src,
                     "video_file": r.video_file, "frame_time_s": r.frame_time_s,
                     "nearest_same_class_train_crop_same_video": near, "time_gap_s": gap,
                     f"near_duplicate_within_{args.gap:g}s": gap is not None and gap <= args.gap})
    out = pd.DataFrame(rows)
    pred = pd.read_csv(args.predictions)[["crop_path", "correct"]].rename(columns={"correct": "final_model_correct"})
    out = out.merge(pred, on="crop_path", how="left")
    assert out.final_model_correct.notna().all() and len(out) == len(test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    flag = f"near_duplicate_within_{args.gap:g}s"
    video = out[out.source_type == "gopro_video_frame"]
    summary = {
        "definition": __doc__.split("Outputs")[0].strip(),
        "test_crops": len(out),
        "test_crops_by_source": out.source_type.value_counts().to_dict(),
        "near_duplicates": int(out[flag].sum()),
        "near_duplicate_pct_of_all_test_crops": round(100 * out[flag].mean(), 2),
        "within_gap_counts": {f"{g:g}s": int(((video.time_gap_s <= g) & video.time_gap_s.notna()).sum())
                              for g in (1, 2, 5, 10)},
        "final_model_accuracy": {
            "all_test_crops": round(100 * out.final_model_correct.mean(), 2),
            "near_duplicates": round(100 * out[out[flag]].final_model_correct.mean(), 2),
            "excluding_near_duplicates": round(100 * out[~out[flag]].final_model_correct.mean(), 2),
            "n_excluding_near_duplicates": int((~out[flag]).sum()),
        },
    }
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main()
