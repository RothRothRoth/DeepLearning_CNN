"""Builds drive-cluster groups and a grouped 5-fold cross-validation split (Phase 3).

Why: the original split separates source PHOTOS, but many photos are video frames of the same drive (and the
NR*_*.jpg photos turned out to be renamed frames of the GoPro recordings), so near-identical scenes can end up in
train and test. Here whole DRIVE CLUSTERS are kept together.

Grouping evidence (no photo has EXIF metadata, so only file names and image content are available):
  GX<chapter><recording>.MP4_snapshot_<min>.<sec>.<ms>  -> GoPro recording (all chapters = one drive)
  NR<road>_<n>                                          -> road; linked to the recording of a near-identical frame
  photo_..._<date>_<time>                                -> one phone session per day
Near-identical = cosine >= 0.96 between ImageNet ResNet-18 features of the whole photos (calibration: GoPro frames
< 1 s apart have median 0.959; photos from different recordings reach 0.945 at the 99.9th percentile).
Clusters = union of: one recording; one NR road; one phone day; NR road <-> recording of its near-identical frames;
two recordings sharing >= 2 near-identical frames.

Folds: the clusters are assigned to 5 folds by a seeded search. For outer fold k: test = fold k,
validation = fold (k+1) mod 5 (early stopping / best epoch only), train = the other three folds. Every cluster is
tested exactly once. Constraint: every training set contains all 26 classes.

Outputs (new files only): dataset/splits_grouped/  (cluster table, fold table, per-fold train/val/test CSVs, summary)
Features are cached in D:\\DL_phase3\\features\\ (outside the repository).
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_NAMES  # noqa: E402

GX = re.compile(r"^GX(\d{2})(\d{4})\.MP4_snapshot_(\d+)\.(\d+)\.(\d+)", re.I)
NR = re.compile(r"^NR(\d+)_(\d+)\.", re.I)
PH = re.compile(r"^photo_(?:\d+_)?(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)", re.I)


def photo_group(name):
    if m := GX.match(name):
        return "gopro", "REC" + m.group(2), int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000
    if m := NR.match(name):
        return "nr", "NR" + m.group(1), float(m.group(2))
    if m := PH.match(name):
        return "photo", "PHOTO_" + m.group(1), float(int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)))
    raise ValueError(f"Unrecognised photo name pattern: {name}")


@torch.no_grad()
def photo_features(paths, device):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((216, 384)), torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    for k in range(0, len(paths), 32):
        batch = []
        for p in paths[k:k + 32]:
            im = Image.open(p)
            im.draft("RGB", (im.width // 4, im.height // 4))
            batch.append(tf(im.convert("RGB")))
        out.append(torch.nn.functional.normalize(net(torch.stack(batch).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-images", type=Path, default=PROJECT_ROOT / "dataset" / "raw" / "images")
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits_grouped")
    ap.add_argument("--feature-dir", type=Path, default=Path(r"D:\DL_phase3\features"))
    ap.add_argument("--threshold", type=float, default=0.96)
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-iters", type=int, default=300000)
    args = ap.parse_args()
    if args.out_dir.exists():
        sys.exit(f"{args.out_dir} already exists; refusing to overwrite. Delete or rename it yourself first.")

    # ---- photos and file-name groups ----
    names = sorted(p.name for p in args.raw_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    photos = pd.DataFrame([(n, *photo_group(n)) for n in names], columns=["photo", "family", "group", "t"])

    # ---- whole-photo similarity ----
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    feat_file = args.feature_dir / "photo_features_resnet18_imagenet.npy"
    names_file = args.feature_dir / "photo_features_names.json"
    if feat_file.exists() and json.loads(names_file.read_text()) == names:
        F = np.load(feat_file)
    else:
        F = photo_features([args.raw_images / n for n in names], "cuda" if torch.cuda.is_available() else "cpu")
        np.save(feat_file, F)
        names_file.write_text(json.dumps(names))
    S = F @ F.T

    # ---- union-find clusters ----
    parent = {g: g for g in photos.group.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    gop = photos.index[photos.family == "gopro"].values
    evidence = []
    nr_idx = photos.index[photos.family == "nr"].values
    best = S[np.ix_(nr_idx, gop)]
    for i, (sc, j) in enumerate(zip(best.max(1), best.argmax(1))):
        if sc >= args.threshold:
            a, b = photos.group[nr_idx[i]], photos.group[gop[j]]
            union(a, b)
            evidence.append({"type": "NR photo ~ GoPro frame", "a": photos.photo[nr_idx[i]], "b": photos.photo[gop[j]],
                             "cosine": round(float(sc), 4)})
    iu = np.triu_indices(len(gop), 1)
    sub = S[np.ix_(gop, gop)][iu]
    shared = {}
    for a, b in zip(iu[0][sub >= args.threshold], iu[1][sub >= args.threshold]):
        ga, gb = photos.group[gop[a]], photos.group[gop[b]]
        if ga != gb:
            shared.setdefault(tuple(sorted((ga, gb))), []).append((photos.photo[gop[a]], photos.photo[gop[b]]))
    for (ga, gb), pairs in shared.items():
        if len(pairs) >= args.min_shared:
            union(ga, gb)
            evidence.append({"type": f"recordings share {len(pairs)} near-identical frames", "a": ga, "b": gb,
                             "example": list(pairs[0])})
    photos["cluster"] = photos.group.map(find)
    cluster_name = {c: "+".join(sorted(photos[photos.cluster == c].group.unique())) for c in photos.cluster.unique()}
    photos["cluster"] = photos.cluster.map(cluster_name)

    # ---- crops (all 3,070 from the original manifests) ----
    crops = pd.concat([pd.read_csv(args.splits_dir / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    crops = crops.rename(columns={"split": "original_split"})
    crops["cluster"] = crops.parent_filename.map(dict(zip(photos.photo, photos.cluster)))
    assert crops.cluster.notna().all()
    counts = crops.groupby(["cluster", "class_name"]).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES,
                                                                                           fill_value=0)
    clusters = counts.index.tolist()
    C = counts.values
    total = C.sum()

    # ---- seeded fold search ----
    K = args.folds
    rng = np.random.default_rng(args.seed)
    best_key, best_assign = None, None
    for _ in range(args.search_iters):
        a = rng.integers(0, K, len(clusters))
        per = np.stack([C[a == k].sum(0) for k in range(K)])
        if (per.sum(1) == 0).any():
            continue
        ok = all((per[[j for j in range(K) if j not in (k, (k + 1) % K)]].sum(0) > 0).all() for k in range(K))
        if not ok:  # every training set (3 folds) must contain all 26 classes
            continue
        classes_per_fold = (per > 0).sum(1)
        size = per.sum(1) / total
        key = (classes_per_fold.min(), -float(np.std(size)), classes_per_fold.sum())
        if best_key is None or key > best_key:
            best_key, best_assign = key, a.copy()
    if best_assign is None:
        sys.exit("No fold assignment satisfies the constraints.")
    fold_of = dict(zip(clusters, best_assign))
    crops["fold"] = crops.cluster.map(fold_of)
    photos["fold"] = photos.cluster.map(fold_of)

    # ---- write ----
    args.out_dir.mkdir(parents=True)
    photos.to_csv(args.out_dir / "photo_clusters.csv", index=False)
    pd.DataFrame({"cluster": clusters, "fold": [fold_of[c] for c in clusters],
                  "crops": C.sum(1), "photos": [int((photos.cluster == c).sum()) for c in clusters]}
                 ).sort_values(["fold", "crops"], ascending=[True, False]).to_csv(args.out_dir / "cluster_folds.csv",
                                                                                 index=False)
    (args.out_dir / "grouping_evidence.json").write_text(json.dumps(evidence, indent=1))
    summary = {"design": __doc__, "threshold": args.threshold, "min_shared_frames": args.min_shared,
               "seed": args.seed, "search_iters": args.search_iters, "n_clusters": len(clusters), "folds": []}
    for k in range(K):
        v = (k + 1) % K
        roles = {"test": crops[crops.fold == k], "val": crops[crops.fold == v],
                 "train": crops[~crops.fold.isin([k, v])]}
        d = args.out_dir / f"fold_{k}"
        d.mkdir()
        for role, df in roles.items():
            df.assign(split=role).to_csv(d / f"{role}.csv", index=False)
        summary["folds"].append({
            "outer_fold": k, "test_fold": k, "val_fold": v,
            **{f"{r}_crops": len(df) for r, df in roles.items()},
            **{f"{r}_photos": int(df.parent_filename.nunique()) for r, df in roles.items()},
            **{f"{r}_clusters": sorted(df.cluster.unique()) for r, df in roles.items()},
            **{f"{r}_classes": int(df.class_name.nunique()) for r, df in roles.items()},
        })
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(clusters)} clusters -> {K} folds written to {args.out_dir}")


if __name__ == "__main__":
    main()
