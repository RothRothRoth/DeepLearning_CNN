"""Grouped 5-fold cross-validation of the FINAL ResNet-18 configuration (Phase 3).

Configuration (identical to the final tuned model, config B of notebooks/05_hyperparameter_tuning.ipynb):
  ImageNet ResNet-18, all layers trainable, new 512 -> 26 head; Adam lr 0.001, weight decay 0.001; batch 32;
  max 40 epochs; ReduceLROnPlateau (mode max, factor 0.5, patience 3) on val_acc; early stopping patience 10;
  best-validation-epoch weights restored; train / eval transforms from src/data/transforms.py.
No hyperparameter tuning. Seeds 42, 43, 44.
Deviation (data loading only): training loader num_workers = 4 with persistent_workers = True; validation / test
loaders num_workers = 0 (main process). The Colab setting was 2 workers for all loaders. On this Windows laptop:
2 workers -> GPU ~38% busy (19 min per run); 8 non-persistent workers -> 8 processes restarted every epoch, GPU ~7%
busy; 8 persistent workers per loader -> out of virtual memory (each worker ~2.5 GB). Loader settings change only
the random-augmentation streams, not the model setup (learning rate, weight decay, batch size, epochs, early
stopping and scheduler are unchanged).
The single run made with 2 workers is kept in results/grouped_evaluation/superseded_num_workers2/ (not reported).

For outer fold k (dataset/splits_grouped/fold_k): train on train.csv, early stopping / best epoch on val.csv
(= fold (k+1) mod 5), then ONE evaluation on test.csv (= fold k). Every drive cluster is tested exactly once.

Outputs (new folders only):
  D:\\DL_phase3\\checkpoints\\fold{k}_seed{s}.pt                (checkpoints, outside the repository)
  results/grouped_evaluation/runs/fold{k}_seed{s}/            (metrics.json, history.json, test_predictions.csv,
                                                              per_class.csv, confusion_matrix.csv)
  results/grouped_evaluation/summary/                         (written by --summarise)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from src.data.dataset import TrafficSignDataset  # noqa: E402
from src.data.transforms import get_eval_transforms, get_train_transforms  # noqa: E402
from src.models import create_resnet18_finetuned  # noqa: E402
from src.seed import seed_worker, set_seed  # noqa: E402
from src.training.train import train_model  # noqa: E402

FIXED = {"model_name": "resnet18_finetuned", "learning_rate": 0.001, "weight_decay": 0.001, "batch_size": 32,
         "num_workers": 4, "persistent_workers": True, "eval_num_workers": 0, "max_epochs": 40, "scheduler": "ReduceLROnPlateau", "scheduler_mode": "max",
         "scheduler_factor": 0.5, "scheduler_patience": 3, "monitor": "val_acc", "early_stopping_patience": 10}
SEEDS = (42, 43, 44)


def loaders(fold_dir, processed, seed):
    """Same construction as src.data.dataset.get_dataloaders, but for one grouped fold."""
    ds = {r: TrafficSignDataset(fold_dir / f"{r}.csv", processed_root=processed,
                                transform=get_train_transforms() if r == "train" else get_eval_transforms())
          for r in ("train", "val", "test")}
    g = torch.Generator()
    g.manual_seed(seed)
    pin = torch.cuda.is_available()
    kw = dict(batch_size=FIXED["batch_size"], pin_memory=pin, worker_init_fn=seed_worker, generator=g)
    train = DataLoader(ds["train"], shuffle=True, num_workers=FIXED["num_workers"],
                       persistent_workers=FIXED["persistent_workers"], **kw)
    evals = [DataLoader(ds[r], shuffle=False, num_workers=FIXED["eval_num_workers"], **kw) for r in ("val", "test")]
    return train, evals[0], evals[1]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, confs = [], [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        prob = torch.softmax(model(x.to(device)), 1).cpu()
        ys.append(y)
        ps.append(prob.argmax(1))
        confs.append(prob.max(1).values)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(confs).numpy()


def metrics_of(y, p):
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, labels=range(NUM_CLASSES), average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y, p, labels=range(NUM_CLASSES), average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
            "f1_weighted": fw}


def run_one(k, seed, args, device):
    name = f"fold{k}_seed{seed}"
    out = args.results / "runs" / name
    ckpt = args.checkpoints / f"{name}.pt"
    if (out / "metrics.json").exists():
        print(f"[skip] {name} already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    fold_dir = args.grouped_dir / f"fold_{k}"
    train_loader, val_loader, test_loader = loaders(fold_dir, args.processed, seed)
    model = create_resnet18_finetuned(num_classes=NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=FIXED["learning_rate"], weight_decay=FIXED["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=FIXED["scheduler_mode"],
                                                       factor=FIXED["scheduler_factor"],
                                                       patience=FIXED["scheduler_patience"])
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, opt, num_epochs=FIXED["max_epochs"], device=device,
                          scheduler=sched, criterion=torch.nn.CrossEntropyLoss(), monitor=FIXED["monitor"],
                          early_stopping_patience=FIXED["early_stopping_patience"], checkpoint_path=ckpt,
                          model_name=FIXED["model_name"], num_classes=NUM_CLASSES, seed=seed,
                          hyperparameters={**FIXED, "outer_fold": k, "grouped_split": str(fold_dir)}, verbose=False)
    train_time = time.time() - t0
    y, p, conf = predict(model, test_loader, device)  # the ONE test evaluation of this run
    test_df = pd.read_csv(fold_dir / "test.csv")
    assert (test_df.class_idx.values == y).all(), "test order mismatch"
    m = metrics_of(y, p)
    rep = classification_report(y, p, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).T.to_csv(out / "per_class.csv")
    pd.DataFrame(confusion_matrix(y, p, labels=range(NUM_CLASSES)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out / "confusion_matrix.csv")
    test_df.assign(pred_label=[CLASS_NAMES[i] for i in p], confidence=conf, correct=(y == p)).to_csv(
        out / "test_predictions.csv", index=False)
    (out / "history.json").write_text(json.dumps({k2: v for k2, v in history.items()}, indent=1, default=str))
    info = {"run": name, "outer_fold": k, "val_fold": (k + 1) % 5, "seed": seed, **FIXED,
            "train_crops": len(train_loader.dataset), "val_crops": len(val_loader.dataset),
            "test_crops": len(test_loader.dataset), "best_epoch": history["best_epoch"],
            "best_val_acc": history["best_metric"], "epochs_completed": history["epochs_completed"],
            "stopped_early": history["stopped_early"], "train_time_sec": round(train_time, 1),
            "checkpoint": str(ckpt), "test": m, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__}
    (out / "metrics.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: best epoch {info['best_epoch']} (val acc {info['best_val_acc']:.4f}), "
          f"test acc {m['accuracy']:.4f}, macro F1 {m['f1_macro']:.4f}, {train_time / 60:.1f} min", flush=True)


def summarise(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in sorted((args.results / "runs").glob("*/metrics.json"))]
    table = pd.DataFrame([{"run": r["run"], "outer_fold": r["outer_fold"], "seed": r["seed"],
                           "test_crops": r["test_crops"], "best_epoch": r["best_epoch"],
                           "best_val_acc": r["best_val_acc"], **r["test"],
                           "train_time_min": r["train_time_sec"] / 60} for r in runs])
    table.to_csv(out / "per_run_metrics.csv", index=False)
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    per_fold = table.groupby("outer_fold")[cols].agg(["mean", "std"])
    per_fold.to_csv(out / "per_fold_mean_std_over_seeds.csv")
    per_seed = table.groupby("seed")[cols].mean()
    per_seed.to_csv(out / "per_seed_mean_over_folds.csv")

    # pooled over the 5 folds (every crop tested exactly once per seed)
    pooled, cms = [], []
    for s in SEEDS:
        preds = pd.concat([pd.read_csv(args.results / "runs" / f"fold{k}_seed{s}" / "test_predictions.csv")
                           for k in range(5)])
        y = preds.class_name.map(CLASS_NAMES.index).values
        p = preds.pred_label.map(CLASS_NAMES.index).values
        pooled.append({"seed": s, "crops": len(preds), **metrics_of(y, p)})
        rep = pd.DataFrame(classification_report(y, p, labels=range(NUM_CLASSES), target_names=CLASS_NAMES,
                                                 output_dict=True, zero_division=0)).T.loc[CLASS_NAMES]
        rep["seed"] = s
        cms.append(confusion_matrix(y, p, labels=range(NUM_CLASSES)))
        pooled[-1]["per_class"] = rep
    pooled_df = pd.DataFrame([{k: v for k, v in d.items() if k != "per_class"} for d in pooled])
    pooled_df.to_csv(out / "pooled_over_folds_per_seed.csv", index=False)
    pc = pd.concat([d["per_class"] for d in pooled])
    per_class = pc.groupby(level=0)[["precision", "recall", "f1-score", "support"]].agg(["mean", "std"]).loc[CLASS_NAMES]
    per_class.to_csv(out / "per_class_pooled_mean_std_over_seeds.csv")
    cm = np.sum(cms, axis=0)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / "confusion_matrix_pooled_all_seeds.csv")
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=7)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Grouped 5-fold CV, pooled over folds and 3 seeds (row-normalised colour, counts in cells)")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_pooled_all_seeds.png", dpi=150)

    # sensitivity check: test crops whose sign+surroundings strongly match a same-class TRAINING crop of that fold
    # (possible same physical sign photographed on another trip; see scripts/check_grouped_leakage.py)
    leak = None
    feat, order = args.leakage_dir / "context_features.npy", args.leakage_dir / "context_features_order.json"
    leak_report = args.results / "leakage_check.json"
    if feat.exists() and order.exists() and leak_report.exists():
        F = np.load(feat)
        key = {c: i for i, c in enumerate(json.loads(order.read_text()))}
        thr = json.loads(leak_report.read_text())["calibration"]["reference_p999"]
        strong = set()
        for k in range(5):
            tr = pd.read_csv(args.grouped_dir / f"fold_{k}" / "train.csv")
            te = pd.read_csv(args.grouped_dir / f"fold_{k}" / "test.csv")
            for r in te.itertuples():
                cand = [key[c] for c in tr[tr.class_name == r.class_name].crop_path]
                if cand and float((F[cand] @ F[key[r.crop_path]]).max()) >= thr:
                    strong.add(r.crop_path)
        rows = []
        for s in SEEDS:
            preds = pd.concat([pd.read_csv(args.results / "runs" / f"fold{k}_seed{s}" / "test_predictions.csv")
                               for k in range(5)])
            m = preds.crop_path.isin(strong)
            rows.append({"seed": s, "acc_strong_match": preds[m].correct.mean(), "n_strong_match": int(m.sum()),
                         "acc_other": preds[~m].correct.mean(), "n_other": int((~m).sum())})
        leak = pd.DataFrame(rows)
        leak.to_csv(out / "strong_match_sensitivity.csv", index=False)

    overall = {
        "runs": len(table),
        "strong_match_sensitivity": None if leak is None else {
            "threshold": thr, "n_strong_match": int(leak.n_strong_match.iloc[0]),
            "acc_strong_match_mean_std": [float(leak.acc_strong_match.mean()), float(leak.acc_strong_match.std())],
            "acc_other_mean_std": [float(leak.acc_other.mean()), float(leak.acc_other.std())]},
        "mean_std_over_15_runs": {c: [float(table[c].mean()), float(table[c].std())] for c in cols},
        "pooled_over_folds_mean_std_over_seeds": {c: [float(pooled_df[c].mean()), float(pooled_df[c].std())]
                                                  for c in cols},
        "fold_means": {int(k): float(v) for k, v in table.groupby("outer_fold").accuracy.mean().items()},
        "train_time_min_total": float(table.train_time_min.sum()),
        "train_time_min_mean": float(table.train_time_min.mean()),
    }
    (out / "summary.json").write_text(json.dumps(overall, indent=1))
    print(json.dumps(overall, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grouped-dir", type=Path, default=PROJECT_ROOT / "dataset" / "splits_grouped")
    ap.add_argument("--processed", type=Path, default=PROJECT_ROOT / "dataset" / "processed")
    ap.add_argument("--checkpoints", type=Path, default=Path(r"D:\DL_phase3\checkpoints"))
    ap.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "grouped_evaluation")
    ap.add_argument("--leakage-dir", type=Path, default=Path(r"D:\DL_phase3\leakage"))
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--summarise", action="store_true", help="only aggregate finished runs")
    args = ap.parse_args()
    if args.summarise:
        summarise(args)
        return
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    for k in args.folds:
        for s in args.seeds:
            run_one(k, s, args, device)


if __name__ == "__main__":
    main()
