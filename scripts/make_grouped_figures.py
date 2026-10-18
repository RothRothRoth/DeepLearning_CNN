"""Figures for the grouped evaluation, made only from existing outputs (no training, no new metrics).

  results/grouped_evaluation/figures/per_fold_variation.png
      test accuracy and macro F1 of every run (5 folds x 3 seeds) from summary/per_run_metrics.csv
  results/grouped_evaluation/figures/cross_road_vs_priority_road_examples.png
      real test crops (dataset/processed) of CROSS_ROAD and PRIORITY_ROAD, plus CROSS_ROAD crops that the seed-42
      models predicted as PRIORITY_ROAD (runs/fold*_seed42/test_predictions.csv); first crops in alphabetical order
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "grouped_evaluation"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- per-fold variation ----
runs = pd.read_csv(RES / "summary" / "per_run_metrics.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, col, title in zip(axes, ("accuracy", "f1_macro"), ("Test accuracy", "Test macro F1")):
    for k in range(5):
        r = runs[runs.outer_fold == k]
        ax.bar(k, 100 * r[col].mean(), color="#C9DDD2", width=0.6, zorder=1)
        ax.scatter([k] * len(r), 100 * r[col], color="#0B5A3C", zorder=2, s=28)
        ax.text(k, 90.4, f"{100 * r[col].mean():.1f}", ha="center", fontsize=10, fontweight="bold", color="#0B5A3C")
    ax.set_xticks(range(5), [f"fold {k}\n({int(runs[runs.outer_fold == k].test_crops.iloc[0])} crops)"
                             for k in range(5)], fontsize=9)
    ax.set_title(title)
    ax.set_ylim(90, 100.8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("%")
fig.suptitle("Grouped 5-fold CV: bar = mean of 3 seeds, dots = individual seeds (42, 43, 44)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "per_fold_variation.png", dpi=160)
plt.close()

# ---- Cross Road vs Priority Road examples ----
preds = pd.concat([pd.read_csv(RES / "runs" / f"fold{k}_seed42" / "test_predictions.csv") for k in range(5)])
preds = preds.sort_values("crop_path")
rows = [("CROSS_ROAD (correct)", preds[(preds.class_name == "CROSS_ROAD") & preds.correct].head(5)),
        ("PRIORITY_ROAD (correct)", preds[(preds.class_name == "PRIORITY_ROAD") & preds.correct].head(5)),
        ("CROSS_ROAD predicted as PRIORITY_ROAD",
         preds[(preds.class_name == "CROSS_ROAD") & (preds.pred_label == "PRIORITY_ROAD")].head(5))]
fig, axes = plt.subplots(3, 5, figsize=(9, 7.6), gridspec_kw={"hspace": 0.75, "wspace": 0.08})
plt.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.02)  # fix the layout before reading positions
for i, (label, df) in enumerate(rows):
    for j in range(5):
        ax = axes[i, j]
        ax.axis("off")
        if j < len(df):
            r = df.iloc[j]
            ax.imshow(Image.open(ROOT / "dataset" / "processed" / r.crop_path).convert("RGB").resize((160, 160)))
            ax.set_title(f"pred. {r.pred_label.replace('_', ' ').title()}\nconf {100 * r.confidence:.0f}%",
                         fontsize=7.5)
    pos = axes[i, 0].get_position()
    fig.text(0.02, pos.y1 + 0.062, ("True " if i < 2 else "") + label.replace("_", " ").title()
             .replace("Predicted As", "predicted as"), fontsize=11, fontweight="bold",
             color="#C44E52" if i == 2 else "#0B5A3C")
fig.suptitle("Real test crops from the grouped evaluation (seed-42 models)", fontsize=12, y=0.995)
plt.savefig(OUT / "cross_road_vs_priority_road_examples.png", dpi=160)
plt.close()
print("written:", sorted(p.name for p in OUT.iterdir()))
