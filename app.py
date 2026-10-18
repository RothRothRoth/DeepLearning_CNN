"""Gradio web demo: Cambodian traffic-sign recognition with the final fine-tuned ResNet-18.

Demo only: nothing is trained. The model, checkpoint loader, 26-class mapping and
evaluation preprocessing all come from the project code in src/:

    create_resnet18_finetuned()   src/models/resnet.py
    load_checkpoint()             src/training/checkpoint.py
    get_eval_transforms()         src/data/transforms.py   (resize 224, ImageNet normalisation)
    CLASS_NAMES                   src/config.py

Every number and every "correct" / "incorrect" label on the page is read from the saved
results in results/; nothing is hard-coded.

Optional second tab, "Detect All Signs in a Street Photo": a 1-class YOLO detector (scripts/train_sign_detector.py)
finds every traffic sign; each box is cropped from the full-resolution photo and classified by the SAME ResNet-18.
The detector's single class ("Traffic Sign") is only a detection label; the sign type always comes from ResNet-18.

Usage (local, serves http://127.0.0.1:7860 only; Colab: see notebooks/08_web_demo.ipynb):
    python app.py --checkpoint path/to/resnet18_finetuned_tuned_best.pt
Optional: --processed-dir path/to/dataset/processed, --port N, --share (temporary public link),
          --detector path/to/sign_detector_best.pt, --detector-metrics path/to/detector_and_pipeline_metrics.json.
"""

import argparse
import html
import inspect
import json
import os
import pathlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from src.data.dataset import resolve_crop_path
from src.data.transforms import get_eval_transforms
from src.models import create_resnet18_finetuned
from src.training import get_device, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_RUN_DIR = PROJECT_ROOT / "results" / "hyperparameter_tuning" / "resnet18_finetuned_lr_wd_seed42"
SAVED_PREDICTIONS = FINAL_RUN_DIR / "final_tuned_model" / "test_predictions.csv"
FINAL_METRICS = FINAL_RUN_DIR / "metrics" / "final_metrics.json"
COMPARISON_TABLE = PROJECT_ROOT / "results" / "final_comparison" / "final_results_table.csv"
MISCLASSIFIED_ANALYSIS = PROJECT_ROOT / "results" / "final_model" / "misclassified_test_images.csv"
TEST_MANIFEST = PROJECT_ROOT / "dataset" / "splits" / "test.csv"
CHECKPOINT_NAME = "resnet18_finetuned_tuned_best.pt"
DRIVE_CHECKPOINT_RELATIVE = ("results/runs/hyperparameter_tuning/resnet18_finetuned_lr_wd_seed42/"
                             "final_tuned_model/" + CHECKPOINT_NAME)
LOW_CONFIDENCE = 0.5


def pretty(class_name: str) -> str:
    """Readable class name, e.g. 'ROAD_JUNCTION_ON_THE_LEFT' -> 'Road Junction On The Left'."""
    return class_name.replace("_", " ").title()


class SignClassifier:
    """The final fine-tuned ResNet-18, loaded from its real checkpoint, in eval mode."""

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Final checkpoint not found: {self.checkpoint_path}\n"
                f"Expected the file '{CHECKPOINT_NAME}' produced by notebooks/05_hyperparameter_tuning.ipynb.\n"
                f"On Google Drive it is at MyDrive/Road_Sign_Recognition_CNN/{DRIVE_CHECKPOINT_RELATIVE}"
            )
        self.device = device or get_device()
        self.model = create_resnet18_finetuned(num_classes=NUM_CLASSES)
        self.checkpoint = load_checkpoint(self.checkpoint_path, model=self.model, map_location=self.device)
        self.model.to(self.device).eval()
        self.transform = get_eval_transforms(IMAGE_SIZE)

    @torch.no_grad()
    def probabilities(self, image: Image.Image) -> torch.Tensor:
        """Softmax probabilities over the 26 classes for one image."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu()

    def top_k(self, image: Image.Image, k: int = 3):
        """[(class_name, probability), ...] for the k most likely classes."""
        probs = self.probabilities(image)
        top = torch.topk(probs, k)
        return [(CLASS_NAMES[i], float(p)) for p, i in zip(top.values, top.indices)]


def _import_yolo():
    """Imports Ultralytics' YOLO despite a Windows quirk (see scripts/train_sign_detector.py).

    With Microsoft Store Python, Ultralytics' start-up check for a .git folder touches the protected
    D:\\WpSystem\\.git and raises WinError 1337; during the import only, that OSError counts as "does not exist"."""
    original_exists = pathlib.Path.exists

    def safe_exists(self, *args, **kwargs):
        try:
            return original_exists(self, *args, **kwargs)
        except OSError:
            return False

    pathlib.Path.exists = safe_exists
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    finally:
        pathlib.Path.exists = original_exists
    SETTINGS.update({"sync": False})  # no anonymous usage statistics
    return YOLO


class SignDetector:
    """The trained 1-class YOLO detector. It only says WHERE a traffic sign is ("Traffic Sign");
    WHICH sign it is always comes from the ResNet-18 classifier."""

    def __init__(self, weights_path, imgsz=1280):
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Detector weights not found: {self.weights_path}")
        self.model = _import_yolo()(str(self.weights_path))
        if self.model.task != "detect":
            raise ValueError(f"{self.weights_path.name} is a '{self.model.task}' model, not a detector")
        self.imgsz = imgsz
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect(self, image: Image.Image, conf=0.25):
        """[(x1, y1, x2, y2), detector_confidence] in original-image pixels, sorted left to right."""
        boxes = self.model.predict(image, imgsz=self.imgsz, conf=conf, device=self.device, verbose=False)[0].boxes
        dets = [(tuple(float(v) for v in b), float(c)) for b, c in zip(boxes.xyxy.cpu(), boxes.conf.cpu())]
        return sorted(dets, key=lambda d: (d[0][0], d[0][1]))


def load_detector_metrics(path):
    """Detector and end-to-end test metrics saved by scripts/train_sign_detector.py (None if unavailable)."""
    if not path or not Path(path).is_file():
        return None
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("smoke_test"):
        return None
    return {"det": m["detector_test_metrics"], "e2e": m["end_to_end_test"]}


def detection_examples(raw_dir, n=4):
    """Full street photos from the TEST split only (several signs each), for the detection tab.

    A photo is used only if it is a test parent and is NOT a parent in the train or val manifests."""
    raw_dir = Path(raw_dir) if raw_dir else None
    splits_dir = TEST_MANIFEST.parent
    if raw_dir is None or not raw_dir.is_dir() or not all((splits_dir / f"{s}.csv").is_file()
                                                          for s in ("train", "val", "test")):
        return []
    parents = {s: set(pd.read_csv(splits_dir / f"{s}.csv")["parent_filename"]) for s in ("train", "val", "test")}
    test_only = parents["test"] - parents["train"] - parents["val"]
    counts = pd.read_csv(TEST_MANIFEST).groupby("parent_filename").size()
    candidates = sorted(((-c, p) for p, c in counts.items() if c >= 2 and p in test_only))
    picked, sources = [], set()
    for _, p in candidates:
        source = p.split("_snapshot")[0].split("_")[0]  # one photo per video / road, for variety
        if source not in sources and (raw_dir / p).is_file():
            picked.append(str(raw_dir / p))
            sources.add(source)
        if len(picked) == n:
            break
    return picked


def verify_checkpoint(classifier, processed_dir, n_correct=18, tolerance=1e-3):
    """Re-predicts saved test images and compares with the saved final-model predictions.

    Uses all 6 misclassified test images plus `n_correct` correctly classified ones
    (fixed random_state 42). Returns {"status": "verified" | "failed" | "not_checked", "detail": str}.
    The checkpoint is reported as verified ONLY if every re-prediction matches.
    """
    processed_dir = Path(processed_dir) if processed_dir else None
    if not SAVED_PREDICTIONS.is_file():
        return {"status": "not_checked", "detail": f"saved predictions not found: {SAVED_PREDICTIONS}"}
    if processed_dir is None or not processed_dir.is_dir():
        return {"status": "not_checked", "detail": f"test images folder not found: {processed_dir}"}

    saved = pd.read_csv(SAVED_PREDICTIONS)
    sample = pd.concat([saved[~saved["correct"]], saved[saved["correct"]].sample(n_correct, random_state=42)])
    mismatches, missing = [], []
    for _, row in sample.iterrows():
        path = resolve_crop_path(processed_dir, row["crop_path"])
        if not path.is_file():
            missing.append(row["crop_path"])
            continue
        (pred, conf), *_ = classifier.top_k(Image.open(path), k=1)
        if pred != row["pred_label"] or abs(conf - row["confidence"]) > tolerance:
            mismatches.append(row["crop_path"])

    checked = len(sample) - len(missing)
    if missing:
        return {"status": "not_checked",
                "detail": f"{len(missing)} of {len(sample)} test images missing (e.g. {missing[0]}); "
                          f"{len(mismatches)} mismatches among the {checked} found"}
    if mismatches:
        return {"status": "failed",
                "detail": f"{len(mismatches)} of {len(sample)} re-predictions differ from the saved "
                          f"final-model predictions (e.g. {mismatches[0]})"}
    return {"status": "verified",
            "detail": f"all {len(sample)} re-predictions match the saved final-model test predictions"}


def load_reported_metrics():
    """Final test metrics as saved by notebooks/05_hyperparameter_tuning.ipynb (None if unavailable)."""
    if not FINAL_METRICS.is_file():
        return None
    m = json.loads(FINAL_METRICS.read_text(encoding="utf-8"))
    t, best = m["final_test_metrics"], m["best_config"]
    return {"accuracy": t["accuracy"], "f1_macro": t["f1_macro"], "num_test": t["num_test_samples"],
            "config": best["config"], "lr": best["learning_rate"], "wd": best["weight_decay"]}


def load_comparison():
    """Test accuracy and macro F1 of every approach, from notebooks/06 (empty list if unavailable)."""
    if not COMPARISON_TABLE.is_file():
        return []
    table = pd.read_csv(COMPARISON_TABLE)
    return [{"approach": r["Approach"], "accuracy": r["Test Accuracy"], "f1_macro": r["Macro F1"],
             "params": int(r["Trainable Parameters"])} for _, r in table.iterrows()]


def example_images(processed_dir, classes=None, n_failures=2, raw_dir=None):
    """Example test-set crops, labelled ONLY from the saved final-model predictions and ground truth.

    Correct examples: one per class (all 26 by default, KEEP_RIGHT first). For each class, the correctly
    classified test image with the MEDIAN confidence is used, so the examples are typical, not cherry-picked.
    Failure examples: the first `n_failures` test images analysed as "model error" in
    results/final_model/misclassified_test_images.csv (correct == False in the saved predictions).
    If the original street photos are available in `raw_dir` (default dataset/raw/images), each example also
    gets its street photo and the annotated sign box from the test manifest ("scene" / "bbox").
    Returns a list of dicts: {"path", "crop_path", "true_label", "saved_pred", "saved_conf", "correct", "note",
    "scene", "bbox"}.
    """
    if not processed_dir or not Path(processed_dir).is_dir() or not SAVED_PREDICTIONS.is_file():
        return []
    saved = pd.read_csv(SAVED_PREDICTIONS)
    raw_dir = Path(raw_dir) if raw_dir else PROJECT_ROOT / "dataset" / "raw" / "images"
    manifest = pd.read_csv(TEST_MANIFEST).set_index("crop_path") if TEST_MANIFEST.is_file() else None

    def case(row, note=""):
        path = resolve_crop_path(processed_dir, row["crop_path"])
        if not path.is_file():
            return None
        scene, bbox = None, None
        if manifest is not None and row["crop_path"] in manifest.index:
            m = manifest.loc[row["crop_path"]]
            if (raw_dir / m["parent_filename"]).is_file():
                scene = str(raw_dir / m["parent_filename"])
                bbox = tuple(int(m[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        return {"path": str(path), "crop_path": row["crop_path"], "true_label": row["true_label"],
                "saved_pred": row["pred_label"], "saved_conf": float(row["confidence"]),
                "correct": bool(row["correct"]), "note": note, "scene": scene, "bbox": bbox}

    if classes is None:
        classes = ["KEEP_RIGHT"] + [c for c in CLASS_NAMES if c != "KEEP_RIGHT"]
    cases = []
    for c in classes:
        rows = saved[(saved["true_label"] == c) & saved["correct"]].sort_values(["confidence", "crop_path"])
        if len(rows):
            cases.append(case(rows.iloc[(len(rows) - 1) // 2]))

    if MISCLASSIFIED_ANALYSIS.is_file():
        analysis = pd.read_csv(MISCLASSIFIED_ANALYSIS)
        for _, a in analysis[analysis["category"] == "model error"].head(n_failures).iterrows():
            rows = saved[(saved["crop_path"] == a["crop_path"]) & ~saved["correct"]]
            if len(rows):
                cases.append(case(rows.iloc[0], note=a["visual_observation"]))
    return [c for c in cases if c is not None]


# ----------------------------------------------------------------------------------------------
# Page styling (dark research-dashboard look; one accent colour, semantic colours only for status)
# ----------------------------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }
.tsr-header { padding: 28px 0 8px; }
.tsr-eyebrow { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: #38bdf8; font-weight: 600; }
.tsr-title { font-size: 34px; line-height: 1.15; font-weight: 700; color: #f1f5f9; margin: 6px 0 8px; }
.tsr-subtitle { font-size: 16px; color: #94a3b8; max-width: 760px; margin: 0; }
.tsr-meta { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin-top: 16px;
            font-size: 13px; color: #94a3b8; }
.tsr-status { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px;
              border: 1px solid #1f2a44; background: #0f172a; font-size: 13px; color: #e2e8f0; }
.tsr-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.tsr-ok .tsr-dot { background: #34d399; }  .tsr-bad .tsr-dot { background: #f87171; }
.tsr-warn .tsr-dot { background: #fbbf24; }
.tsr-bad { border-color: #7f1d1d; } .tsr-warn { border-color: #78350f; }
.tsr-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 8px; }
.tsr-card { background: #111a2e; border: 1px solid #1f2a44; border-radius: 10px; padding: 14px 16px; }
.tsr-card-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }
.tsr-card-value { font-size: 26px; font-weight: 700; color: #f1f5f9; margin-top: 4px; font-variant-numeric: tabular-nums; }
.tsr-card-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
.tsr-note { font-size: 12.5px; color: #64748b; margin: 10px 2px 0; }
.tsr-approaches { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin-top: 12px; }
.tsr-approach { border: 1px solid #1f2a44; border-radius: 10px; padding: 10px 14px; background: #0f172a; }
.tsr-approach.final { border-color: #38bdf8; }
.tsr-approach-name { font-size: 13.5px; color: #e2e8f0; font-weight: 600; }
.tsr-approach-num { font-size: 12.5px; color: #94a3b8; margin-top: 3px; font-variant-numeric: tabular-nums; }
.tsr-tag { display: inline-block; margin-left: 6px; padding: 1px 7px; border-radius: 999px; font-size: 11px;
           background: rgba(56,189,248,.14); color: #7dd3fc; font-weight: 600; vertical-align: 1px; }
.tsr-section-title { font-size: 15px; font-weight: 600; color: #e2e8f0; margin: 2px 0 2px; }
.tsr-help { font-size: 13px; color: #94a3b8; margin: 0 0 4px; }
.tsr-panel { background: #111a2e !important; border: 1px solid #1f2a44 !important; border-radius: 12px !important;
             padding: 16px !important; }
.tsr-classify button, button.tsr-classify { font-size: 16px !important; font-weight: 600 !important; min-height: 46px; }
.tsr-gallery-label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8;
                     margin: 10px 0 0; font-weight: 600; }
.tsr-gallery-label.fail { color: #fca5a5; }
.tsr-gallery { background: transparent !important; border: none !important; padding: 0 !important; }
.tsr-gallery .grid-wrap { overflow: hidden !important; }
.tsr-gallery .grid-container { grid-template-columns: repeat(var(--grid-cols), minmax(0, 1fr)) !important; }
.tsr-gallery .caption-label { font-size: 10.5px !important; max-width: 100%; }
.tsr-legend { font-size: 12.5px; color: #94a3b8; margin: 4px 0 0; padding-left: 18px; line-height: 1.5; }
.tsr-legend strong { color: #cbd5e1; }
.tsr-empty { min-height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center;
             text-align: center; color: #64748b; font-size: 14px; border: 1px dashed #1f2a44; border-radius: 10px; padding: 24px; }
.tsr-empty strong { color: #94a3b8; }
.tsr-result-label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; font-weight: 600; }
.tsr-pred { font-size: 32px; line-height: 1.15; font-weight: 700; color: #f8fafc; margin: 6px 0 2px; }
.tsr-code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; color: #64748b; }
.tsr-conf { display: flex; align-items: baseline; gap: 8px; margin: 14px 0 4px; }
.tsr-conf-value { font-size: 28px; font-weight: 700; color: #38bdf8; font-variant-numeric: tabular-nums; }
.tsr-conf-label { font-size: 13px; color: #94a3b8; }
.tsr-alert { border-radius: 8px; padding: 10px 12px; font-size: 13.5px; margin: 12px 0 0; line-height: 1.45; }
.tsr-alert-warn { background: rgba(251,191,36,.08); border: 1px solid #78350f; color: #fde68a; }
.tsr-alert-bad { background: rgba(248,113,113,.08); border: 1px solid #7f1d1d; color: #fecaca; }
.tsr-alert-ok { background: rgba(52,211,153,.07); border: 1px solid #065f46; color: #a7f3d0; }
.tsr-bars { margin-top: 18px; display: flex; flex-direction: column; gap: 10px; }
.tsr-bar-row { display: grid; grid-template-columns: 20px 1fr 64px; gap: 10px; align-items: center; font-size: 14px; }
.tsr-rank { color: #64748b; font-variant-numeric: tabular-nums; }
.tsr-bar-name { color: #e2e8f0; margin-bottom: 5px; }
.tsr-bar-track { height: 8px; background: #1e293b; border-radius: 999px; overflow: hidden; }
.tsr-bar-fill { height: 100%; background: #38bdf8; border-radius: 999px; }
.tsr-bar-row.rest .tsr-bar-fill { background: #475569; }
.tsr-bar-pct { text-align: right; color: #cbd5e1; font-variant-numeric: tabular-nums; }
.tsr-fine { font-size: 12.5px; color: #64748b; margin-top: 16px; line-height: 1.5; }
.tsr-whole { margin-top: 18px; padding-top: 18px; border-top: 1px dashed #1f2a44; }
.tsr-details { font-size: 14px; color: #cbd5e1; line-height: 1.6; }
.tsr-details h4 { color: #e2e8f0; font-size: 14px; margin: 14px 0 4px; }
.tsr-details ul { margin: 0; padding-left: 18px; }
.tsr-details code { font-size: 12px; }
.tsr-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.tsr-chip { font-size: 12px; padding: 3px 9px; border-radius: 999px; border: 1px solid #1f2a44; color: #cbd5e1; background: #0f172a; }
.tsr-footer { text-align: center; font-size: 12.5px; color: #475569; padding: 18px 0 8px; }
@media (max-width: 640px) {
  .tsr-title { font-size: 26px; }
  .tsr-panel { padding: 10px !important; }
  .tsr-pred { font-size: 26px; }
  .tsr-status { border-radius: 10px; }
}
"""


def _theme(gr):
    """Navy/charcoal theme, applied identically in light and dark browser modes."""
    colours = dict(
        body_background_fill="#0b1120", body_text_color="#e2e8f0", body_text_color_subdued="#94a3b8",
        background_fill_primary="#111a2e", background_fill_secondary="#0f172a",
        border_color_primary="#1f2a44", border_color_accent="#38bdf8", color_accent="#38bdf8",
        color_accent_soft="rgba(56,189,248,.15)", block_background_fill="#111a2e", block_border_color="#1f2a44",
        block_label_background_fill="#0f172a", block_label_text_color="#94a3b8", block_title_text_color="#e2e8f0",
        input_background_fill="#0f172a", input_border_color="#1f2a44", panel_background_fill="#111a2e",
        panel_border_color="#1f2a44", button_primary_background_fill="#38bdf8",
        button_primary_background_fill_hover="#7dd3fc", button_primary_text_color="#0b1120",
        button_secondary_background_fill="#1e293b", button_secondary_background_fill_hover="#273449",
        button_secondary_text_color="#e2e8f0", link_text_color="#7dd3fc",
    )
    theme = gr.themes.Base(primary_hue="sky", neutral_hue="slate",
                           font=[gr.themes.Font(f) for f in ("Inter", "Segoe UI", "system-ui", "sans-serif")],
                           font_mono=[gr.themes.Font(f) for f in ("ui-monospace", "Consolas", "monospace")])
    accepted = inspect.signature(theme.set).parameters
    both = {**colours, **{k + "_dark": v for k, v in colours.items() if k + "_dark" in accepted}}
    return theme.set(**{k: v for k, v in both.items() if k in accepted})


def _pct(x):
    return f"{100 * x:.2f}%"


def _header_html(verification):
    cls, label = {"verified": ("tsr-ok", "Checkpoint verified"),
                  "failed": ("tsr-bad", "Checkpoint verification FAILED"),
                  "not_checked": ("tsr-warn", "Checkpoint not verified")}[verification["status"]]
    return (
        '<div class="tsr-header">'
        '<div class="tsr-eyebrow">Deep Learning final project</div>'
        '<div class="tsr-title">Cambodian Traffic Sign Recognition</div>'
        '<p class="tsr-subtitle">A deep learning comparison of three approaches for classifying '
        '26 selected Cambodian traffic-sign classes.</p>'
        '<div class="tsr-meta">'
        f'<span class="tsr-status {cls}" title="{html.escape(verification["detail"])}">'
        f'<span class="tsr-dot"></span>{label}: {html.escape(verification["detail"])}</span>'
        '<span>Tes Sothyroth &middot; Kirirom Institute of Technology</span>'
        '</div></div>'
    )


def _summary_html(metrics, comparison):
    if metrics:
        cards = [("Test accuracy", _pct(metrics["accuracy"]), "final model, held-out test set"),
                 ("Macro F1", _pct(metrics["f1_macro"]), "averaged over the 26 classes"),
                 ("Test set", f"{metrics['num_test']} images", "never used for training or tuning"),
                 ("Supported classes", str(NUM_CLASSES), "selected Cambodian sign types")]
    else:
        cards = [("Test accuracy", "n/a", "results file not found"), ("Macro F1", "n/a", "results file not found"),
                 ("Test set", "n/a", "results file not found"),
                 ("Supported classes", str(NUM_CLASSES), "selected Cambodian sign types")]
    out = '<div class="tsr-cards">' + "".join(
        f'<div class="tsr-card"><div class="tsr-card-label">{a}</div><div class="tsr-card-value">{b}</div>'
        f'<div class="tsr-card-sub">{c}</div></div>' for a, b, c in cards) + "</div>"
    out += ('<p class="tsr-note">These are evaluation results on the saved test set from one training run. '
            'They are not a guarantee that every new image will be classified correctly.</p>')

    if comparison:
        # Approaches 1 and 2, plus the final (tuned) version of approach 3 used by this website.
        shown = [comparison[0], comparison[1], comparison[-1]]
        names = ["Small CNN trained from scratch", "ResNet-18 with frozen pretrained backbone",
                 "ResNet-18 with full fine-tuning"]
        out += '<div class="tsr-approaches">'
        for i, (row, name) in enumerate(zip(shown, names)):
            final = i == 2
            out += (f'<div class="tsr-approach{" final" if final else ""}"><div class="tsr-approach-name">'
                    f'{i + 1}. {name}{"<span class=tsr-tag>used here</span>" if final else ""}</div>'
                    f'<div class="tsr-approach-num">Test accuracy {_pct(row["accuracy"])} &middot; '
                    f'macro F1 {_pct(row["f1_macro"])}{" (after tuning)" if final else ""}</div></div>')
        out += "</div>"
    out += ('<p class="tsr-note">This website runs the final fine-tuned ResNet-18 checkpoint '
            '(approach 3 after hyperparameter tuning).</p>')
    return out


EMPTY_RESULT = ('<div class="tsr-empty"><div><strong>No prediction yet</strong></div>'
                '<div style="margin-top:6px">Upload a tightly cropped traffic sign or pick an example, '
                'then press <strong>Classify</strong>.</div></div>')


def _result_html(top3, case=None, heading="Predicted class"):
    name, conf = top3[0]
    out = (f'<div class="tsr-result-label">{html.escape(heading)}</div>'
           f'<div class="tsr-pred">{html.escape(pretty(name))}</div>'
           f'<div class="tsr-code">class code {html.escape(name)}</div>'
           f'<div class="tsr-conf"><span class="tsr-conf-value">{_pct(conf)}</span>'
           '<span class="tsr-conf-label">confidence</span></div>')
    if conf < LOW_CONFIDENCE:
        out += ('<div class="tsr-alert tsr-alert-warn"><strong>Low confidence.</strong> The image may be unclear, '
                'not a tight crop, or not one of the 26 supported classes. Treat this prediction with caution.</div>')
    if case is not None:
        # Ground truth comes from the saved test labels; the comparison uses the live prediction.
        truth = html.escape(pretty(case["true_label"]))
        if name == case["true_label"]:
            out += (f'<div class="tsr-alert tsr-alert-ok"><strong>Correct.</strong> Test-set ground truth: '
                    f'{truth}.</div>')
        else:
            failure = case.get("correct") is False and heading == "Predicted class"
            note = f' Observation: {html.escape(case["note"])}' if failure and case["note"] else ""
            label = "Incorrect (failure analysis example)." if failure else "Incorrect."
            out += (f'<div class="tsr-alert tsr-alert-bad"><strong>{label}</strong> '
                    f'Test-set ground truth: {truth}, but the model predicts {html.escape(pretty(name))}.{note}</div>')
    out += '<div class="tsr-result-label" style="margin-top:20px">Top-3 predictions</div><div class="tsr-bars">'
    for rank, (n, p) in enumerate(top3, 1):
        out += (f'<div class="tsr-bar-row{"" if rank == 1 else " rest"}"><span class="tsr-rank">{rank}</span>'
                f'<div><div class="tsr-bar-name">{html.escape(pretty(n))}</div><div class="tsr-bar-track">'
                f'<div class="tsr-bar-fill" style="width:{max(100 * p, 0.5):.2f}%"></div></div></div>'
                f'<span class="tsr-bar-pct">{_pct(p)}</span></div>')
    out += ('</div><p class="tsr-fine">Confidence is the model\'s softmax probability, not a guarantee that the '
            'prediction is correct. The model always chooses one of its 26 classes; it has no "unknown" class.</p>')
    return out


def _details_html(classifier, verification, metrics):
    cfg = (f"Config {metrics['config']} (learning rate {metrics['lr']:g}, weight decay {metrics['wd']:g})"
           if metrics else "not available")
    test_n = f"{metrics['num_test']} images" if metrics else "not available"
    chips = "".join(f'<span class="tsr-chip">{html.escape(pretty(c))}</span>' for c in CLASS_NAMES)
    return (
        '<div class="tsr-details">'
        '<h4>Model and checkpoint</h4><ul>'
        '<li>ResNet-18 pretrained on ImageNet, fully fine-tuned on Cambodian traffic signs '
        '(all 11,189,850 parameters trainable), with a new 26-class output layer.</li>'
        f'<li>Selected hyperparameters: {cfg}.</li>'
        f'<li>Checkpoint: <code>{html.escape(classifier.checkpoint_path.name)}</code>, saved at epoch '
        f'{classifier.checkpoint.get("epoch")}; running on <code>{classifier.device}</code>.</li>'
        f'<li>Verification: {html.escape(verification["status"].replace("_", " "))}: '
        f'{html.escape(verification["detail"])} (6 misclassified + 18 correctly classified test images, '
        'tolerance 0.001).</li>'
        '<li>Preprocessing: resize to 224 &times; 224 and ImageNet normalisation, identical to the test evaluation.</li>'
        '</ul>'
        '<h4>Data</h4><ul>'
        '<li>Cambodia Traffic Signs Dataset (Kaggle, Sophal Ratitya, Apache 2.0), cropped to single signs.</li>'
        '<li>Train / validation / test split by original photo, so crops of the same photo never appear in '
        f'two splits. Test set: {test_n}.</li></ul>'
        '<h4>Limitations</h4><ul>'
        '<li>Only the 26 selected classes below are supported. Other signs (for example STOP) are not a class, '
        'so the model maps them to the closest supported class.</li>'
        '<li>There is no unknown-class detector: unrelated images still receive a prediction, '
        'sometimes with high confidence.</li>'
        '<li>Small, blurred, occluded or visually similar signs can be misclassified; the final model made '
        '6 errors on the test set, some with confidence above 95%.</li>'
        '<li>The model was trained on tight crops of one sign, not full street scenes.</li>'
        '<li>All results come from a single training run (seed 42) evaluated on one test set.</li></ul>'
        f'<h4>Supported classes ({NUM_CLASSES})</h4><div class="tsr-chips">{chips}</div>'
        '</div>'
    )


EMPTY_DETECTION = ('<div class="tsr-empty" style="min-height:120px"><div><strong>No photo analysed yet</strong></div>'
                   '<div style="margin-top:6px">Upload a street photo or pick an example, then press '
                   '<strong>Detect &amp; Classify</strong>.</div></div>')


def _detection_intro_html(detector, detector_metrics, detector_error):
    if detector is None:
        return ('<div class="tsr-alert tsr-alert-warn"><strong>Detector not loaded.</strong> '
                f'{html.escape(detector_error or "Start the app with --detector path/to/sign_detector_best.pt.")} '
                'The cropped-sign classifier in the first tab still works.</div>')
    out = ('<div class="tsr-alert tsr-alert-warn" style="margin-top:4px"><strong>Two models, two jobs.</strong> '
           'A YOLO detector finds every traffic sign (its only class is "Traffic Sign"); the ResNet-18 classifier '
           'then names each crop as one of its <strong>26 supported classes</strong>. Signs outside those classes '
           '(for example STOP) still receive the closest of the 26 labels, sometimes with high confidence. '
           'Low classifier confidence is a hint to check the box, not proof that the sign is unknown.</div>')
    if detector_metrics:
        d, e = detector_metrics["det"], detector_metrics["e2e"]
        out += (f'<p class="tsr-note">Evaluated once on {e["test_photos"]} held-out test photos (single run): '
                f'detector precision {_pct(d["precision"])}, recall {_pct(d["recall"])}, mAP@50 '
                f'{d["mAP50"]:.3f}; pipeline found {_pct(e["detection_recall_26_classes"])} of the '
                f'{e["test_signs_26_classes"]} test signs of the 26 classes and named '
                f'{_pct(e["end_to_end_accuracy"])} of them correctly. Results on new photos may differ.</p>')
    return out


def _label_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_demo(classifier, verification, metrics=None, examples=None, detector=None, detector_metrics=None,
               det_examples=None, detector_error=None):
    """Builds the Gradio interface (gradio is imported here, so the model code can be used without it).

    `examples` is the list returned by example_images(); plain image paths are also accepted.
    `detector` (a SignDetector) enables the "Detect All Signs in a Street Photo" tab; `det_examples` are
    test-split street photos from detection_examples().
    """
    import gradio as gr

    gradio_major = int(gr.__version__.split(".")[0])
    theme = _theme(gr)

    cases = [e if isinstance(e, dict) else {"path": str(e), "correct": None} for e in (examples or [])]
    # Only show a "correct" example if the live model still reproduces the saved correct prediction.
    for c in cases:
        if c["correct"]:
            (pred, _), *_ = classifier.top_k(Image.open(c["path"]), k=1)
            c["live_ok"] = pred == c["true_label"]
    correct_cases = [c for c in cases if c["correct"] and c.get("live_ok")]
    failure_cases = [c for c in cases if c["correct"] is False]
    other_cases = [c for c in cases if c["correct"] is None]

    def scene_image(c, context=None, width=1280):
        """The original street photo with the annotated sign outlined (bbox from the test manifest).

        context=None: the whole photo. context=k: a square zoom k times the sign size around the sign."""
        scene = Image.open(c["scene"])
        full_width = scene.width
        if context is None:
            scene.draft("RGB", (scene.width // 2, scene.height // 2))  # faster reduced-size JPEG decode
        scale = scene.size[0] / full_width
        scene = scene.convert("RGB")
        x, y, w, h = (v * scale for v in c["bbox"])
        if context:
            side = max(context * max(w, h), 480 * scale)  # at least 480 original pixels of surroundings
            cx, cy = x + w / 2, y + h / 2
            left = min(max(cx - side / 2, 0), max(scene.width - side, 0))
            top = min(max(cy - side / 2, 0), max(scene.height - side, 0))
            scene = scene.crop((int(left), int(top), int(left + side), int(top + side)))
            x, y = x - int(left), y - int(top)
        f = width / scene.width
        scene = scene.resize((width, max(1, int(scene.height * f))))
        pad, line = 4, max(3, width // 160)
        ImageDraw.Draw(scene).rectangle([x * f - pad, y * f - pad, (x + w) * f + pad, (y + h) * f + pad],
                                        outline="#38bdf8", width=line)
        return scene

    def thumb(c):
        return scene_image(c, context=4, width=256) if c.get("scene") else Image.open(c["path"])

    def gallery_items(items, failure=False):
        if failure:
            return [(thumb(c), f"Failure {i}") for i, c in enumerate(items, 1)]
        return [(thumb(c), pretty(c["true_label"]) if c.get("true_label") else "") for c in items]

    def gallery(items, failure=False):
        cols = 3 if failure else 5
        rows = -(-len(items) // cols)
        return gr.Gallery(gallery_items(items, failure), columns=cols, rows=rows,
                          height=16 + (176 if failure else 112) * rows,
                          allow_preview=False, show_label=False, object_fit="cover", elem_classes="tsr-gallery")

    failure_legend = "".join(
        f'<li><strong>Failure {i}:</strong> ground truth {html.escape(pretty(c["true_label"]))}, saved prediction '
        f'{html.escape(pretty(c["saved_pred"]))} ({_pct(c["saved_conf"])}). {html.escape(c["note"])}</li>'
        for i, c in enumerate(failure_cases, 1))

    def classify(image, case):
        if image is None:
            return ('<div class="tsr-alert tsr-alert-warn">Please upload an image (or pick an example) first.</div>'
                    + EMPTY_RESULT)
        return _result_html(classifier.top_k(image, k=3), case if case and case.get("true_label") else None)

    def pick(items):
        def handler(evt: gr.SelectData):
            c = items[evt.index]
            msg = ('<div class="tsr-empty"><div><strong>Example loaded</strong></div>'
                   '<div style="margin-top:6px">Press <strong>Classify</strong> to run the model.</div></div>')
            # The input is the exact test crop (the annotated sign region), as used in the evaluation.
            has_scene = bool(c.get("scene"))
            scene = gr.Image(value=scene_image(c), visible=True) if has_scene else gr.Image(visible=False)
            return (Image.open(c["path"]), c, msg, scene, gr.Button(visible=has_scene),
                    gr.HTML(value="", visible=False))
        return handler

    def classify_whole(case):
        """Classifies the ENTIRE original street photo (no crop), with the same model and preprocessing."""
        if not case or not case.get("scene"):
            return gr.HTML(value="", visible=False)
        scene = Image.open(case["scene"])
        body = _result_html(classifier.top_k(scene, k=3), case,
                            heading=f"Whole uncropped photo ({scene.width}×{scene.height}) predicted class")
        note = ('<p class="tsr-fine">The whole photo is resized to 224 × 224 like any input, so the sign shrinks '
                'to a few pixels. Compare with the cropped-sign result above.</p>')
        return gr.HTML(value='<div class="tsr-whole">' + body + note + '</div>', visible=True)

    has_scenes = any(c.get("scene") for c in cases)

    det_columns = ["#", "Predicted class (ResNet-18)", "Classifier confidence", "Detector confidence",
                   "2nd guess", "3rd guess", "Box x, y, w, h (px)", "Note"]

    def detect_and_classify(photo_path, conf):
        """Detector finds boxes -> each box is cropped from the full-resolution photo -> ResNet-18 names it."""
        empty_table = pd.DataFrame(columns=det_columns)
        if detector is None:
            return _detection_intro_html(None, None, detector_error), None, empty_table, []
        if not photo_path:
            return ('<div class="tsr-alert tsr-alert-warn">Please upload a street photo (or pick an example) '
                    'first.</div>'), None, empty_table, []
        photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")  # respect phone rotation
        dets = detector.detect(photo, conf=conf)

        scale = min(1.0, 1600 / photo.width)
        shown = photo.resize((round(photo.width * scale), round(photo.height * scale))) if scale < 1 else photo.copy()
        draw = ImageDraw.Draw(shown)
        line = max(2, round(shown.width / 400))
        font = _label_font(max(14, round(shown.width / 75)))
        rows, crops = [], []
        for i, ((x1, y1, x2, y2), dconf) in enumerate(dets, 1):
            crop = photo.crop((int(x1), int(y1), int(x2), int(y2)))
            top3 = classifier.top_k(crop, k=3)          # the 26-class prediction always comes from ResNet-18
            (name, cconf), second, third = top3
            low = cconf < LOW_CONFIDENCE
            note = ("Low classifier confidence: possibly an unclear sign, a sign outside the 26 classes, "
                    "or a false detection" if low else "")
            rows.append([i, pretty(name), _pct(cconf), _pct(dconf), f"{pretty(second[0])} ({_pct(second[1])})",
                         f"{pretty(third[0])} ({_pct(third[1])})",
                         f"{int(x1)}, {int(y1)}, {int(x2 - x1)}, {int(y2 - y1)}", note])
            crops.append((crop, f"#{i} {pretty(name)} {_pct(cconf)}"))
            colour = "#fbbf24" if low else "#38bdf8"
            bx = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
            draw.rectangle(bx, outline=colour, width=line)
            label = f"#{i} {pretty(name)} {100 * cconf:.0f}%"
            tx, ty = bx[0], max(0, bx[1] - font.size - 2 * line)
            tb = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=colour)
            draw.text((tx, ty), label, fill="#0b1120", font=font)

        if not dets:
            summary = (f'<div class="tsr-alert tsr-alert-warn"><strong>No traffic signs detected</strong> at '
                       f'detector threshold {conf:.2f}. Try lowering the threshold. The detector was trained on '
                       'Cambodian road footage, so very small, distant, unusual or non-Cambodian signs may be '
                       'missed. The photo is shown unchanged.</div>')
        else:
            n_low = sum(1 for r in rows if r[-1])
            summary = (f'<div class="tsr-result-label">Result</div><div class="tsr-pred" style="font-size:26px">'
                       f'{len(dets)} sign{"s" if len(dets) != 1 else ""} detected</div>'
                       f'<p class="tsr-help" style="margin-top:6px">Detector threshold {conf:.2f}. Blue box: '
                       f'classifier confidence at least {LOW_CONFIDENCE:.0%}; amber box: lower '
                       f'({n_low} of {len(dets)}). Each label is ResNet-18\'s best match among its 26 classes and '
                       'is not checked against any ground truth here.</p>')
        return summary, shown, pd.DataFrame(rows, columns=det_columns), crops

    blocks_kwargs = {"theme": theme, "css": CSS} if gradio_major < 6 else {}
    with gr.Blocks(title="Cambodian Traffic Sign Recognition", **blocks_kwargs) as demo:
        case_state = gr.State(None)
        gr.HTML(_header_html(verification))
        gr.HTML(_summary_html(metrics, load_comparison()))

        with gr.Tabs():
            with gr.Tab("Classify a Cropped Sign"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=260, elem_classes="tsr-panel"):
                        gr.HTML('<div class="tsr-section-title">1. Input image</div>'
                                '<p class="tsr-help">Upload a <strong>tight crop containing one traffic sign</strong>. '
                                'Full street photos or several signs in one image reduce accuracy.</p>')
                        # PNG is lossless: Gradio's default (WebP) re-encoding would alter pixels and confidences.
                        image_in = gr.Image(type="pil", format="png", label="Traffic-sign image", height=300)
                        with gr.Row():
                            classify_btn = gr.Button("Classify", variant="primary", scale=3, min_width=120,
                                                     elem_classes="tsr-classify")
                            clear_btn = gr.Button("Clear", variant="secondary", scale=1, min_width=80)
                        scene_view = gr.Image(label="Where this sign is in the original street photo "
                                                    "(box from the dataset annotation)",
                                              interactive=False, visible=False, height=240, buttons=["fullscreen"])
                        whole_btn = gr.Button("Classify the whole uncropped photo", variant="secondary", visible=False)
                        galleries = []
                        if correct_cases:
                            if has_scenes:
                                gr.HTML(f'<div class="tsr-gallery-label">Real street scenes from the test set: one per '
                                        f'sign ({len(correct_cases)} classes)</div>'
                                        '<p class="tsr-help" style="margin-top:4px">Click a scene: the outlined sign is '
                                        'cropped and placed in the input, exactly as in the test evaluation. The box comes '
                                        'from the dataset annotation; <strong>the model does not detect signs itself</strong>. '
                                        'Classifying a whole street photo mostly fails, because the sign covers well under '
                                        '1% of the photo. Each example is the median-confidence correct test image of its '
                                        'class, checked against the saved ground truth.</p>')
                            else:
                                gr.HTML(f'<div class="tsr-gallery-label">Test-set examples: one per class '
                                        f'({len(correct_cases)} classes), correctly classified, typical confidence</div>'
                                        '<p class="tsr-help" style="margin-top:4px">Each is the median-confidence correct '
                                        'test image of its class, checked against the saved ground truth.</p>')
                            galleries.append((gallery(correct_cases), correct_cases))
                        if failure_cases:
                            gr.HTML('<div class="tsr-gallery-label fail">Failure analysis examples: misclassified '
                                    'test images</div>')
                            galleries.append((gallery(failure_cases, failure=True), failure_cases))
                            gr.HTML(f'<ul class="tsr-legend">{failure_legend}</ul>')
                        if other_cases:
                            galleries.append((gallery(other_cases), other_cases))
                    with gr.Column(scale=5, min_width=260, elem_classes="tsr-panel"):
                        gr.HTML('<div class="tsr-section-title">2. Prediction</div>'
                                '<p class="tsr-help">Final fine-tuned ResNet-18, 26 supported classes.</p>')
                        result = gr.HTML(EMPTY_RESULT)
                        result_whole = gr.HTML("", visible=False)

            with gr.Tab("Detect All Signs in a Street Photo"):
                gr.HTML(_detection_intro_html(detector, detector_metrics, detector_error))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=260, elem_classes="tsr-panel"):
                        gr.HTML('<div class="tsr-section-title">1. Street photo</div>'
                                '<p class="tsr-help">Upload a full road or street photo. The original file is used '
                                'at full resolution; every detected sign is cropped from it and classified.</p>')
                        # PNG is lossless: Gradio's default (WebP) re-encoding would alter the uploaded photo.
                        photo_in = gr.Image(type="filepath", format="png", label="Street photo", height=320)
                        det_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                             label="Detector confidence threshold (0.25 was used in the evaluation)")
                        with gr.Row():
                            detect_btn = gr.Button("Detect & Classify", variant="primary", scale=3, min_width=120,
                                                   elem_classes="tsr-classify", interactive=detector is not None)
                            det_clear_btn = gr.Button("Clear", variant="secondary", scale=1, min_width=80)
                        if det_examples:
                            gr.Examples(examples=[[p] for p in det_examples], inputs=photo_in,
                                        label="Example street photos from the TEST split (never used for "
                                              "training either model)")
                    with gr.Column(scale=5, min_width=260, elem_classes="tsr-panel"):
                        gr.HTML('<div class="tsr-section-title">2. Detected signs</div>'
                                '<p class="tsr-help">Box = YOLO detector ("a traffic sign is here"). '
                                'Label = ResNet-18 classifier (which of the 26 classes).</p>')
                        det_summary = gr.HTML(EMPTY_DETECTION)
                        det_image = gr.Image(label="Detections", interactive=False, height=360,
                                             buttons=["fullscreen", "download"])
                det_table = gr.Dataframe(label="Detections (numbered left to right, as in the image)",
                                         interactive=False, wrap=True)
                det_crops = gr.Gallery(label="Crops sent to the classifier", columns=6, height="auto",
                                       allow_preview=True, object_fit="contain")

        with gr.Accordion("Model Details & Limitations", open=False):
            gr.HTML(_details_html(classifier, verification, metrics))
        gr.HTML('<div class="tsr-footer">Cambodian Traffic Sign Recognition &middot; Tes Sothyroth &middot; '
                'Kirirom Institute of Technology &middot; Built with PyTorch and Gradio</div>')

        def reset_scene():
            return gr.Image(value=None, visible=False), gr.Button(visible=False), gr.HTML(value="", visible=False)

        scene_outputs = [scene_view, whole_btn, result_whole]
        classify_btn.click(classify, inputs=[image_in, case_state], outputs=result)
        whole_btn.click(classify_whole, inputs=case_state, outputs=result_whole)
        clear_btn.click(lambda: (None, None, EMPTY_RESULT, *reset_scene()),
                        outputs=[image_in, case_state, result, *scene_outputs])
        # A user upload replaces any example, so its ground truth and street scene no longer apply.
        image_in.upload(lambda: (None, *reset_scene()), outputs=[case_state, *scene_outputs])
        image_in.clear(lambda: (None, EMPTY_RESULT, *reset_scene()), outputs=[case_state, result, *scene_outputs])
        for gallery, items in galleries:
            gallery.select(pick(items), outputs=[image_in, case_state, result, *scene_outputs])

        det_outputs = [det_summary, det_image, det_table, det_crops]
        detect_btn.click(detect_and_classify, inputs=[photo_in, det_conf], outputs=det_outputs)
        det_clear_btn.click(lambda: (None, EMPTY_DETECTION, None, pd.DataFrame(columns=det_columns), []),
                            outputs=[photo_in, *det_outputs])

    if gradio_major >= 6:
        # Gradio 6 takes theme and css in launch(); apply them by default for every caller (also notebook 08).
        plain_launch = demo.launch

        def styled_launch(*args, **kwargs):
            kwargs.setdefault("theme", theme)
            kwargs.setdefault("css", CSS)
            return plain_launch(*args, **kwargs)

        demo.launch = styled_launch
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH", str(PROJECT_ROOT / CHECKPOINT_NAME)))
    parser.add_argument("--processed-dir", default=os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "dataset" / "processed")))
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--detector", default=os.environ.get("DETECTOR_PATH"),
                        help="trained 1-class YOLO weights (enables the street-photo detection tab)")
    parser.add_argument("--detector-metrics", default=os.environ.get("DETECTOR_METRICS"),
                        help="detector_and_pipeline_metrics.json from scripts/train_sign_detector.py")
    parser.add_argument("--raw-dir", default=os.environ.get("RAW_DIR", str(PROJECT_ROOT / "dataset" / "raw" / "images")),
                        help="original street photos (test-split examples for the detection tab)")
    args = parser.parse_args()

    classifier = SignClassifier(args.checkpoint)
    verification = verify_checkpoint(classifier, args.processed_dir)
    print(f"Checkpoint verification: {verification['status'].upper()}: {verification['detail']}")
    detector, detector_error = None, None
    if args.detector:
        try:
            detector = SignDetector(args.detector)
            print(f"Detector loaded: {detector.weights_path} (classes: {detector.model.names})")
        except Exception as e:  # the classification tab must keep working without the detector
            detector_error = f"{type(e).__name__}: {e}"
            print(f"Detector NOT loaded: {detector_error}")
    else:
        detector_error = "No --detector weights were given."
    demo = build_demo(classifier, verification, load_reported_metrics(), example_images(args.processed_dir),
                      detector=detector, detector_metrics=load_detector_metrics(args.detector_metrics),
                      det_examples=detection_examples(args.raw_dir) if detector else [],
                      detector_error=detector_error)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
