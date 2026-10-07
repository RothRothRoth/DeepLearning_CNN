# Final Model Error Analysis

**Model:** ResNet-18 pretrained on ImageNet, fully fine-tuned and hyperparameter-tuned (Config B: learning rate 0.001, weight decay 0.001)  
**Test set:** 463 crops from the fixed, leak-free test split (never used for training or model selection)  
**Source of results:** `results/runs/hyperparameter_tuning/resnet18_finetuned_lr_wd_seed42/` produced by `notebooks/05_hyperparameter_tuning.ipynb`, specifically `metrics/final_metrics.json` and `final_tuned_model/{test_predictions,classification_report,confusion_matrix}.csv`

## Final test metrics

| Metric | Value |
|---|---:|
| Accuracy | 98.70% (457 / 463 correct) |
| Macro precision | 98.72% |
| Macro recall | 98.36% |
| Macro F1 | 98.47% |
| Weighted F1 | 98.69% |

## Facts

- **457 test images were classified correctly and 6 incorrectly.**
- **16 of the 26 classes have perfect precision, recall and F1 (1.000)**, among them PEDESTRAIN_CR_AREA (32 test images), DIRECTION (28), ROAD_JUNCTION_ON_THE_RIGHT (28), NO_UTURN (21) and all four speed-limit classes.
- **Lowest-F1 classes:**

  | Class | Precision | Recall | F1 | Test images | Training crops |
  |---|---:|---:|---:|---:|---:|
  | NO_ENTRY | 1.000 | 0.857 | 0.923 | 7 | 41 |
  | RIGHT_BEND | 0.875 | 1.000 | 0.933 | 7 | 33 |
  | LEFT_BEND | 1.000 | 0.889 | 0.941 | 9 | 40 |
  | PRIORITY_ROAD | 1.000 | 0.889 | 0.941 | 18 | 75 |

- **No confusion occurs more than once.** All six errors involve a different (true → predicted) pair. PRIORITY_ROAD is the only class with two errors.
- **The six misclassified images** are listed in [`misclassified_test_images.csv`](misclassified_test_images.csv) and shown in [`misclassified_gallery.png`](misclassified_gallery.png):

  | # | True label | Predicted | Confidence | Category |
  |---|---|---|---:|---|
  | 1 | LEFT_BEND | RIGHT_BEND | 1.00 | probable annotation error |
  | 2 | KM_POST | KEEP_RIGHT | 1.00 | probable annotation error |
  | 3 | PEDESTRAIN_CROSSING | CHILDREN_CROSSING | 0.96 | probable annotation error |
  | 4 | PRIORITY_ROAD | CROSS_ROAD | 0.84 | model error |
  | 5 | NO_ENTRY | SLOW_DOWN | 0.32 | model error |
  | 6 | PRIORITY_ROAD | KM_POST | 0.25 | model error |

## Probable annotation errors (examples 1–3)

**These examples appear to be annotation errors based on visual inspection and comparison with the original dataset annotations.**

- In each case the test label matches the label in the original CamTSD annotation file (`CAM_TSR_v3_json.json`). These labels therefore come from the dataset itself, not from this project's processing.
- What the images show:
  - **Example 1**, annotated "LEFT BEND": the arrow curves to the right.
  - **Example 2**, annotated "KM POST": a blue round sign with a white arrow pointing down-right.
  - **Example 3**, annotated "PEDESTRAIN CROSSING": two figures, an adult and a child, walking together.
- In all three cases the model's prediction matches the visible sign.
- [`annotation_error_context.png`](annotation_error_context.png) shows each annotated box in its original source photo.

This finding concerns only these three test images. Only the six misclassified test images were inspected, so no conclusion is drawn about annotation quality in the rest of the dataset.

The reported metrics are **not** adjusted for these examples. The official result remains 98.70% accuracy and 98.47% macro F1.

## Model errors (examples 4–6)

These are observations from the images; the stated reasons are interpretations.

- **Example 4 (PRIORITY_ROAD → CROSS_ROAD, confidence 0.84):** a clear image. Both classes use a crossing-roads symbol, so this is plausibly a confusion between visually similar signs.
- **Example 5 (NO_ENTRY → SLOW_DOWN, confidence 0.32):** the crop is only 39 × 39 pixels, smaller than 98% of test crops, and very blurry.
- **Example 6 (PRIORITY_ROAD → KM_POST, confidence 0.25):** the sign is largely covered by tree branches and leaves.
- **Interpretation:** the remaining genuine errors are mostly low-confidence predictions on very small or occluded signs.

## Class imbalance

- **Facts:**
  - The training set ranges from 523 crops (KM_POST) down to 20 (CARRIAGE_WAY_NARROWS).
  - The lowest-F1 classes have only 7–9 test images. With 7 test images, a single error lowers recall to 85.7%.
  - Some of the smallest classes are classified perfectly: CARRIAGE_WAY_NARROWS (20 training crops, 5 test images) and 30_SPEED_LIMIT (21 training crops, 3 test images).
- **Interpretation:** for the final model, imbalance does not appear as a systematic failure on rare classes. Rather, per-class scores for small classes are fragile, because one image changes them substantially.

## Files in this folder

| File | Content |
|---|---|
| `misclassified_test_images.csv` | The six misclassified test images: path, true and predicted label, confidence, original annotation, crop size, category, visual observation |
| `misclassified_gallery.png` | The six misclassified crops |
| `annotation_error_context.png` | Examples 1–3 in their original source photos |
| Confusion matrix | Not stored in this folder. It is generated by `notebooks/05_hyperparameter_tuning.ipynb` and stored with the experiment outputs in `results/runs/hyperparameter_tuning/resnet18_finetuned_lr_wd_seed42/` (Google Drive): `final_tuned_model/confusion_matrix.csv`, `figures/final_confusion_matrix.png`, `figures/final_confusion_matrix_normalized.png` |
