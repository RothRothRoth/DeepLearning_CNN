# Road Sign Recognition Using Deep Learning

## 1. Project Overview

This project develops and compares deep learning approaches for recognising Cambodian traffic signs.

**Task:** Image classification (one cropped traffic sign → one of 26 classes)  
**Dataset:** Cambodia Traffic Signs Dataset (CamTSD)  
**Framework:** PyTorch  
**Final model:** ResNet-18 pretrained on ImageNet, fully fine-tuned and hyperparameter-tuned. It reaches **98.70% test accuracy** and **98.47% macro F1** on a held-out test set of 463 sign crops.

Three approaches of increasing transfer-learning strength are compared on exactly the same data split:

1. A small CNN trained from scratch
2. ResNet-18 with a frozen ImageNet-pretrained backbone
3. ResNet-18 with full fine-tuning, which is then hyperparameter-tuned

## 2. Problem Statement

Traffic sign recognition is a core computer-vision task for driver assistance and intelligent transportation systems. Cambodian traffic signs include country-specific classes, such as kilometre posts with Khmer text, alongside several look-alike junction and bend signs. The available data is also small and strongly imbalanced.

The goal is to find out how much each level of transfer learning helps on this small, imbalanced dataset, measured fairly on a test set that is never used for model selection.

## 3. Dataset and Source

**Source:** Cambodia Traffic Signs Dataset (CamTSD), obtained from Kaggle (dataset `cambodia-traffic-signs-dataset`). The raw release contains 2,766 street-scene images and 3,490 bounding-box sign annotations in VIA JSON format (`CAM_TSR_v3_json.json`).

**From raw annotations to classification crops:**

- Only the 26 target classes are kept. **`UNDEFINED` annotations (117 in the raw data) are excluded.**
- Each remaining bounding box is cropped from its source photo, giving **3,070 sign crops** from **2,523 source images**.
- Classes are strongly imbalanced, from 738 crops (`KM_POST`) down to 29 crops (`CARRIAGE_WAY_NARROWS`).

**Leak-free split at source-image level.** Many photos contain more than one sign: 627 raw images contain multiple annotated signs. If crops were split randomly, two crops from the *same photo* (same lighting, camera and background) could end up in both training and test data, which inflates test scores.

To prevent this, **the split is made on the original source images, before cropping**. Every crop then goes to the split of the photo it came from. The 70 / 15 / 15 split uses seed 42:

| Split | Crops | Source images |
|---|---:|---:|
| Train | 2,153 | 1,766 |
| Validation | 454 | 378 |
| Test | 463 | 379 |
| **Total** | **3,070** | **2,523** |

- **Zero parent-image overlap** between train, validation and test (verified).
- Exactly 26 classes, no `UNDEFINED` class.
- The fixed split is committed as manifests in `dataset/splits/{train,val,test}.csv`. They use portable relative paths such as `train/KM_POST/<crop>.jpg`, so every experiment uses the identical split on any machine.

The raw images and the cropped dataset are not stored in this repository. They are kept on Google Drive (see Section 10).

## 4. Data Preprocessing

Implemented in `src/data/transforms.py` and shared by all approaches:

- **Training:**
  - resize to **224 × 224**
  - random rotation (±10°)
  - colour jitter (brightness 0.15, contrast 0.15, saturation 0.10)
  - ImageNet normalisation
- **Validation and test:** resize to 224 × 224 and ImageNet normalisation only. This is deterministic and identical for every approach.
- **No horizontal flipping:** flipping can change a sign's meaning, e.g. left bend vs. right bend, or junction on the left vs. on the right.

## 5. Deep Learning Approaches

All approaches use the same split, preprocessing and training protocol:
- seed 42 and batch size 32
- cross-entropy loss and the Adam optimizer
- up to 40 epochs, with `ReduceLROnPlateau` on validation accuracy and early stopping after 10 epochs without improvement
- the **best-validation checkpoint** is kept, and the test set is evaluated **once**, after the model is selected

| Approach | Notebook | Model code |
|---|---|---|
| 1. CNN from scratch | `notebooks/02_cnn_from_scratch.ipynb` | `src/models/cnn.py` → `SmallCNN` |
| 2. ResNet-18 frozen backbone | `notebooks/03_resnet18_frozen.ipynb` | `src/models/resnet.py` → `create_resnet18_frozen` |
| 3. ResNet-18 fully fine-tuned | `notebooks/04_resnet18_finetuned.ipynb` | `src/models/resnet.py` → `create_resnet18_finetuned` |

### 5.1 Small CNN from Scratch
- **Architecture:** three convolutional blocks (Conv → BatchNorm → ReLU → MaxPool, with 32 / 64 / 128 channels), then global average pooling and a 256-unit fully connected layer with dropout 0.5, leading to 26 outputs.
- **No pretrained weights:** every one of the **133,178 parameters** is learned from the training split.

### 5.2 ResNet-18 with Frozen Backbone
- **Starting point:** ResNet-18 with ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`).
- **New classifier:** the 1000-class layer is replaced by `Linear(512, 26)`.
- **What trains:** only that layer (**13,338 trainable parameters**). The backbone acts as a fixed feature extractor, and its BatchNorm layers stay in evaluation mode so the ImageNet statistics are preserved.

### 5.3 ResNet-18 with Full Fine-Tuning
- **Starting point:** the same ImageNet weights and the same new `Linear(512, 26)` classifier.
- **What trains:** **all 11,189,850 parameters**, so the convolutional features themselves adapt to Cambodian traffic signs.

## 6. Hyperparameter Tuning

Notebook: `notebooks/05_hyperparameter_tuning.ipynb`. Tuning was applied to the best baseline, Approach 3.

**Search space (a 2 × 2 grid, everything else fixed):**

| Config | Learning rate | Weight decay (regularisation) |
|---|---:|---:|
| A | 1e-3 | 1e-4 (baseline settings) |
| **B** | **1e-3** | **1e-3** |
| C | 1e-4 | 1e-4 |
| D | 1e-4 | 1e-3 |

**Protocol:**
- Every configuration was trained on the training set and compared **only on validation accuracy**. Ties were broken by validation loss, using a rule fixed before the runs.
- **The test set was not used during tuning.** It was evaluated once, for the selected configuration only, after the selection was made.

**Selected configuration: Config B** (learning rate 0.001, weight decay 0.001), with a best validation accuracy of **99.78%**. Compared with the baseline settings, stronger weight decay (more regularisation) gave the selected model.

## 7. Final Results

All numbers are on the same held-out test set of **463 crops**, which was never used for training or model selection.

| Approach | Test Accuracy | Macro F1 | Trainable Parameters | Training Time |
|---|---:|---:|---:|---:|
| 1. Small CNN from scratch | 70.19% | 50.78% | 133,178 | 8.1 min |
| 2. ResNet-18, frozen backbone | 92.44% | 88.37% | 13,338 | 5.5 min |
| 3. ResNet-18, fully fine-tuned (**untuned baseline**, lr 1e-3, wd 1e-4) | 98.06% | 96.80% | 11,189,850 | 4.0 min |
| **3. ResNet-18, fully fine-tuned + tuned (FINAL MODEL, lr 1e-3, wd 1e-3)** | **98.70%** | **98.47%** | **11,189,850** | — |

**Final tuned model, all test metrics:**

| Metric | Value |
|---|---:|
| Test accuracy | 98.70% |
| Macro precision | 98.72% |
| Macro recall | 98.65% |
| **Macro F1** | **98.47%** |
| Weighted F1 | 98.69% |

**Baseline vs. tuned (Approach 3):**
- **Test accuracy:** 98.06% (untuned baseline) → **98.70%** (tuned final model), **+0.65 percentage points**. That is 454 → 457 of 463 test crops correct.
- **Macro F1:** 96.80% → **98.47%**, **+1.67 percentage points**.

Macro F1 is the mean of the 26 per-class F1 scores, so every class counts equally regardless of size. Macro precision and macro recall are averaged the same way.

**Discussion:**
- **Transfer learning is the largest factor.** Moving from a CNN trained from scratch to frozen ImageNet features raises test accuracy from 70.19% to 92.44%.
- **Fine-tuning adds a further large gain**, mostly on the rarer classes: macro F1 rises from 88.37% to 96.80%, because the network can adapt all of its features to this domain.
- **Tuning adds a smaller, final improvement.**
- The frozen model is the most parameter-efficient, reaching 92.44% with only 13,338 trainable parameters.

The side-by-side table and comparison figures are produced by `notebooks/06_final_comparison.ipynb`.

## 8. Error Analysis

Each experiment notebook saves a full error analysis next to its results:
- normalised and raw confusion matrices
- per-class precision, recall and F1, ranked worst-first
- the most frequent true → predicted confusions
- a gallery of misclassified test crops, most confident mistakes first

Observations from the results:
- **Class imbalance hurts the model trained from scratch most.** Its macro F1 (50.78%) is far below its accuracy (70.19%), so it does well on large classes such as `KM_POST` but poorly on rare ones.
- **Pretraining largely closes that gap.** For the final model, macro F1 (98.47%) is close to accuracy (98.70%).
- **The final model misclassifies 6 of the 463 test crops.** Its confusion matrix and misclassified gallery (`notebooks/05_hyperparameter_tuning.ipynb`) show which classes remain hard.

## 9. Limitations

- **Small, imbalanced dataset:** 3,070 crops in total, and some classes have fewer than 40 crops (as few as 29). The test set contains only 3–10 crops for several classes, so their per-class scores are unreliable.
- **Small test set:** with 463 test crops, one image is about 0.22 percentage points, so small differences between the strongest models (e.g. +0.65 pp from tuning) amount to only a few images.
- **Single split and seed:** all results come from one fixed split and seed 42. No repeated runs or cross-validation, so variance across seeds is not measured.
- **Classification of given crops only:** the models classify already-cropped signs. Detecting signs in full street scenes is not addressed.
- **Limited tuning:** only learning rate and weight decay were tuned, over a 2 × 2 grid.
- **Visually similar classes:** several classes differ only in small details (left vs. right junction and bend signs, the 30 / 40 / 60 / 80 speed limits), which makes them inherently harder to separate.

## 10. How to Run

The experiments are designed for **Google Colab with a GPU** (Runtime → Change runtime type → T4 GPU). The cropped dataset is read from Google Drive:

```
MyDrive/Road_Sign_Recognition_CNN/dataset/processed/
├── train/<CLASS>/*.jpg
├── val/<CLASS>/*.jpg
└── test/<CLASS>/*.jpg
```

Each notebook mounts Drive, clones this repository, and checks the split against the committed manifests (2,153 / 454 / 463, 26 classes, zero source-image overlap) before doing anything else. Open each one in Colab with `https://colab.research.google.com/github/RothRothRoth/DeepLearning_CNN/blob/main/notebooks/<notebook>.ipynb` and run all cells, in this order:

1. `01_data_preparation.ipynb`: builds the leak-free split and crops from the raw dataset. **Not needed** to reproduce the experiments, because the split manifests are already committed.
2. `02_cnn_from_scratch.ipynb`: Approach 1.
3. `03_resnet18_frozen.ipynb`: Approach 2.
4. `04_resnet18_finetuned.ipynb`: Approach 3 (baseline).
5. `05_hyperparameter_tuning.ipynb`: tunes Approach 3, selects on validation, then evaluates the final model on test once. If the Colab session disconnects, re-running it resumes from finished configurations.
6. `06_final_comparison.ipynb`: reads all saved results (no training), then produces the final table and comparison figures.

Results, checkpoints and figures are written to `MyDrive/Road_Sign_Recognition_CNN/results/runs/<experiment>/`.

**Local sanity tests** (no dataset needed):

```bash
python -m tests.test_cnn
```

```bash
python -m tests.test_training
```

```bash
python -m tests.test_resnet
```

These need PyTorch, torchvision, NumPy, pandas, scikit-learn, Matplotlib and Pillow (`requirements.txt` is currently empty; Colab provides these packages).

## 11. Repository Structure

```
DeepLearning_CNN/
├── README.md
├── Road_Sign_Recognition_CNN.ipynb      # original project notebook (overview, dataset exploration)
├── dataset/
│   └── splits/                          # committed leak-free split manifests
│       ├── train.csv  val.csv  test.csv
│       ├── train.txt  val.txt  test.txt
│       └── split_summary.json           # integrity checks (sizes, zero overlap, 26 classes)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_cnn_from_scratch.ipynb        # Approach 1
│   ├── 03_resnet18_frozen.ipynb         # Approach 2
│   ├── 04_resnet18_finetuned.ipynb      # Approach 3
│   ├── 05_hyperparameter_tuning.ipynb   # tuning of Approach 3 + final test evaluation
│   └── 06_final_comparison.ipynb        # final comparison table and figures
├── src/
│   ├── config.py                        # 26 classes, seed, image size, paths
│   ├── seed.py                          # reproducibility (Python / NumPy / PyTorch seeds)
│   ├── data/                            # split creation, dataset, DataLoaders, transforms
│   ├── models/                          # SmallCNN, ResNet-18 frozen / fine-tuned
│   └── training/                        # shared training loop and checkpointing
├── tests/                               # sanity tests for models and training loop
├── results/                             # local output folders (experiment outputs are gitignored)
└── docs/  slides/
```

The repository also contains two empty placeholders, `notebooks/05_final_comparison.ipynb` and `src/evaluation/`, and an older scratch notebook, `Copy_of_Road_Sign_Recognition_CNN.ipynb`. Evaluation is implemented inside the experiment notebooks.

## 12. AI Use Disclosure

AI tools were used to assist with code development, debugging, documentation and project organisation, including Claude Code (Anthropic). The final implementation, experiments, results and analysis were reviewed by the project author.

## 13. References

- Cambodia Traffic Signs Dataset (CamTSD). Kaggle, dataset `cambodia-traffic-signs-dataset`.
- K. He, X. Zhang, S. Ren and J. Sun. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei. "ImageNet: A Large-Scale Hierarchical Image Database." *CVPR*, 2009.
- D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." *International Conference on Learning Representations (ICLR)*, 2015.
- S. Ioffe and C. Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *International Conference on Machine Learning (ICML)*, 2015.
- A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
- F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2011.
