# Road Sign Recognition Using Deep Learning

## 1. Project Overview

This project develops and compares deep learning approaches for recognizing Cambodian traffic signs from images.

**Task:** Image Classification  
**Dataset:** Cambodia Traffic Signs Dataset (CamTSD)  
**Number of Classes:** 26

## 2. Problem Statement

Traffic sign recognition is an important computer vision task for intelligent transportation systems. This project investigates deep learning methods for automatically classifying Cambodian traffic signs from images.

## 3. Dataset

The processed classification dataset contains:

- 3,070 images
- 26 traffic sign classes
- Training set: 2,149 images
- Validation set: 460 images
- Test set: 461 images

The dataset is divided using a fixed stratified split with a random seed of 42.

## 4. Data Preprocessing

Images are resized to **224 × 224 pixels**.

Training augmentation includes:

- Random rotation
- Color jitter
- ImageNet normalization

Horizontal flipping is not used because flipping traffic signs may change their meaning, particularly for directional signs.

## 5. Deep Learning Approaches

### 5.1 CNN from Scratch

A convolutional neural network is trained from scratch without pretrained weights.

### 5.2 ResNet-18 with Frozen Backbone

A pretrained ResNet-18 model is used while keeping the backbone frozen. Only the classification layer is trained for the traffic sign classes.

### 5.3 ResNet-18 with Full Fine-Tuning

A pretrained ResNet-18 model is used and the entire network is fine-tuned on the traffic sign dataset.

## 6. Experimental Setup

- Framework: PyTorch
- Python: Python 3
- GPU: NVIDIA Tesla T4
- Image size: 224 × 224
- Batch size: 32
- Random seed: 42

## 7. Results

Results will be added after training and evaluation.

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---:|---:|---:|---:|---:|
| CNN from Scratch | TBD | TBD | TBD | TBD | TBD |
| ResNet-18 Frozen | TBD | TBD | TBD | TBD | TBD |
| ResNet-18 Fine-Tuned | TBD | TBD | TBD | TBD | TBD |

## 8. Error Analysis

Error analysis will examine misclassified traffic signs using the confusion matrix and incorrectly classified test images.

## 9. Discussion

The three approaches will be compared based on classification performance, training efficiency, parameter count, and generalization performance.

## 10. Limitations

Potential limitations include class imbalance, limited dataset size for some traffic sign categories, and visual similarity between certain traffic sign classes.

## 11. Conclusion

This project compares multiple deep learning approaches for Cambodian traffic sign classification and evaluates their performance using a common held-out test set.

## 12. How to Run

1. Open the provided Google Colab notebook.
2. Mount Google Drive.
3. Prepare the processed dataset.
4. Create the fixed train/validation/test split.
5. Train each model.
6. Evaluate all models using the same test set.

## 13. References

- Cambodia Traffic Signs Dataset (CamTSD)
- He, K. et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- PyTorch Documentation

## 14. AI Use Disclosure

AI tools were used to assist with code development, debugging, documentation, and project organization. The final implementation, experiments, results, and analysis were reviewed by the project author.
