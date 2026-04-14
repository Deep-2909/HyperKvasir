# 🏥 Kvasir GI Tract Image Classification with Stacking Ensembles

A deep learning pipeline for classifying gastrointestinal (GI) tract images from the **Kvasir dataset** using transfer learning, two-stage fine-tuning, and stacking ensemble methods with multiple meta-learners.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Base Models (Level 1)](#base-models-level-1)
- [Fine-Tuning Strategy](#fine-tuning-strategy)
- [Ensemble Methods](#ensemble-methods)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## 🔬 Overview

This project tackles multi-class classification of GI tract endoscopy images. Instead of relying on a single deep learning model, it employs a **stacking ensemble** approach:

1. **Level 1 — Base Learners:** Five state-of-the-art CNN architectures are independently fine-tuned on the Kvasir dataset using a two-stage warmup + fine-tune strategy.
2. **Level 2 — Meta-Learners:** The softmax probability outputs of all five base models are concatenated into a meta-feature vector and used to train several classical and gradient-boosted meta-learners, which learn the optimal way to combine the base predictions.

This two-level stacking consistently outperforms any single base model by leveraging the complementary strengths of different architectures.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  ResNet-50   │ │   VGG-16     │ │ MobileNetV2  │  ...
   │ (fine-tuned) │ │ (fine-tuned) │ │ (fine-tuned) │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                │
          ▼                ▼                ▼
   ┌──────────────────────────────────────────────────┐
   │     Softmax Probabilities  (N classes each)      │
   │     Concatenated → Meta-Feature Vector           │
   │     (5 models × N classes = 5N features)         │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │           META-LEARNER  (Level 2)                │
   │  Logistic Regression │ SVM │ KNN │ MLP │ XGBoost │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
                   FINAL PREDICTION
```

---

## 📁 Dataset

| Property | Detail |
|---|---|
| **Name** | Kvasir / Labeled GI Tract Images |
| **Format** | JPEG images organized in class-labeled subdirectories |
| **Split** | 80% Train · 10% Validation · 10% Test (stratified) |
| **Source** | [Kvasir Dataset](https://datasets.simula.no/kvasir/) |

The dataset is loaded from Google Drive and unzipped into the Colab runtime. Images are organized into subdirectories where each folder name represents a class label.

---

## 🧠 Base Models (Level 1)

| # | Model | ImageNet Weights | Unfrozen Layers (Fine-Tune) |
|---|---|---|---|
| 1 | **ResNet-50** | ✅ | `layer4` |
| 2 | **VGG-16** | ✅ | `features[24:]` (last conv block) |
| 3 | **MobileNetV2** | ✅ | `features[14:]` (last few blocks) |
| 4 | **DenseNet-121** | ✅ | `denseblock4` |
| 5 | **EfficientNet-B6** | ✅ | `features[8]` (last feature block) |

All models use ImageNet-pretrained weights and have their final classifier layer replaced to match the number of target classes.

---

## 🔧 Fine-Tuning Strategy

Each base model follows a **two-stage training protocol**:

### Stage 1 — Warmup (5–10 epochs)
- All backbone layers are **frozen**.
- Only the newly added classifier head is trained.
- **Learning rate:** `1e-3` (Adam optimizer).
- Goal: Initialize the classifier without corrupting pretrained features.

### Stage 2 — Fine-Tune (25 epochs)
- The **last convolutional block** of the backbone is unfrozen.
- **Discriminative learning rates** are applied:
  - Backbone layers: `1e-5`
  - Classifier head: `1e-4`
- Goal: Adapt high-level features to the medical domain while preserving low-level representations.

### Data Augmentation

| Transform | Value |
|---|---|
| `RandomResizedCrop` | 224 × 224 |
| `RandomHorizontalFlip` | 50% probability |
| `RandomRotation` | ±15° |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 |
| `Normalize` | ImageNet mean/std |

Validation and test images use `Resize(256)` → `CenterCrop(224)` → `Normalize`.

---

## 🤝 Ensemble Methods

### Soft Voting (Baseline Ensemble)
Averages the softmax probabilities across all 5 base models and selects the class with the highest mean probability.

### Weighted Ensemble
Each model's predictions are weighted proportionally to its individual accuracy before averaging:

| Model | Accuracy | Weight |
|---|---|---|
| ResNet-50 | 94% | ~20.9% |
| MobileNetV2 | 92% | ~20.5% |
| DenseNet-121 | 91% | ~20.3% |
| VGG-16 | 89% | ~19.8% |
| EfficientNet-B6 | 83% | ~18.5% |

### Stacking with Meta-Learners

Meta-features are generated by concatenating softmax probability vectors from all 5 base models (producing `5 × num_classes` features). The following Level 2 meta-learners are trained and compared:

| Meta-Learner | Implementation | Key Hyperparameters |
|---|---|---|
| **Logistic Regression** | `sklearn.linear_model` | `C=1.0`, `solver='liblinear'`, `max_iter=1000` |
| **Linear SVM** | `sklearn.svm.SVC` | `kernel='linear'` |
| **K-Nearest Neighbors** | `sklearn.neighbors` | `n_neighbors=7` |
| **Multi-Layer Perceptron** | `sklearn.neural_network` | `hidden_layers=(100,)`, `max_iter=500`, `relu`, `adam` |
| **XGBoost** | `xgboost` | `n_estimators=200`, `lr=0.1`, `max_depth=3`, GPU-accelerated |

> **Note:** MLP meta-features are scaled with `StandardScaler` before training.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| **Deep Learning** | PyTorch, TorchVision |
| **Machine Learning** | scikit-learn, XGBoost |
| **Data Handling** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Image Processing** | Pillow (PIL) |
| **Environment** | Google Colab (GPU runtime) |

---

## 🚀 Getting Started

### Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- The Kvasir dataset uploaded to Google Drive as `labeled-images.zip`

### Installation

All dependencies are pre-installed in Google Colab. If running locally, install:

```bash
pip install torch torchvision scikit-learn xgboost pandas numpy matplotlib seaborn pillow tqdm
```

### Running the Pipeline

1. **Upload the notebook/script** to Google Colab.
2. **Mount Google Drive** — the script automatically mounts and unzips the dataset.
3. **Select a GPU runtime** — go to `Runtime → Change runtime type → T4 GPU`.
4. **Run all cells sequentially** — the pipeline will:
   - Unzip and explore the dataset
   - Fine-tune all 5 base models (saved to Google Drive)
   - Generate meta-features from base model predictions
   - Train and evaluate all meta-learners
   - Display classification reports and confusion matrices for each approach

> ⏱️ **Estimated runtime:** ~3–5 hours on a T4 GPU (depending on dataset size).

---

## 📂 Project Structure

```
KVASIR/
├── kvasir_classification_meta_learners_finetune.py   # Main pipeline (Colab export)
├── research_final_ppt.pptx                           # Research presentation
└── README.md                                         # This file
```

### Pipeline Sections (in order of execution)

| Section | Lines | Description |
|---|---|---|
| Data Loading & Splitting | 1–70 | Mount Drive, unzip data, create train/val/test splits |
| Dataset & DataLoader Setup | 72–119 | Custom PyTorch `Dataset`, augmentations, `DataLoader` creation |
| Model Definition Helper | 121–182 | `get_model()` factory with architecture-specific classifier replacement |
| ResNet-50 Fine-Tuning | 184–356 | Two-stage training for ResNet-50 |
| VGG-16 Fine-Tuning | 357–510 | Two-stage training for VGG-16 |
| MobileNetV2 Fine-Tuning | 511–665 | Two-stage training for MobileNetV2 |
| DenseNet-121 Fine-Tuning | 666–814 | Two-stage training for DenseNet-121 |
| EfficientNet-B6 Fine-Tuning | 815–963 | Two-stage training for EfficientNet-B6 |
| Soft Voting Ensemble | 964–1032 | Average-probability ensemble baseline |
| Meta-Feature Generation | 1034–1108 | Generate stacking features from base model predictions |
| Logistic Regression Meta | 1110–1142 | Stacking with Logistic Regression |
| SVM Meta-Learner | 1144–1172 | Stacking with Linear SVM |
| KNN Meta-Learner | 1174–1202 | Stacking with K-Nearest Neighbors |
| MLP Meta-Learner | 1204–1242 | Stacking with scaled MLP |
| XGBoost Meta-Learner | 1244–1281 | Stacking with XGBoost |
| Weighted Ensemble | 1283–1388 | Accuracy-weighted ensemble voting |

---

## 📊 Results

Each method produces a full **classification report** (precision, recall, F1-score per class) and a **confusion matrix** heatmap.

### Base Model Accuracies (Level 1)

| Model | Test Accuracy |
|---|---|
| ResNet-50 | 94% |
| MobileNetV2 | 92% |
| DenseNet-121 | 91% |
| VGG-16 | 89% |
| EfficientNet-B6 | 83% |

### Stacking Ensemble Accuracies (Level 2 — Meta-Learners on 5 Fine-Tuned Base Learners)

| Meta-Learner | Test Accuracy |
|---|---|
| **XGBoost** | **97%** 🏆 |
| Logistic Regression | 96% |
| Linear SVM | 96% |
| MLP (Neural Network) | 95% |
| K-Nearest Neighbors | 94% |
| Soft Voting (baseline) | 95% |
| Weighted Voting | 95% |

> [!IMPORTANT]
> **Maximum Accuracy Achieved: 97%** — The XGBoost meta-learner stacking ensemble over all 5 fine-tuned base models achieved the highest overall accuracy of **97%**, surpassing every individual base model and all other ensemble strategies in this project. This demonstrates the effectiveness of the two-level stacking approach in combining complementary predictions across architectures.

---

## 📄 License

This project is for academic and research purposes. The Kvasir dataset is subject to its own [license and terms of use](https://datasets.simula.no/kvasir/).

---
