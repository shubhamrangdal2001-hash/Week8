# Week 08 · Wednesday — CNNs + Semantic Embeddings
### Two-Stage Content Moderation Pipeline

---

## Overview

This notebook implements a full end-to-end content moderation pipeline that combines **convolutional neural networks** (MNIST image classification) with **semantic text embeddings** (hate speech detection + similarity search). The two systems are unified into a production-ready two-stage moderation architecture.

---

## Project Structure

```
W8_Wednesday_DailyAssignment.ipynb   ← Main notebook (all 7 sub-steps)
README.md                            ← This file
social_media_posts.csv               ← Required dataset (place in working directory)
```

---

## Datasets

| Dataset | Source | Used For |
|---|---|---|
| `social_media_posts.csv` | Provided by course | Text classification, semantic search, pipeline |
| MNIST | `tensorflow.keras.datasets.mnist` | CNN training & evaluation |

> **Note:** If `social_media_posts.csv` is not found, the notebook automatically generates a synthetic dataset (800 benign + 120 hate + 80 spam posts) so all cells remain runnable.  
> Set the path via environment variable: `export DATA_PATH=/path/to/social_media_posts.csv`

---

## Requirements

### Python
```
Python >= 3.8
```

### Core Dependencies
```
tensorflow >= 2.10
scikit-learn >= 1.1
pandas >= 1.4
numpy >= 1.22
matplotlib >= 3.5
seaborn >= 0.11
```

### Optional (Enhanced Features)
```
sentence-transformers    # Semantic BERT embeddings (falls back to TF-IDF + SVD if absent)
imbalanced-learn         # SMOTE oversampling (falls back to class_weight if absent)
```

### Install All
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn sentence-transformers imbalanced-learn
```

---

## Sub-steps Summary

| # | Sub-step | Key Functions | Status |
|---|---|---|---|
| 1 | Data Understanding | `load_social_media_data`, `summarise_dataset`, `plot_class_distributions` | Required |
| 2 | MNIST Preparation | `load_and_prepare_mnist`, `visualise_mnist_samples`, `plot_mnist_class_balance` | Required |
| 3 | CNN Model (MNIST) | `build_cnn_model`, `train_cnn_model`, `evaluate_cnn_model`, `visualise_first_layer_filters` | Required |
| 4A | Hate Speech Classifier | `build_tfidf_features`, `handle_class_imbalance`, `train_hate_speech_classifier`, `evaluate_classifier` | Required |
| 4B | Semantic Search | `compute_embeddings`, `find_similar_posts` | Required |
| 5 | Two-Stage Pipeline | `stage1_classify`, `stage2_semantic_flag`, `run_moderation_pipeline`, `business_impact_analysis` | Required |
| 6 | TF-IDF vs Embeddings | `compute_tfidf_similarities`, `compare_retrieval_methods` | Optional |
| 7 | Transfer Learning | `text_to_char_image`, `extract_cnn_features`, `evaluate_transfer_learning` | Optional |

---

## Constants (Configurable)

All parameters are defined at the top of the notebook — no magic numbers appear elsewhere.

| Constant | Default | Description |
|---|---|---|
| `DATA_PATH` | `social_media_posts.csv` | Path to text dataset |
| `CNN_EPOCHS` | `10` | Max training epochs (EarlyStopping active) |
| `CNN_BATCH_SIZE` | `128` | Batch size for CNN training |
| `CONV1_FILTERS` | `32` | Filters in first Conv2D layer |
| `CONV2_FILTERS` | `64` | Filters in second Conv2D layer |
| `DROPOUT_RATE` | `0.5` | Dropout regularisation rate |
| `SBERT_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-BERT model |
| `TOP_K_SIMILAR` | `5` | Top-K results for semantic search |
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity cutoff for Stage 2 |
| `TEST_SPLIT` | `0.2` | Train/test split ratio |
| `DAILY_POSTS` | `100,000` | Business volume assumption |
| `AVG_REVIEW_MINUTES` | `2` | Minutes per human review |

---

## CNN Architecture

```
Input (28×28×1)
    │
    ├── Conv2D(32 filters, 3×3, ReLU)   ← learns edges, corners
    ├── MaxPooling2D(2×2)
    │
    ├── Conv2D(64 filters, 3×3, ReLU)   ← learns strokes, curves
    ├── MaxPooling2D(2×2)
    │
    ├── Flatten
    ├── Dense(128, ReLU)
    ├── Dropout(0.5)
    └── Dense(10, Softmax)              ← digit classification
```

**Expected performance:** ~99% test accuracy on MNIST.

---

## Two-Stage Moderation Pipeline

```
Incoming Post
     │
     ▼
Stage 1: TF-IDF + Logistic Regression Classifier
     ├── FLAGGED  ──────────────────────► Human Review Queue
     │
     └── Cleared
          │
          ▼
     Stage 2: Sentence-BERT Cosine Similarity
     (compared against known harmful post embeddings)
          ├── Similarity ≥ 0.75  ───────► Human Review Queue
          └── Similarity < 0.75  ───────► CLEARED
```

---

## Class Imbalance Strategy

The dataset is imbalanced (~20% harmful posts). The notebook handles this explicitly:

1. **SMOTE** (if `imbalanced-learn` installed): oversample minority class in feature space
2. **`class_weight='balanced'`** (fallback): penalise misclassification of minority class
3. **Metrics**: F1-score, Recall, Precision, AUC-ROC — never raw accuracy as the primary metric

> A model predicting *everything as benign* achieves ~80% accuracy but catches zero harmful posts.  
> **Recall is the primary business metric** — minimise missed harmful posts (false negatives).

---

## Semantic Search: Why Embeddings Beat TF-IDF

| Method | Approach | Weakness |
|---|---|---|
| TF-IDF | Keyword frequency matching | Fails on paraphrases, synonyms |
| Sentence-BERT | Contextual meaning encoding | Computationally heavier |

**Example:** *"I hate that group"* and *"I despise those people"* share no keywords → TF-IDF similarity ≈ 0.  
Sentence-BERT similarity ≈ 0.85 (same meaning captured).

**CNN analogy:** TF-IDF is like comparing raw pixels. Embeddings are like comparing CNN feature maps — learned, invariant representations that generalise beyond surface patterns.

---

## Transfer Learning Conclusion (Sub-step 7)

The MNIST CNN was applied to character-frequency heatmap "images" of text posts.

**Result:** Transfer does **not** work well.  
**Reason:** MNIST CNN learned digit-stroke primitives (domain-specific). Character heatmaps encode statistical text properties — an entirely different feature space. No shared low-level structure exists between handwritten digits and character frequency distributions.

**Contrast:** ImageNet → Medical imaging transfer *does* work (shared edges, textures, shapes).

---

## Engineering Quality Checklist

- [x] Minimum 2 functions per sub-step
- [x] All constants defined — no magic numbers
- [x] No hardcoded file paths
- [x] `try/except` defensive coding throughout
- [x] Input validation in all public functions
- [x] Graceful fallbacks (no SBERT, no SMOTE, no CSV)
- [x] Visualisations: class distribution, training curves, confusion matrices, filter plots
- [x] Class imbalance handled explicitly with justification
- [x] F1/Recall/AUC-ROC as primary metrics (accuracy excluded)
- [x] Docstrings on every function (Parameters / Returns / Raises)
- [x] AI usage compliance section with critique

---

## Running the Notebook

```bash
# 1. Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn \
            sentence-transformers imbalanced-learn jupyter

# 2. Place dataset in working directory (or set env variable)
export DATA_PATH=/path/to/social_media_posts.csv

# 3. Launch Jupyter
jupyter notebook W8_Wednesday_DailyAssignment.ipynb
```

Run cells **top to bottom** — each sub-step builds on the previous.

---

## Author

Assignment: Week 08 · Wednesday  
Course: AI/ML Engineering  
Target Grade: Band 4
