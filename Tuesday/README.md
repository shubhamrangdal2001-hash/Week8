# Hospital Readmission Prediction

**Week 08 · Tuesday** | PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar

![NumPy NN](https://img.shields.io/badge/NN-NumPy%20only-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![No PyTorch](https://img.shields.io/badge/PyTorch-not%20used-red)

---

## Overview

End-to-end pipeline to predict 30-day hospital readmissions. Covers data auditing, professional cleaning, a 3-layer neural network built from scratch in NumPy, business-cost threshold tuning, and two optional advanced steps.

---

## Pipeline steps

| Step | Title | Description |
|------|-------|-------------|
| 1 | Data quality audit | Missing values, outliers, duplicates, categorical inconsistencies, class balance |
| 2 | Data cleaning | Winsorisation, median imputation, gender normalisation, duplicate removal |
| 3 | NumPy neural network | 3-layer NN, He init, ReLU + sigmoid, forward prop, BCE loss, backprop |
| 4 | Training & evaluation | Mini-batch SGD, loss curve, ROC-AUC, sklearn LR baseline, imbalance fix |
| 5 | Business optimisation | Asymmetric cost function, threshold sweep, optimal threshold, clinical recommendation |
| 6 ✦ | Misleading accuracy demo | Shows why 94% accuracy is meaningless on imbalanced data |
| 7 ✦ | NN feature extractor | Hidden-layer embeddings → LR classifier on top, performance comparison |

✦ Optional / take-home steps

---

## Architecture

```
Input [n_features]  →  Hidden-1 [32, ReLU]  →  Hidden-2 [16, ReLU]  →  Output [1, Sigmoid]

Initialisation : He  (variance = 2 / fan_in)
Loss           : Binary Cross-Entropy
Optimiser      : Mini-batch SGD  (lr = 0.005, batch = 64, epochs = 500)
```

---

## Key results

| Metric | Value |
|--------|-------|
| NumPy NN ROC-AUC | 0.72 |
| sklearn LR ROC-AUC | 0.72 |
| Embedding LR ROC-AUC | 0.72 |
| Dataset rows (clean) | 2,000 |

> ROC-AUC is used instead of accuracy because the class ratio is ~81:19. A trivial all-negative predictor scores 81% accuracy while catching zero readmissions.

---

## Business cost structure

| Outcome | Cost |
|---------|------|
| False Negative (missed readmission) | 10 |
| False Positive (unnecessary follow-up) | 1 |

Threshold is tuned from 0.05 → 0.95 in steps of 0.01 to minimise `FN × 10 + FP × 1`.

---

## Project structure

```
week-08/tuesday/
├── hospital_readmission_solution.py   # main script
├── README.md                          # this file
├── hospital_records.csv               # dataset (or auto-generated)
└── plots/
    ├── loss_curve.png                 # training & validation loss
    ├── threshold_cost.png             # cost vs threshold sweep
    └── misleading_accuracy.png        # step 6 confusion matrix
```

---

## Quick start

```bash
pip install numpy pandas scikit-learn matplotlib

# Place hospital_records.csv in the same folder, or let the script
# generate a synthetic dataset automatically.

python hospital_readmission_solution.py
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Neural network (the only library used for the NN) |
| `pandas` | Data loading and cleaning |
| `scikit-learn` | Baseline model, metrics, train/test split |
| `matplotlib` | Loss curve and cost-threshold plots |

> No TensorFlow, PyTorch, or Keras. The NN is pure NumPy.

---

## Prompt used

> "You are an expert Data Scientist and AI Engineer. Help me complete the following assignment step-by-step with clean, production-quality code and clear explanations. […] Do NOT use TensorFlow or PyTorch. Only NumPy for neural network. Keep code readable and structured."

---

## Critique

**Was the AI output correct?**
Yes — all 7 steps run end-to-end. The backprop math, He initialisation, BCE loss, and threshold sweep are all correct.

**What was modified and why?**

- Added synthetic data generator — `hospital_records.csv` is not available at runtime
- Threshold sweep step changed from 0.05 → 0.01 for finer granularity
- Added `np.random.seed` + sklearn `random_state` for full reproducibility
- Switched to `matplotlib Agg` backend to prevent GUI popup in headless environments
- Oversampling ratio made data-driven instead of a hardcoded ×3 multiplier

---

*Deadline: Wednesday 09:15 AM · Submit: push to `week-08/tuesday/` and paste link in LMS*
