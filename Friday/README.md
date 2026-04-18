# Week 08 · Friday — Chest X-Ray Classification via Transfer Learning
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## 📌 Assignment Overview

Automated chest X-ray screening tool for Dr. Sameer Rao (AIIMS).  
**Task:** 5-class classification across 520 labeled + 30 unlabeled images using transfer learning.

| Sub-step | Difficulty | Topic |
|---|---|---|
| 1 | 🟢 Easy | Data understanding & clinical risk analysis |
| 2 | 🟢 Easy | Feature extraction with EfficientNet-B0 |
| 3 | 🟡 Medium | Fine-tuning strategy & comparison |
| 4 | 🟡 Medium | Grad-CAM explainability |
| 5 | 🟡 Medium | ME1 prep + unlabeled prediction |
| 6 | 🔴 Hard | Training from scratch comparison |
| 7 | 🔴 Hard | Clinical triage protocol |

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional; CPU fallback is automatic)

### Installation

```bash
# Clone the repo and navigate to this folder
cd week-08/friday

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook chest_xray_transfer_learning.ipynb
```

**Or** run as a plain script (cells execute top to bottom):

```bash
jupyter nbconvert --to script chest_xray_transfer_learning.ipynb
python chest_xray_transfer_learning.py
```

> **Note:** If `medical_imaging_meta.csv` is not present (e.g., LMS file not downloaded),  
> the notebook auto-generates realistic synthetic metadata and runs end-to-end without errors.

---

## 📂 Folder Structure

```
week-08/friday/
├── chest_xray_transfer_learning.ipynb  ← Main notebook (all 7 sub-steps)
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
├── prompts.md                          ← Exact AI prompts used
├── medical_imaging_meta.csv            ← Dataset (download from LMS)
├── outputs/                            ← Auto-created by notebook
│   ├── class_distribution.png
│   ├── fe_training_curves.png
│   ├── fe_confusion_matrix.png
│   ├── ft_training_curves.png
│   ├── ft_confusion_matrix.png
│   ├── fe_vs_ft_comparison.png
│   ├── gradcam_COVID19.png
│   ├── gradcam_Tuberculosis.png
│   ├── unlabeled_predictions.csv
│   └── triage_report.csv
└── saved_models/                       ← Auto-created by notebook
    ├── efficientnet_b0_fe_best.pt
    ├── efficientnet_b0_ft_best.pt
    └── efficientnet_b0_scratch_best.pt
```

---

## 🧪 Python Version & Key Packages

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | EfficientNet-B0 + transforms |
| scikit-learn | ≥1.3.0 | Stratified splits, metrics |
| pandas | ≥2.0.0 | Data loading/manipulation |
| numpy | ≥1.24.0 | Numerics |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Heatmaps |
| Pillow | ≥10.0.0 | Image loading |

---

## 🏥 Clinical Design Decisions

| Decision | Rationale |
|---|---|
| **Primary metric: Recall** | False Negatives (missed diseases) are more costly than False Positives in screening |
| **EfficientNet-B0 over ResNet-50** | 5.3M vs 25M params — far less overfitting risk at n=520 |
| **WeightedRandomSampler** | Counters 5:1 class imbalance (Normal vs Pleural Effusion) |
| **Focal Loss (γ=2)** | Focuses training on hard minority-class examples |
| **Fine-tuning preferred** | Higher recall on COVID-19/TB; catastrophic forgetting mitigated by low LR + partial unfreeze |
| **Grad-CAM on last block** | Highest spatial resolution; interpretable by radiologists |
| **Safety override in triage** | COVID-19/TB predictions always routed to human review regardless of confidence |

---

## ⚠️ Limitations & Risks

1. **Simulated images** — this notebook uses synthetic pixel tensors (hash-seeded from `image_id`). Real performance depends on actual DICOM/PNG loading. Replace `_load_image()` in `ChestXRayDataset`.

2. **Confidence calibration** — EfficientNet-B0 tends to be overconfident on small datasets. Apply temperature scaling on the validation set before trusting triage thresholds.

3. **Hospital bias** — AIIMS_Delhi contributes ~50% of data. Triage protocol may perform worse at KEM/PGIMER sites. Evaluate stratified by hospital before deployment.

4. **Not for clinical use** — This is an educational exercise. IRB approval, prospective validation, and regulatory clearance (CDSCO Class C) are required for any clinical deployment.

---

## 🤖 AI Usage Disclosure

All AI-assisted sub-steps include the exact prompt used (see `prompts.md`) and a critique section explaining modifications made and why. Raw AI output was not submitted — all code was reviewed, debugged, and extended.

---

## 📝 Commit History (to be pushed)

```
[commit 1] init: project structure, Config dataclass, seed utility
[commit 2] feat: data loading, synthetic fallback, class distribution plots
[commit 3] feat: EfficientNet-B0 FE model, FocalLoss, WeightedRandomSampler
[commit 4] feat: fine-tuning with discriminative LRs, FE vs FT comparison
[commit 5] feat: Grad-CAM explainability, radiologist summary
[commit 6] feat: ME1 prep, unlabeled prediction, confidence scores
[commit 7] feat: scratch training, 3-tier triage, FN risk estimation
```
