# Week 08 · Monday — Time Series Analysis
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## How to Run

### Prerequisites
- Python 3.10+
- Jupyter Notebook / JupyterLab

### Install dependencies
```bash
pip install -r requirements.txt
```

### Place datasets
Download from LMS and place in this folder:
```
week-08/monday/ecommerce_sales_ts.csv
week-08/monday/sensor_data.csv
```
> **Note:** If the CSVs are absent, the notebook auto-generates realistic synthetic stand-ins so it always runs end-to-end. Replace with real LMS files before final submission.

### Run the notebook
```bash
cd week-08/monday
jupyter notebook time_series_analysis.ipynb
```
Run all cells top-to-bottom (`Kernel → Restart & Run All`).

---

## Package Versions
```
Python       3.10+
pandas       2.x
numpy        1.26+
matplotlib   3.8+
seaborn      0.13+
statsmodels  0.14+
scikit-learn 1.4+
prophet      1.1+      (optional — SARIMA fallback if absent)
nbformat     5.x       (for notebook build script only)
```

---

## Folder Structure
```
week-08/
└── monday/
    ├── time_series_analysis.ipynb     ← main notebook (source)
    ├── time_series_analysis_executed.ipynb  ← pre-executed with outputs
    ├── README.md                      ← this file
    ├── requirements.txt
    ├── prompts.md                     ← AI usage log (all prompts + critiques)
    └── *.png                          ← saved figures (auto-generated)
```

---

## Sub-steps Covered

| Sub-step | Difficulty | Description |
|---|---|---|
| 1 | 🟢 Easy | E-commerce EDA: stationarity, decomposition, ACF/PACF |
| 2 | 🟢 Easy | Sensor data audit & cleaning (dedup, reindex, interpolation) |
| 3 | 🟡 Medium | ARIMA baseline with temporal hold-out and MAPE evaluation |
| 4 | 🟡 Medium | SARIMA + Prophet improvement; complexity vs accuracy tradeoff |
| 5 | 🟡 Medium | Sensor failure prediction (Random Forest, recall-first, feature importance) |
| 6 | 🔴 Hard | Rule-based vs ML comparison with cost matrix |
| 7 | 🔴 Hard | Fleet-scale (100k sensors) threshold optimisation |

---

## Key Design Decisions

- **No random train/test split** — all splits are strictly temporal
- **MAPE as primary forecasting metric** — scale-invariant, communicates % buffer to planners
- **Recall as primary failure metric** — FN costs 20× more than FP
- **Cost threshold ≠ F1 threshold** — F1 is symmetric; this problem is not
- **Modular functions** — every sub-step has 2+ named functions, no monolithic cells
- **No magic numbers** — all constants defined in the `Global Constants` cell

---

## Commit History
```
feat: add environment setup, constants, and synthetic data generator
feat: sub-steps 1-2 ecommerce EDA and sensor data cleaning
feat: sub-steps 3-4 ARIMA and SARIMA/Prophet forecasting models
feat: sub-step 5 sensor failure prediction with RF and feature engineering
feat: sub-steps 6-7 cost-matrix comparison and fleet threshold optimisation
```
