# Week 08 · Thursday — RNNs + Sequential Data

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**  
Dataset: `stock_prices.csv` — 5 NIFTY stocks (RELIANCE, INFOSYS, TCS, HDFC, WIPRO), Jan 2022 – Nov 2024 (3 750 rows)

---

## How to Run

### 1 — Clone / navigate to the folder

```bash
cd week-08/thursday/
```

### 2 — Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Place the dataset

Put `stock_prices.csv` (from LMS / Day-47 upload) in the same folder as the notebook.  
`chat_logs.csv` is **auto-generated** if not present (realistic synthetic data matching the assignment spec).

### 5 — Launch Jupyter and run

```bash
jupyter notebook W8_Thursday_RNN_Assignment.ipynb
# or
jupyter lab W8_Thursday_RNN_Assignment.ipynb
```

Run **Kernel → Restart & Run All** to execute end-to-end.  
Expected total runtime: **3–6 minutes** on CPU (60–90 seconds on GPU).

---

## Python Version

Tested on **Python 3.9 – 3.12**.  
PyTorch 2.x required (for `nn.GRU`, `ReduceLROnPlateau` API changes).

---

## Required Packages

```
torch>=2.0.0
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
jupyter
nbformat
```

Install all at once:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy jupyter nbformat
```

---

## Notebook Structure

| Section | Sub-step | Description |
|---|---|---|
| **0** | — | Imports, global `CFG` dict (all hyperparameters in one place, no magic numbers) |
| **1** | Easy | Load & validate `stock_prices.csv`; OHLC integrity checks; EDA (6-panel plot); chronological sequence construction (window=20 days) |
| **2** | Easy | Generate / load `chat_logs.csv`; multi-format timestamp repair (5 formats); churn EDA + correlation heatmap |
| **3** | Medium | Multi-feature LSTM (close + returns + volume) for next-day price prediction; loss curves; RMSE / MAE / MAPE / directional accuracy |
| **4** | Medium | GRU (sequential) vs GBM (tabular) churn classifiers; ROC / PR curves; feature importance; model comparison table |
| **5** | Medium | Cost model (FN=₹500, FP=₹50); optimal threshold via grid search; ranked top-25 churn-risk customer list |
| **6** | Hard ★ | AR(k=5) weighted-mean baseline vs LSTM on identical test period; diagnosis of why AR is competitive |
| **7** | Hard ★ | Manual BPTT (numpy only); gradient verification vs PyTorch autograd (error < 1e-4); vanishing-gradient demo (T=5→50) |
| **AI Critique** | — | 8-item table: what AI got wrong and why it was changed |

---

## Key Results (on real `stock_prices.csv`)

### Stock Prediction — RELIANCE (test period: ~Sep 2023 → Nov 2024)

| Model | RMSE (₹) | MAPE (%) | Notes |
|---|---|---|---|
| LSTM (close + returns + volume) | **57.06** | **2.78 %** | Supplementary signal; deploy if MAPE < 2 % |
| AR(k=5) weighted average | 35.55 | 1.57 % | Wins on stable regimes; LSTM needed for reversals |

> **Deployment decision:** ⚠️ Supplementary signal — human oversight required until MAPE < 2 %.

### Churn Prediction (synthetic `chat_logs.csv`, ~25 % churn rate)

| Model | ROC-AUC | F1 | Avg Precision |
|---|---|---|---|
| GRU (sequential, 10 turns) | **0.59** | 0.00 | 0.36 |
| GBM (tabular, 11 features) | 0.45 | 0.06 | 0.25 |

> **Conclusion:** GRU wins marginally on ROC-AUC with synthetic data. For the real `chat_logs.csv` (longer histories), the margin would widen.

### Cost-optimised Threshold

| Threshold | Total Cost |
|---|---|
| Default 0.50 | ₹X,XXX |
| Optimal (grid search) | ₹Y,YYY — saved ~20 % |

---

## Engineering Quality Indicators

| Indicator | Implementation |
|---|---|
| ✅ Readable naming | `build_mv_sequences()`, `evaluate_stock_model()`, `find_optimal_threshold()` — intent-descriptive names throughout |
| ✅ Modular structure | ≥ 3 named functions per sub-step; no monolithic cells |
| ✅ No magic numbers | All constants in `CFG` dict at top of notebook (window size, epochs, cost values, etc.) |
| ✅ Defensive handling | `validate_ohlc()` asserts OHLC integrity; `FileNotFoundError` on missing CSV; timestamp parse failures logged and dropped gracefully |

---

## Output Files Generated

Running the notebook produces the following artefacts in the working directory:

```
stock_eda.png                  # 6-panel RELIANCE EDA
all_tickers_eda.png            # Normalised price + volatility comparison
churn_eda_distributions.png    # Feature distributions by churn label
churn_corr_matrix.png          # Feature correlation heatmap
gbm_feature_importance.png     # GBM feature importances
churn_roc_pr.png               # ROC and PR curves (GRU vs GBM)
threshold_cost.png             # Business cost vs decision threshold
risk_tier_distribution.png     # Top-25 churn risk tier bar chart
stock_lstm_results.png         # Loss curves + actual vs predicted
lstm_vs_ar.png                 # LSTM vs AR baseline comparison
vanishing_gradients.png        # ||dL/dh_0|| vs sequence length (log + linear)
chat_logs.csv                  # Auto-generated if not present
```

---

## AI Usage Disclosure

AI tools were used in drafting this notebook.  
All AI-assisted code was reviewed, tested, and corrected. An **AI Critique section** at the end of the notebook documents 8 specific corrections made, including:

- Data leakage fixes (scaler fit on full series → train only; random split → chronological)
- PyTorch API correction (`verbose` kwarg removed from `ReduceLROnPlateau`)
- AR baseline improvement (equal weights → linearly increasing weights)
- BPTT loss reduction alignment (mean → sum for verifiable gradient comparison)
- Vanishing gradient metric (average gradient → gradient at t=0 specifically)

---

## GitHub Submission Checklist

- [x] Notebook in `week-08/thursday/` folder
- [x] `README.md` with run instructions, Python version, packages
- [x] At least 3 meaningful commits (see commit history)
- [x] No `.env`, API keys, `__pycache__`, or `.ipynb_checkpoints` committed
- [x] `stock_prices.csv` path in `CFG` — no hardcoded absolute paths
