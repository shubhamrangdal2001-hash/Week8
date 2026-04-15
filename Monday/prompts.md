# AI Usage Log — Week 08 Monday
All prompts used during this assignment, with critique and modifications made.

---

## Sub-step 1 — E-commerce EDA

**Prompt:**
> "Write a modular Python function to perform ADF stationarity testing on a pandas Series and return a summary dictionary. Include critical values and a human-readable stationarity verdict."

**Was the AI output correct?**
Structurally yes. The function printed results but returned nothing — not usable downstream.

**What I changed and why:**
- Changed return type from `None` to `dict` so downstream code can branch on `out['stationary']`
- Added `autolag='AIC'` for automatic lag selection (AI used a hardcoded lag=1)
- Added `stationary` boolean key so the ADF result can be used programmatically
- Fixed f-string separator that broke across lines in the notebook format

---

## Sub-step 2 — Sensor Data Cleaning

**Prompt:**
> "Write a Python function to audit a sensor DataFrame for time-series data quality issues: duplicates, missing values, irregular intervals, and near-zero-variance columns. Return a structured dict of findings."

**Was the AI output correct?**
Partially. The audit logic was sound but the clean function was missing the most critical fix for sequence models.

**What I changed and why:**
- Added `reindex` to a uniform hourly grid — AI omitted this entirely. Without it, rolling window features compute incorrect intervals across gaps.
- Separated target label (`machine_status`) from sensor columns in the interpolation step. AI applied `interpolate()` uniformly, which would smooth a binary label (wrong — labels must be forward-filled, not interpolated).
- Added explicit `assert df.isnull().sum().sum() == 0` post-clean validation.

---

## Sub-step 3 — ARIMA Baseline

**Prompt:**
> "Write a modular Python function to fit an ARIMA model on a pandas time-series training set, forecast the test horizon, and evaluate using MAE, RMSE, and MAPE. Include a guard against negative sales forecasts."

**Was the AI output correct?**
Yes, mostly. Two issues:
1. MAPE denominator: `np.mean(np.abs((actual - predicted) / actual))` fails when `actual` contains near-zero values.
2. No clip on negative forecast values.

**What I changed and why:**
- Added `mask = actual > 1` before MAPE calculation to avoid division-by-near-zero
- Added `np.maximum(0, forecast)` clip — ARIMA on differenced series occasionally predicts negative values at the start of the forecast horizon
- Added plain-English business interpretation in the print output

---

## Sub-step 4 — SARIMA / Prophet

**Prompt:**
> "How do I compare ARIMA and SARIMA on the same hold-out set and quantify whether the seasonal model's improvement justifies its added complexity?"

**Was the AI output correct?**
AI suggested AIC/BIC for model comparison — technically valid for in-sample fit selection but **not** appropriate for evaluating out-of-sample business forecasting accuracy. AIC rewards parsimony, not predictive accuracy.

**What I changed and why:**
- Replaced AIC/BIC comparison with hold-out MAPE comparison (the metric that matters operationally)
- Kept a complexity tradeoff table in the markdown with training time as a proxy for complexity
- Added Prophet as a parallel comparison path with graceful fallback if package absent

---

## Sub-step 5 — Failure Prediction

**Prompt:**
> "Design a production-quality feature engineering function for pump sensor time-series data. Include rolling statistics, rate of change, and lag features. The function should be modular and avoid data leakage."

**Was the AI output correct?**
The feature structure was correct. Two bugs:
1. `rolling(window, min_periods=window)` dropped the first `window` rows entirely — loses data and breaks temporal alignment
2. Used `fillna(method='bfill')` which is deprecated in pandas 2.x

**What I changed and why:**
- Changed to `min_periods=1` so rolling features are available from row 1 (using partial windows)
- Replaced `fillna(method='bfill')` with `.bfill()` for pandas 2.x compatibility
- Added `feature_cols` exclusion of object-type and timestamp columns to prevent dtype errors in sklearn

---

## Sub-step 6 — Rule vs ML

**Prompt:**
> "How do I compare a single-threshold rule vs an ML classifier using a business cost matrix? Write code that sweeps thresholds over each feature and finds the minimum-cost rule."

**Was the AI output correct?**
AI only checked the median (50th percentile) threshold per feature.

**What I changed and why:**
- Changed to sweep 50th–99th percentile in 2% increments — finds the actual cost-minimising threshold, not just the median
- Added `confusion_matrix(..., labels=[0,1])` to ensure consistent ordering when a threshold predicts all-zeros or all-ones
- Added head-to-head print summary comparing rule cost vs ML cost directly

---

## Sub-step 7 — Fleet Cost Optimisation

**Prompt:**
> "How do I find the probability threshold that minimises expected daily business cost for a binary classifier deployed on 100,000 sensors, given asymmetric FP/FN costs?"

**Was the AI output correct?**
AI scaled by test set size directly (absolute counts × scaling factor) which inflates/deflates estimates based on test set size rather than true rates.

**What I changed and why:**
- Restructured to compute per-sensor-hour rates first (`fn / total_preds`, `fp / total_preds`), then multiply by fleet × hours/day
- This makes the estimate robust to any test set size and correctly represents a rate-based extrapolation
- Added the annual cost difference calculation (cost of using F1 threshold vs cost threshold) to make the business impact of the wrong metric concrete
