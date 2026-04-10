# JPM Weekly Closing Price Forecasting (Machine Learning Regression)

ELEC3544 *Data Science with Foundation Models* course project: predict **JPMorgan Chase (NYSE: JPM) next-week weekly closing price** from public market data using classical regression, and benchmark against a **naive persistence baseline (NPB)**.

**Data policy:** **Federal Reserve Economic Data (FRED)** supplies official daily series for the **10-year Treasury (DGS10)**, **VIX (VIXCLS)**, **S&P 500 index (SP500)**, and **USD spot rates vs EUR/GBP (DEXUSEU, DEXUSUK)**; these are aggregated to **week-ending Friday** in `fetch_data.py`. **Equity OHLCV** for JPM, peer banks (BAC, C, WFC), **XLF**, **gold futures (GC=F)**, and **FTSE 100 (^FTSE)** comes from **Yahoo Finance** via `yfinance` (aggregated quotes, not exchange tape). Stooq’s automated CSV endpoint requires a separate API key; this project avoids that for reproducibility.

---

## Requirements

- Python **3.9+** (Anaconda recommended)
- Dependencies in `requirements.txt` (`yfinance`, `pandas`, `scikit-learn`, `statsmodels`, `pandas-ta`, `xgboost`, `matplotlib`, `seaborn`)

---

## How to Run

From the project root:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

Or run the steps separately:

```bash
python fetch_data.py          # Download weekly panel → data/raw/
python build_dataset.py       # Clean, features, VIF, split, scale → data/processed/
python train_and_evaluate.py  # Train & evaluate → models/, reports/
```

**Note:** `fetch_data.py` uses the network (Yahoo + FRED). FRED graph CSV does **not** require an API key. If a Yahoo symbol changes, update `TICKERS_YF` in `config.py`.

---

## Data

| Item | Description |
|------|-------------|
| **Horizon** | Default `2005-01-01`–`2025-12-31` (weekly), configurable in `config.py` |
| **Target** | Next week’s **JPM** weekly close; features use **current week and earlier** only; `shift(-1)` aligns the target to avoid lookahead |
| **Equities / futures (yfinance)** | JPM (primary), BAC, C, WFC, XLF, `GC=F`, `^FTSE` |
| **Macro / rates (FRED)** | `DGS10` → `tnx_Close`, `VIXCLS` → `vix_Close`, `SP500` → `gspc_Close`, `DEXUSEU` → `eurusd_Close`, `DEXUSUK` → `gbpusd_Close` |
| **Indicators** | SMA/RSI/ATR on primary (JPM) in `build_dataset.py`; after VIF screening, the final feature set is in `data/processed/feature_list.json` |

**Key outputs**

- `data/raw/weekly_raw_merged.csv` — merged raw weekly panel  
- `data/processed/X_train_scaled.csv`, `X_test_scaled.csv` — features (all scaled except `primary_Close`)  
- `data/processed/y_train.csv`, `y_test.csv` — targets (`target_next_primary_close`)  
- `data/processed/primary_close_this_week_aligned.csv` — this-week close for NPB alignment  
- `data/processed/dataset_meta.json` — sample sizes, date ranges, feature list  

---

## Methods & Modeling

1. **Missing values & outliers** — Tiered imputation; quantile capping uses bounds **estimated on the training set** and applied to train and test; **`primary_Close` is not capped** so test-period price levels are not clipped to the training upper bound.  
2. **Multicollinearity** — Iteratively drop features with VIF > 10 on the training set; **`primary_Close` is protected** so level-based comparison with NPB remains meaningful.  
3. **Scaling** — `StandardScaler` fit **only on training data**; **`primary_Close` stays in raw price units**; other features are standardized.  
4. **Split** — Chronological **80% / 20%** split, no shuffling.  
5. **Models** — NPB; OLS; Ridge / Lasso (`TimeSeriesSplit` + grid search); Random Forest and XGBoost (same TSCV).  
6. **Tree models** — RF / XGBoost are trained on the **residual target** `y − primary_Close` and predictions are **`primary_Close + model output`**, reducing leaf collapse when price levels lie outside the training range.

---

## Results

Metrics below are from the **held-out test set** (`n_test` = 435; train ends `2021-10-29`, test `2021-10-30`–`2025-12-27`; see `data/processed/dataset_meta.json`). RMSE/MAE are in **USD**; R² is on **price levels**. **`vs_npb_rmse_pct`** = \((\mathrm{RMSE}_{\mathrm{NPB}} - \mathrm{RMSE}_{\mathrm{model}}) / \mathrm{RMSE}_{\mathrm{NPB}} \times 100\) — positive means lower RMSE than NPB.

| Model | RMSE | MAE | R² | vs NPB (RMSE %) |
|-------|------|-----|-----|-----------------|
| NPB | 4.768 | 2.496 | 0.9941 | 0.0 |
| OLS | 4.762 | 2.519 | 0.9941 | 0.12 |
| Ridge | 4.761 | 2.518 | 0.9941 | 0.13 |
| **Lasso** | **4.754** | 2.549 | 0.9942 | **0.29** |
| RF | 4.763 | 2.579 | 0.9941 | 0.10 |
| XGBoost | 4.891 | 2.791 | 0.9938 | −2.59 ·|

**Takeaways (this run):** Lasso achieved the **lowest test RMSE**, about **0.29%** below NPB; OLS, Ridge, and RF show **small** gains (~0.10–0.13%). **XGBoost underperformed** NPB on RMSE (~2.6% worse). High R² is expected when `primary_Close` is in the feature set; it does not by itself imply large economic edge over persistence.

**Artifacts:** `reports/test_metrics.csv`, `reports/evaluation_summary.json`, `reports/correlation_train.png`, `reports/test_actual_vs_npb.png`, `reports/vif_final.csv`, `reports/importance_*.csv`. **Linear importance files** store regression coefficients (`coef_`); Lasso often has **many exact zeros** (L1 sparsity), while Ridge/OLS keep small non-zero weights on most features. Tree models use `feature_importances_` instead.

---

## Saved Models & Inference

- Location: `models/*.pkl`  
- **Linear models (OLS / Ridge / Lasso)** — regress **next-week close** directly; call `predict` with inputs aligned to `X_*_scaled.csv`.  
- **Random Forest / XGBoost** — pickled estimators predict the **residual vs current-week close**; final level forecast:  
  **`next_close = primary_Close + model.predict(X)`** (keep `primary_Close` in raw units in `X`, same layout as in training).

---

## Discussion (for the report)

1. **NPB is strong** — On this split, linear models and RF only **narrowly** beat NPB on RMSE (≈0.1–0.3%); that is still far from a **15%** RMSE reduction if the proposal used that bar. XGBoost did worse than NPB here, so tree complexity did not pay off on the residual target setup.  
2. **Data sourcing** — Cite **FRED** (with series IDs) for rates and broad market; cite **Yahoo Finance / yfinance** for individual stock and futures quotes, and note aggregation vs official exchange data.  
3. **Reproducibility** — Fix dates and seeds in `config.py`; remote data updates over time, so re-runs may shift samples and metrics slightly.

---

## Repository layout (core)

```
HSBC Project/
├── README.md
├── requirements.txt
├── config.py
├── fetch_data.py
├── build_dataset.py
├── train_and_evaluate.py
├── run_pipeline.py
├── data/raw/
├── data/processed/
├── models/
└── reports/
```

---

## Course documents

- Deliverables and success criteria: **`Project Proposal.pdf`** (course handout).  
- Update the written proposal to describe **JPM + FRED/yfinance** if the original targeted HSBC.
