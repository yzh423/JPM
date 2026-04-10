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

After `run_pipeline.py`, metrics are in `reports/test_metrics.csv` (RMSE/MAE in **USD**; R² on **price levels**). Figures: `reports/correlation_train.png`, `reports/test_actual_vs_npb.png`, `reports/vif_final.csv`, `reports/importance_*.csv`, `reports/evaluation_summary.json`.

---

## Saved Models & Inference

- Location: `models/*.pkl`  
- **Linear models (OLS / Ridge / Lasso)** — regress **next-week close** directly; call `predict` with inputs aligned to `X_*_scaled.csv`.  
- **Random Forest / XGBoost** — pickled estimators predict the **residual vs current-week close**; final level forecast:  
  **`next_close = primary_Close + model.predict(X)`** (keep `primary_Close` in raw units in `X`, same layout as in training).

---

## Discussion (for the report)

1. **NPB is strong** — at a one-week horizon, “next week ≈ this week” is a hard baseline; ML gains are often small.  
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
