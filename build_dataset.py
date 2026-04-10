"""
Clean weekly panel, engineer features, resolve multicollinearity (VIF), time split, scale.
Target: next week's primary (JPM) weekly closing price (no lookahead in features).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config

RAW_PATH = Path(__file__).resolve().parent / "data" / "raw" / "weekly_raw_merged.csv"
PROC_DIR = Path(__file__).resolve().parent / "data" / "processed"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

PRIMARY_CLOSE = f"{config.PRIMARY_KEY}_Close"


def _safe_div(a: pd.Series, b: pd.Series, name: str) -> pd.Series:
    out = a.astype(float) / b.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).rename(name)


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}; run fetch_data.py first.", file=sys.stderr)
        raise SystemExit(1)
    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def clean_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def impute_series(s: pd.Series) -> pd.Series:
    s = s.ffill().bfill()
    if s.isna().any():
        s = s.interpolate(method="time", limit_direction="both")
    if s.isna().any():
        med = s.median()
        s = s.fillna(med if pd.notna(med) else 0.0)
    return s


def impute_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = impute_series(out[c])
    return out


def cap_columns(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def train_quantile_bounds(
    df: pd.DataFrame, low_q: float = 0.005, high_q: float = 0.995
) -> dict[str, tuple[float, float]]:
    bounds = {}
    for c in df.columns:
        lo = float(df[c].quantile(low_q))
        hi = float(df[c].quantile(high_q))
        bounds[c] = (lo, hi)
    return bounds


def add_primary_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pk = config.PRIMARY_KEY
    close = out[f"{pk}_Close"]
    high = out[f"{pk}_High"]
    low = out[f"{pk}_Low"]
    vol = out[f"{pk}_Volume"]

    out[f"{pk}_SMA5"] = ta.sma(close, length=5)
    out[f"{pk}_SMA10"] = ta.sma(close, length=10)
    out[f"{pk}_SMA20"] = ta.sma(close, length=20)
    out[f"{pk}_RSI14"] = ta.rsi(close, length=14)
    atr = ta.atr(high=high, low=low, close=close, length=14)
    out[f"{pk}_ATR14"] = atr
    out[f"{pk}_vol_chg"] = vol.pct_change(fill_method=None)
    out[f"{pk}_ret_1w"] = close.pct_change(fill_method=None)
    return out


def add_cross_asset_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = out[PRIMARY_CLOSE]
    out["ratio_primary_xlf"] = _safe_div(h, out["xlf_Close"], "ratio_primary_xlf")
    out["ratio_gspc_ftse"] = _safe_div(out["gspc_Close"], out["ftse_Close"], "ratio_gspc_ftse")
    out["ratio_sp_vix"] = _safe_div(out["gspc_Close"], out["vix_Close"], "ratio_sp_vix")
    out["ratio_yield_gold"] = _safe_div(out["tnx_Close"], out["gold_Close"], "ratio_yield_gold")
    return out


def add_peer_momentum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in ("bac", "c", "wfc", "xlf"):
        c = f"{name}_Close"
        if c in out.columns:
            out[f"{name}_ret_1w"] = out[c].pct_change(fill_method=None)
    for name in ("gspc", "ftse", "gold"):
        c = f"{name}_Close"
        if c in out.columns:
            out[f"{name}_ret_1w"] = out[c].pct_change(fill_method=None)
    out["tnx_chg"] = out["tnx_Close"].pct_change(fill_method=None)
    out["eurusd_ret"] = out["eurusd_Close"].pct_change(fill_method=None)
    out["gbpusd_ret"] = out["gbpusd_Close"].pct_change(fill_method=None)
    return out


def compute_vif(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.dropna()
    if x.empty:
        return pd.DataFrame(columns=["feature", "vif"])
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.shape[0] < x.shape[1] + 2:
        return pd.DataFrame(columns=["feature", "vif"])
    vifs = []
    for i, col in enumerate(x.columns):
        try:
            v = variance_inflation_factor(x.values.astype(float), i)
        except Exception:
            v = np.nan
        vifs.append((col, float(v) if np.isfinite(v) else np.nan))
    return pd.DataFrame(vifs, columns=["feature", "vif"]).sort_values("vif", ascending=False)


def iterative_vif_drop(
    train_df: pd.DataFrame,
    max_vif: float = 10.0,
    max_rounds: int = 50,
    protected: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    protected = protected or frozenset()
    x = train_df.copy()
    reports: list[pd.DataFrame] = []
    for _ in range(max_rounds):
        rep = compute_vif(x)
        reports.append(rep)
        if rep.empty or rep["vif"].isna().all():
            break
        droppable = rep[~rep["feature"].isin(protected)]
        if droppable.empty:
            break
        worst = droppable.iloc[0]
        if worst["vif"] <= max_vif or np.isnan(worst["vif"]):
            break
        feat = worst["feature"]
        if feat not in x.columns:
            break
        x = x.drop(columns=[feat])
    return x, reports


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pk = config.PRIMARY_KEY

    raw = load_raw()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = clean_structure(raw)
    raw = impute_frame(raw)

    if PRIMARY_CLOSE not in raw.columns:
        raise SystemExit(f"Expected column {PRIMARY_CLOSE} missing.")

    feat = add_primary_technicals(raw)
    feat = add_peer_momentum(feat)
    feat = add_cross_asset_ratios(feat)

    candidate_features = [
        f"{pk}_SMA5",
        f"{pk}_SMA10",
        f"{pk}_SMA20",
        f"{pk}_RSI14",
        f"{pk}_ATR14",
        f"{pk}_vol_chg",
        f"{pk}_ret_1w",
        PRIMARY_CLOSE,
        f"{pk}_Volume",
        "ratio_primary_xlf",
        "ratio_gspc_ftse",
        "ratio_sp_vix",
        "ratio_yield_gold",
        "bac_ret_1w",
        "c_ret_1w",
        "wfc_ret_1w",
        "xlf_ret_1w",
        "gspc_ret_1w",
        "ftse_ret_1w",
        "gold_ret_1w",
        "tnx_chg",
        "eurusd_ret",
        "gbpusd_ret",
    ]
    present = [c for c in candidate_features if c in feat.columns]
    X = feat[present].copy()
    y = feat[PRIMARY_CLOSE].shift(-1).rename("target_next_primary_close")

    full = X.copy()
    full["target_next_primary_close"] = y
    full["primary_close_this_week"] = feat[PRIMARY_CLOSE]
    full = full.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(full)
    split_idx = int(n * config.TRAIN_FRAC)
    train_full = full.iloc[:split_idx]
    test_full = full.iloc[split_idx:]

    X_train_raw = train_full[present]
    X_test_raw = test_full[present]

    cap_cols = [c for c in present if c != PRIMARY_CLOSE]
    bounds = train_quantile_bounds(X_train_raw[cap_cols])
    X_train_cap = X_train_raw.copy()
    X_test_cap = X_test_raw.copy()
    X_train_cap[cap_cols] = cap_columns(X_train_raw[cap_cols], bounds)
    X_test_cap[cap_cols] = cap_columns(X_test_raw[cap_cols], bounds)

    X_vif, vif_reports = iterative_vif_drop(
        X_train_cap, protected=frozenset({PRIMARY_CLOSE})
    )
    kept_features = list(X_vif.columns)
    X_train = X_train_cap[kept_features]
    X_test = X_test_cap[kept_features]

    scale_cols = [c for c in kept_features if c != PRIMARY_CLOSE]
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    if scale_cols:
        X_train_s[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

    y_train = train_full["target_next_primary_close"].loc[X_train.index]
    y_test = test_full["target_next_primary_close"].loc[X_test.index]

    X_train_s.to_csv(PROC_DIR / "X_train_scaled.csv", date_format="%Y-%m-%d")
    X_test_s.to_csv(PROC_DIR / "X_test_scaled.csv", date_format="%Y-%m-%d")
    y_train.to_csv(PROC_DIR / "y_train.csv", header=True, date_format="%Y-%m-%d")
    y_test.to_csv(PROC_DIR / "y_test.csv", header=True, date_format="%Y-%m-%d")

    this_week = full["primary_close_this_week"]
    this_week.to_csv(
        PROC_DIR / "primary_close_this_week_aligned.csv", header=True, date_format="%Y-%m-%d"
    )

    with open(PROC_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(kept_features, f, indent=2)

    if vif_reports:
        vif_reports[-1].to_csv(REPORT_DIR / "vif_final.csv", index=False)

    meta = {
        "n_train": int(len(X_train_s)),
        "n_test": int(len(X_test_s)),
        "n_features": len(kept_features),
        "train_index_start": str(X_train_s.index.min().date()),
        "train_index_end": str(X_train_s.index.max().date()),
        "test_index_start": str(X_test_s.index.min().date()),
        "test_index_end": str(X_test_s.index.max().date()),
        "features": kept_features,
        "unscaled_columns": [c for c in kept_features if c == PRIMARY_CLOSE],
    }
    with open(PROC_DIR / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Dataset ready:", json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=2))

    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train_s.corr(), cmap="vlag", center=0, square=False)
    plt.title("Training-set feature correlation (scaled + raw primary close)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_train.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
