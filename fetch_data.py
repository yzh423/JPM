"""
Download weekly OHLCV (yfinance) and merge official macro series from FRED (graph CSV).

Equities / futures / FTSE: Yahoo Finance aggregation (yfinance).
Treasury 10Y, VIX, S&P 500, EUR/USD, GBP/USD: Federal Reserve Economic Data (FRED).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config

OUT_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_yf_weekly(symbol: str, name: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            start=config.START_DATE,
            end=config.END_DATE,
            interval="1wk",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"[warn] yfinance failed {symbol} ({name}): {e}", file=sys.stderr)
        return None
    if df is None or df.empty:
        print(f"[warn] empty series {symbol} ({name})", file=sys.stderr)
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(
        columns={
            "Open": f"{name}_Open",
            "High": f"{name}_High",
            "Low": f"{name}_Low",
            "Close": f"{name}_Close",
            "Adj Close": f"{name}_AdjClose",
            "Volume": f"{name}_Volume",
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def fetch_fred_daily_series(series_id: str) -> pd.Series:
    url = f"{FRED_GRAPH}?id={series_id}"
    df = pd.read_csv(url)
    if "observation_date" not in df.columns or len(df.columns) < 2:
        raise ValueError(f"Unexpected FRED CSV shape for {series_id}")
    value_col = [c for c in df.columns if c != "observation_date"][0]
    idx = pd.to_datetime(df["observation_date"])
    vals = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    s = pd.Series(vals.values, index=idx, name=value_col).sort_index()
    start = pd.Timestamp(config.START_DATE)
    end = pd.Timestamp(config.END_DATE)
    return s.loc[(s.index >= start) & (s.index <= end)]


def fred_daily_to_weekly_close(s: pd.Series) -> pd.Series:
    """Last observation of each Fri-ended week (aligns with typical US weekly equity bars)."""
    return s.resample("W-FRI").last()


def build_fred_weekly_frame() -> pd.DataFrame:
    frames: list[pd.Series] = []
    for series_id, col_name in config.FRED_DAILY_CLOSE.items():
        daily = fetch_fred_daily_series(series_id)
        w = fred_daily_to_weekly_close(daily).rename(col_name)
        frames.append(w)
    out = pd.concat(frames, axis=1)
    return out.sort_index()


def main() -> None:
    frames: list[pd.DataFrame] = []
    for key, sym in config.TICKERS_YF.items():
        part = download_yf_weekly(sym, key)
        if part is not None:
            frames.append(part)
    if not frames:
        raise SystemExit("No Yahoo Finance data downloaded.")
    merged = frames[0]
    for nxt in frames[1:]:
        merged = merged.join(nxt, how="outer")
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    fred_w = build_fred_weekly_frame()
    merged = merged.join(fred_w, how="outer")
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    out_path = OUT_DIR / "weekly_raw_merged.csv"
    merged.to_csv(out_path, date_format="%Y-%m-%d")
    print(f"Wrote {out_path} shape={merged.shape}")
    print("Sources: yfinance:", ", ".join(f"{k}={v}" for k, v in config.TICKERS_YF.items()))
    print("Sources: FRED:", ", ".join(config.FRED_DAILY_CLOSE.keys()))


if __name__ == "__main__":
    main()
