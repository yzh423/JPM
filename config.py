"""Central configuration: JPM weekly close forecast; FRED (macro) + yfinance (equities)."""

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

# Target: JPMorgan Chase (NYSE), liquid US bank stock with long history.
TARGET_TICKER = "JPM"
# Column prefix for merged OHLCV (must match keys below).
PRIMARY_KEY = "primary"

# US-listed OHLCV via yfinance (Stooq CSV requires a manual API key for automation).
TICKERS_YF = {
    PRIMARY_KEY: "JPM",
    "bac": "BAC",
    "c": "C",
    "wfc": "WFC",
    "xlf": "XLF",
    "gold": "GC=F",
    "ftse": "^FTSE",
}

# FRED graph CSV (https://fred.stlouisfed.org/) — no API key; daily series → weekly in fetch.
# Maps FRED series id -> logical {name}_Close column name used in the pipeline.
FRED_DAILY_CLOSE = {
    "DGS10": "tnx_Close",
    "VIXCLS": "vix_Close",
    "SP500": "gspc_Close",
    "DEXUSEU": "eurusd_Close",
    "DEXUSUK": "gbpusd_Close",
}

TRAIN_FRAC = 0.8
RANDOM_STATE = 42
