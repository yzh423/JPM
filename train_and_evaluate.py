"""
Train NPB + OLS / Ridge / Lasso / RandomForest / XGBoost; tune with TimeSeriesSplit CV.
Evaluate on held-out test with RMSE, MAE, R²; save metrics and feature importance.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

import config

PROC_DIR = Path(__file__).resolve().parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "models"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

PRIMARY_CLOSE = f"{config.PRIMARY_KEY}_Close"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_arrays():
    X_tr = pd.read_csv(PROC_DIR / "X_train_scaled.csv", index_col=0, parse_dates=True)
    X_te = pd.read_csv(PROC_DIR / "X_test_scaled.csv", index_col=0, parse_dates=True)
    y_tr = pd.read_csv(PROC_DIR / "y_train.csv", index_col=0, parse_dates=True).iloc[:, 0]
    y_te = pd.read_csv(PROC_DIR / "y_test.csv", index_col=0, parse_dates=True).iloc[:, 0]
    close = pd.read_csv(
        PROC_DIR / "primary_close_this_week_aligned.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    return X_tr, X_te, y_tr, y_te, close


def main() -> None:
    if not (PROC_DIR / "X_train_scaled.csv").exists():
        print("Run build_dataset.py first.", file=sys.stderr)
        raise SystemExit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, close_aligned = load_arrays()
    features = list(X_train.columns)

    close_test = close_aligned.reindex(y_test.index)
    if close_test.isna().any():
        close_test = close_test.ffill()
    pred_npb = close_test.values
    npb_metrics = metrics_dict(y_test.values, pred_npb)

    tscv = TimeSeriesSplit(n_splits=5)

    models: dict = {
        "ols": LinearRegression(),
        "ridge": GridSearchCV(
            Ridge(random_state=config.RANDOM_STATE),
            {"alpha": np.logspace(-3, 2, 20)},
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "lasso": GridSearchCV(
            Lasso(random_state=config.RANDOM_STATE, max_iter=10000),
            {"alpha": np.logspace(-4, 1, 20)},
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "rf": GridSearchCV(
            RandomForestRegressor(random_state=config.RANDOM_STATE),
            {
                "n_estimators": [100, 300],
                "max_depth": [4, 8, 12, None],
            },
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "xgb": GridSearchCV(
            XGBRegressor(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror",
            ),
            {
                "learning_rate": [0.03, 0.1],
                "max_depth": [3, 5, 7],
                "n_estimators": [200, 400],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
    }

    rows = [{"model": "NPB", **npb_metrics, "vs_npb_rmse_pct": 0.0}]

    tree_residual = {"rf", "xgb"}

    for name, est in models.items():
        if name in tree_residual:
            y_fit = (y_train - X_train[PRIMARY_CLOSE]).values
        else:
            y_fit = y_train.values
        est.fit(X_train.values, y_fit)
        best = est.best_estimator_ if hasattr(est, "best_estimator_") else est
        if name in tree_residual:
            pred_te = X_test[PRIMARY_CLOSE].values + best.predict(X_test.values)
        else:
            pred_te = best.predict(X_test.values)
        m = metrics_dict(y_test.values, pred_te)
        imp = (npb_metrics["RMSE"] - m["RMSE"]) / npb_metrics["RMSE"] * 100.0
        row = {"model": name.upper() if name != "xgb" else "XGBoost", **m, "vs_npb_rmse_pct": imp}
        rows.append(row)

        out_p = MODEL_DIR / f"{name}_model.pkl"
        with open(out_p, "wb") as f:
            pickle.dump(best, f)

        if hasattr(best, "coef_"):
            imp_df = pd.DataFrame({"feature": features, "coef": best.coef_.ravel()}).sort_values(
                "coef", key=np.abs, ascending=False
            )
            imp_df.to_csv(REPORT_DIR / f"importance_{name}_linear.csv", index=False)
        elif hasattr(best, "feature_importances_"):
            imp_df = pd.DataFrame(
                {"feature": features, "importance": best.feature_importances_}
            ).sort_values("importance", ascending=False)
            imp_df.to_csv(REPORT_DIR / f"importance_{name}.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORT_DIR / "test_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    best_row = metrics_df[metrics_df["model"] != "NPB"].sort_values("RMSE").iloc[0]
    print("\nBest model by test RMSE:", best_row["model"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test.index, y_test.values, label="Actual", color="black", linewidth=1.2)
    ax.plot(y_test.index, pred_npb, label="NPB", alpha=0.7)
    ax.set_title(f"{config.TARGET_TICKER} next-week close: test actual vs NPB")
    ax.legend()
    ax.set_ylabel("Price (USD)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "test_actual_vs_npb.png", dpi=150)
    plt.close(fig)

    summary = {
        "npb_test": npb_metrics,
        "all_models_test": rows,
        "success_criteria_notes": {
            "rmse_15pct_vs_npb": "Proposal target: RMSE 15% below NPB; weekly level persistence often caps gains near ~2–5%.",
            "r2_interpretation": "High R2 on price level is common when lagged close is a feature; it does not imply easy economic predictability.",
        },
    }
    with open(REPORT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
