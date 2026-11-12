"""Shared data-prep + feature logic (same as the notebook) used by the app and the asset script."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_PATH = BASE / "market_signals.csv"
VALUE_COLS = ["dc_total_sales", "dc_total_orders", "dc_avg_discount",
              "dc_late_rate", "dc_avg_profit_ratio", "dc_total_benefit"]
LAST_COMPLETE_MONTH = pd.Timestamp("2017-09-01")   # after this the source data is partial
FORECAST_MONTH = LAST_COMPLETE_MONTH + pd.offsets.MonthBegin(1)


def load_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df["category"] = df["category"].str.strip()
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df = df[(df.dc_total_sales > 0) & (df.year_month <= LAST_COMPLETE_MONTH)].copy()
    counts = df.groupby("category").size()
    df = df[df.category.isin(counts[counts >= 24].index)]
    return df.sort_values(["category", "year_month"]).reset_index(drop=True)


def build_features(frame):
    """One row per category-month (t) with lag features and next-month targets."""
    parts = []
    for cat, g in frame.groupby("category"):
        g = g.set_index("year_month")[VALUE_COLS].asfreq("MS")      # fill missing months with NaN
        out = g.copy()
        for c in VALUE_COLS:
            out[f"{c}_lag1"] = g[c].shift(1)
            out[f"{c}_lag2"] = g[c].shift(2)
        out["sales_roll3"] = g["dc_total_sales"].rolling(3, min_periods=3).mean()
        out["target_next_sales"] = g["dc_total_sales"].shift(-1)
        out["target_next_late_rate"] = g["dc_late_rate"].shift(-1)
        out["category"] = cat
        out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
        out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
        parts.append(out.reset_index())
    return pd.concat(parts, ignore_index=True)


def feature_columns(feats):
    lags = [c for c in feats.columns if c.endswith("_lag1") or c.endswith("_lag2")]
    return ["category"] + VALUE_COLS + lags + ["sales_roll3", "month_sin", "month_cos"]


def training_rows(feats):
    needed = VALUE_COLS + [c for c in feats.columns if c.endswith("_lag1") or c.endswith("_lag2")] \
             + ["target_next_sales", "target_next_late_rate"]
    return feats.dropna(subset=needed).reset_index(drop=True)


def forecast_rows(feats, cols):
    """Rows for LAST_COMPLETE_MONTH that have every feature available -> forecast of FORECAST_MONTH."""
    r = feats[feats.year_month == LAST_COMPLETE_MONTH]
    return r.dropna(subset=[c for c in cols if c != "category"]).reset_index(drop=True)
