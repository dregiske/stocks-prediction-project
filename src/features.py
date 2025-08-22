from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_series(col) -> pd.Series:
    """Return a 1D Series even if col is a DataFrame."""
    if isinstance(col, pd.DataFrame):
        # take the first column if a 2D frame sneaks in
        return col.iloc[:, 0]
    return col

def make_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, leakage-safe features + next-day direction label.
    Returns a DataFrame indexed by Date with feature cols and 'y'.
    """
    df = prices.copy()

    # Ensure core columns are 1D series
    close = _ensure_series(df["Close"]).astype(float)
    volume = _ensure_series(df["Volume"]).astype(float)

    # Basic returns
    df["r1"] = close.pct_change(1)

    # Moving averages (force Series arithmetic)
    sma5  = close.rolling(5,  min_periods=5).mean()
    sma10 = close.rolling(10, min_periods=10).mean()
    df["SMA_5"]  = sma5
    df["SMA_10"] = sma10
    df["SMA_gap"] = (sma5 - sma10) / close

    # Volatility features
    vol10 = df["r1"].rolling(10, min_periods=10).std()
    df["Vol_10"]  = vol10
    df["Vol_chg"] = vol10 / vol10.shift(10) - 1

    # Volume z-score (10-day)
    vol_mean_10 = volume.rolling(10, min_periods=10).mean()
    vol_std_10  = volume.rolling(10, min_periods=10).std()
    df["Volume_z10"] = (volume - vol_mean_10) / vol_std_10

    # Label: Up tomorrow?
    close_fwd = close.shift(-1)
    df["y"] = (close_fwd > close).astype(int)

    # Drop warm-up & last row (no future label)
    df = df.dropna().copy()

    feature_cols = ["r1","SMA_5","SMA_10","SMA_gap","Vol_10","Vol_chg","Volume_z10"]
    return df[feature_cols + ["y"]]

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"].astype(int).values
    return X, y
