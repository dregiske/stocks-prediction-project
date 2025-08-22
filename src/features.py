from __future__ import annotations
import pandas as pd
import numpy as np

def make_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, leakage-safe features + next-day direction label.
    Returns a DataFrame indexed by Date with feature cols and 'y'.
    """
    df = prices.copy()

    # Basic returns
    df["r1"] = df["Close"].pct_change(1)

    # Moving averages
    df["SMA_5"]  = df["Close"].rolling(5,  min_periods=5).mean()
    df["SMA_10"] = df["Close"].rolling(10, min_periods=10).mean()
    df["SMA_gap"] = (df["SMA_5"] - df["SMA_10"]) / df["Close"]

    # Volatility features
    df["Vol_10"] = df["r1"].rolling(10, min_periods=10).std()
    df["Vol_chg"] = df["Vol_10"] / df["Vol_10"].shift(10) - 1

    # Volume z-score (10-day)
    vol_mean_10 = df["Volume"].rolling(10, min_periods=10).mean()
    vol_std_10  = df["Volume"].rolling(10, min_periods=10).std()
    df["Volume_z10"] = (df["Volume"] - vol_mean_10) / vol_std_10

    # Label: Up tomorrow?
    close_fwd = df["Close"].shift(-1)
    df["y"] = (close_fwd > df["Close"]).astype(int)

    # Drop warm-up & last row (no future label)
    df = df.dropna().copy()

    feature_cols = ["r1","SMA_5","SMA_10","SMA_gap","Vol_10","Vol_chg","Volume_z10"]
    return df[feature_cols + ["y"]]

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"].astype(int).values
    return X, y
