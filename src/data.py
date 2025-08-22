from __future__ import annotations
import os
import pandas as pd

def load_prices(ticker: str, start: str, end: str, cache_dir: str = "data", use_cache: bool = True) -> pd.DataFrame:
    """
    Download daily OHLCV with yfinance. Caches to CSV under data/.
    """
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
    if use_cache and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        return df

    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed. Run `pip install yfinance`.") from e

    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.to_csv(csv_path)
    return df
