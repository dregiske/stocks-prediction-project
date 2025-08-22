from __future__ import annotations
import os
import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume"]

def _extract_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handle the different column shapes yfinance may return:
    - Wide single-index: columns like ['Open','High','Low','Close','Adj Close','Volume']
    - MultiIndex level 0 = fields, level 1 = tickers: ('Close','AAPL')
    - MultiIndex level 0 = tickers, level 1 = fields: ('AAPL','Close')
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # single index; just keep the fields that exist
        cols = [c for c in FIELDS if c in df.columns]
        out = df[cols].copy()
        return out

    # MultiIndex case
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # Case A: level 0 are fields (Open/High/...) and level 1 are tickers
    if set(FIELDS).issubset(set(lvl0)):
        # slice by ticker on level 1 → gives a DataFrame with level 0 = fields
        if ticker in set(lvl1):
            sub = df.xs(key=ticker, axis=1, level=1, drop_level=True)
            cols = [c for c in FIELDS if c in sub.columns]
            return sub[cols].copy()

    # Case B: level 0 are tickers, level 1 are fields
    if ticker in set(lvl0):
        sub = df.xs(key=ticker, axis=1, level=0, drop_level=True)
        cols = [c for c in FIELDS if c in sub.columns]
        return sub[cols].copy()

    # Fallback: try flatten by joining levels and look for fields
    flat = df.copy()
    flat.columns = ["|".join(map(str, t)) for t in flat.columns.to_list()]
    # Prefer exact field names if present
    out = pd.DataFrame(index=flat.index)
    for f in FIELDS:
        # possible names: 'Close', 'AAPL|Close', 'Close|AAPL'
        candidates = [c for c in flat.columns if c.endswith(f) or c.startswith(f)]
        if len(candidates) == 1:
            out[f] = flat[candidates[0]]
        elif len(candidates) > 1:
            # Prefer the one that mentions the ticker
            pick = [c for c in candidates if ticker in c] or candidates[:1]
            out[f] = flat[pick[0]]
    if len(out.columns):
        return out

    # If all else fails, return empty — caller will error
    return pd.DataFrame(index=df.index)

def _normalize_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub = _extract_single_ticker(df, ticker)
    if sub is None or sub.empty:
        return pd.DataFrame(index=df.index)

    # Coerce numerics & basic hygiene
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(how="any")
    sub.index.name = "Date"
    return sub

def load_prices(ticker: str, start: str, end: str,
                cache_dir: str = "data", use_cache: bool = True) -> pd.DataFrame:
    """
    Download daily OHLCV with yfinance, normalize columns for a single ticker, and cache as CSV.
    Robust to cache files with or without an explicit 'Date' column and to differing MultiIndex shapes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    # Try cache first
    if use_cache and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
        except Exception:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
            df.index.name = "Date"
        # Already normalized in our save step; just sanity check
        if not df.empty:
            return df

    # Fresh download (force consistent grouping)
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed. Run `pip install yfinance`.") from e

    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="column",   # ensures MultiIndex for some versions
        actions=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")

    df = _normalize_price_frame(raw, ticker)
    if df.empty:
        # Last-ditch: try without group_by (some envs behave differently)
        raw2 = yf.download(
            tickers=ticker, start=start, end=end, interval="1d",
            auto_adjust=False, group_by=False, actions=False, progress=False
        )
        if raw2 is None or raw2.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
        df = _normalize_price_frame(raw2, ticker)

    if df.empty:
        raise ValueError("Downloaded data normalized to empty frame (unexpected).")

    df.to_csv(csv_path, index_label="Date")
    return df
