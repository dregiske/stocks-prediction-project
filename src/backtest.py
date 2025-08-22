from __future__ import annotations
import numpy as np
import pandas as pd

def _ensure_series(x) -> pd.Series:
    """
    Return a 1D series
    (handle accidental 2D arrays/frames).
    """
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    if isinstance(x, np.ndarray):
        return pd.Series(x.ravel())
    return x

def backtest_long_only(df_prices: pd.DataFrame,
                       y_test: np.ndarray,
                       y_prob_test: np.ndarray,
                       test_index: pd.DatetimeIndex,
                       prob_threshold: float = 0.5) -> pd.DataFrame:
    """
    Simple next-day long-or-cash backtest using predicted probability (Up).
    """
    # 1) Close series aligned to test_index
    close = _ensure_series(df_prices.loc[test_index, "Close"]).astype(float)
    close.index = pd.DatetimeIndex(close.index)

    # 2) Next-day return for holding from t to t+1
    r1 = close.pct_change().shift(-1)   # return realized from t -> t+1
    r1 = r1.dropna()

    # 3) Predicted probabilities as Series aligned to test_index
    prob = pd.Series(np.asarray(y_prob_test).ravel(), index=test_index)
    prob.index = pd.DatetimeIndex(prob.index)

    # 4) Trading signal (long if prob >= threshold), reindexed to r1 dates
    sig = (prob >= prob_threshold).astype(int).reindex(r1.index).fillna(0).astype(int)

    # 5) Strategy returns and equity curves (index = r1.index)
    strat_ret = (sig * r1).astype(float)
    eq_curve  = (1.0 + strat_ret).cumprod()
    bh_curve  = (1.0 + r1).cumprod()

    # 6) Build output frame with a shared index (avoids 2D/broadcast issues)
    out = pd.DataFrame({
        "signal": sig.astype(int).values,
        "strategy_ret": strat_ret.values,
        "strategy_eq": eq_curve.values,
        "buy_hold_eq": bh_curve.values,
    }, index=r1.index)
    return out

def summarize_backtest(bt_df: pd.DataFrame) -> dict:
    eq = bt_df["strategy_eq"].dropna()
    bh = bt_df["buy_hold_eq"].dropna()

    # Max drawdown
    roll_max = eq.cummax()
    drawdown = (eq / roll_max) - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    total_ret = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    bh_ret = float(bh.iloc[-1] - 1.0) if len(bh) else 0.0

    return {
        "strategy_total_return": total_ret,
        "buy_hold_total_return": bh_ret,
        "max_drawdown": max_dd
    }
