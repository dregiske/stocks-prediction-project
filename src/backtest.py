from __future__ import annotations
import numpy as np
import pandas as pd

def backtest_long_only(df_prices: pd.DataFrame,
                       y_test: np.ndarray,
                       y_prob_test: np.ndarray,
                       test_index: pd.DatetimeIndex,
                       prob_threshold: float = 0.5) -> pd.DataFrame:
    """
    Simple next-day long-or-cash backtest using predicted probability (Up).
    """
    close = df_prices.loc[test_index, "Close"].copy()
    # Next day return for holding from t to t+1
    r1 = close.pct_change().shift(-1).dropna()

    sig = (pd.Series(y_prob_test, index=test_index) >= prob_threshold).astype(int)
    sig = sig.reindex(r1.index).fillna(0)

    strat_ret = sig * r1
    eq_curve = (1.0 + strat_ret).cumprod()
    bh_curve = (1.0 + r1).cumprod()

    out = pd.DataFrame({
        "signal": sig,
        "strategy_ret": strat_ret,
        "strategy_eq": eq_curve,
        "buy_hold_eq": bh_curve
    })
    return out

def summarize_backtest(bt_df: pd.DataFrame) -> dict:
    eq = bt_df["strategy_eq"].dropna()
    bh = bt_df["buy_hold_eq"].dropna()
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
