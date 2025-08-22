#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from src.data import load_prices
from src.features import make_features_and_labels, split_X_y
from src.split import time_range_split
from src.models import make_logreg_pipeline, fit_and_eval
from src.backtest import backtest_long_only, summarize_backtest
from src.utils import save_json, plot_confusion, plot_equity

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate stock trend classifier (MVP).")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--train-end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--val-end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    p.add_argument("--outputs", type=str, default="outputs")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.outputs); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data → features
    prices = load_prices(args.ticker, args.start, args.end)
    features = make_features_and_labels(prices)
    features.index.name = "Date"

    # 2) Temporal split
    train_df, val_df, test_df = time_range_split(features, args.train_end, args.val_end)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val     = split_X_y(val_df)
    X_test, y_test   = split_X_y(test_df)

    # 3) Model → metrics
    pipe = make_logreg_pipeline(C=1.0)
    metrics, y_prob_test, _ = fit_and_eval(pipe, X_train, y_train, X_val, y_val, X_test, y_test, threshold=args.prob_threshold)

    # 4) Backtest on test set
    bt_df = backtest_long_only(prices, y_test, y_prob_test, X_test.index, prob_threshold=args.prob_threshold)
    bt_summary = summarize_backtest(bt_df)

    # 5) Save artifacts
    results = {
        "ticker": args.ticker,
        "periods": {
            "train_end": args.train_end,
            "val_end": args.val_end,
            "data_start": args.start,
            "data_end": args.end
        },
        "metrics": metrics,
        "backtest": bt_summary
    }
    save_json(results, str(out_dir / "metrics.json"))

    if metrics.get("test") and metrics["test"].get("confusion_matrix"):
        plot_confusion(metrics["test"]["confusion_matrix"], path=str(out_dir / "confusion_matrix.png"))
    plot_equity(bt_df, path=str(out_dir / "equity_curve.png"))

    print("Done. See outputs/ for metrics and plots.")

if __name__ == "__main__":
    main()
