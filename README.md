1) Outline
Predict tomorrow’s direction of a stock (Up = close_{t+1} > close_t, Down otherwise) using daily OHLCV data and a few time-aware features. Evaluate with temporal splits and a tiny backtest to show practical impact.

2) Scope & Assumptions
Universe: 1–3 large-cap tickers (e.g., AAPL, MSFT).
Horizon: Next trading day (t → t+1).
Frequency: Daily.
Educational only (not financial advice). No intraday, no leverage, no shorting in MVP.

3) Tech Stack (MVP)
Python 3.10+
pandas, numpy (data)
scikit-learn (models, pipeline, metrics)
yfinance (data source)
matplotlib (plots)

4) Data Ingestion (MVP)
Pull daily OHLCV from yfinance for a fixed window (e.g., 2015-01-01 → 2024-12-31)
Keep columns: Date, Open, High, Low, Close, Volume.
Handle missing rows/dates; forward-fill if needed; drop leading NA after rolling features.

5) Labeling
y_t = 1 if Close[t+1] > Close[t] else 0.
Shift labels so features at time t predict label at t+1.
Drop last row (no future label).

6) Minimal Feature Set (no leakage)
Compute all with rolling windows; drop rows with NA from window warmups:
r1: 1-day return = Close.pct_change(1).
SMA_5, SMA_10: simple moving averages of Close.
SMA_gap: (SMA_5 - SMA_10) / Close.
Vol_10: rolling std of r1 over 10 days.
Vol_chg: Vol_10 / Vol_10.shift(10) - 1 (stability signal).
Volume_z10: z-score of volume over 10 days.
Keep it tiny at first (5–7 features). Add RSI/MACD later if needed.

7) Baseline & Models (keep small)
Baseline: predict majority class (always Up). Report accuracy for context.
Model 1: Logistic Regression (Pipeline: StandardScaler → LogReg(C=1.0, class_weight=None)).
Model 2 (optional for V2): RandomForest (no scaling needed) or SVM(RBF).

8) Time-Aware Validation
Holdout split by date (simple and clear):
Train: 2015–2021
Val: 2022
Test: 2023–2024
Scale inside each fit (use scikit-learn Pipeline).
Later (V2): expanding-window walk-forward CV.

9) Metrics (report just the essentials)
Accuracy, F1 (positive = Up), MCC (handles imbalance better).
Confusion Matrix on the test period.
(Optional V2) ROC-AUC and precision-recall.

10) Tiny Backtest (toy, for illustration)
Rule: if P(Up) ≥ 0.5 (or class=Up), go long for next day; else stay in cash.
Compute strategy daily returns and cumulative return.
Compare against Buy & Hold over the test window.
Show equity curve plot (two lines: strategy vs. B&H).
(V2) Add simple transaction cost (e.g., 5 bps) and a threshold (e.g., trade only if P(Up) ≥ 0.55).

11) Plots (MVP)
Price with buy/sit-out markers for the test period.
Confusion matrix.
Equity curve vs. buy-and-hold.

12) Repo Structure
stock-trend-ml/
├─ README.md
├─ requirements.txt           # pandas, numpy, scikit-learn, yfinance, matplotlib
├─ data/                      # (optional) cached CSVs
├─ notebooks/
│  └─ 01_eda_and_features.ipynb
├─ src/
│  ├─ data.py                 # download/cache data
│  ├─ features.py             # build features & labels
│  ├─ split.py                # time-based splits
│  ├─ models.py               # pipelines, train/eval
│  ├─ backtest.py             # simple strategy & equity curve
│  └─ utils.py                # common helpers (metrics, plotting)
├─ scripts/
│  ├─ train.py                # end-to-end: load → featurize → fit → evaluate
│  └─ backtest.py             # run backtest on test set; save plots
└─ outputs/
   ├─ metrics.json
   ├─ confusion_matrix.png
   └─ equity_curve.png

13) Minimal Execution Flow
python scripts/train.py --ticker AAPL --start 2015-01-01 --end 2024-12-31
Saves metrics.json, model artifact (optional), and validation results.
python scripts/backtest.py --ticker AAPL --test-start 2023-01-01 --test-end 2024-12-31
Produces equity_curve.png and summary stats.

14) Next Steps (only after MVP)
Add RSI/MACD and permutation importance.
Walk-forward CV; probability thresholds; calibration.
Compare SVM/RandomForest/XGBoost.
Streamlit dashboard; FastAPI endpoint.
Dockerfile + GitHub Actions.

15) Running:
# start environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# sanity check
python scripts/train.py --ticker AAPL --start 2022-01-01 --end 2024-12-31 \
  --train-end 2023-06-30 --val-end 2023-12-31

# full first run
python scripts/train.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 \
  --train-end 2021-12-31 --val-end 2022-12-30