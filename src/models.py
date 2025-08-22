from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score

def make_logreg_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LogisticRegression(C=C, max_iter=1000))
    ])

def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold
    }
    return m

def fit_and_eval(pipeline: Pipeline,
                 X_train: pd.DataFrame, y_train: np.ndarray,
                 X_val: pd.DataFrame, y_val: np.ndarray,
                 X_test: pd.DataFrame, y_test: np.ndarray,
                 threshold: float = 0.5):
    pipe = pipeline.fit(X_train, y_train)

    y_prob_test = pipe.predict_proba(X_test)[:, 1]
    metrics_test = evaluate_classifier(y_test, y_prob_test, threshold=threshold)

    y_prob_val = pipe.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    metrics_val = evaluate_classifier(y_val, y_prob_val, threshold=threshold) if len(X_val) else None

    return {"val": metrics_val, "test": metrics_test}, y_prob_test, pipe
