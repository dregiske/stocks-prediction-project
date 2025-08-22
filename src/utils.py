from __future__ import annotations
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

def save_json(obj: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_confusion(cm, labels=("Down","Up"), path: Optional[str]=None):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,4))
    cm = np.array(cm)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_xticklabels(labels)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Pred")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_equity(bt_df: pd.DataFrame, path: Optional[str]=None):
    fig, ax = plt.subplots(figsize=(8,4))
    bt_df["strategy_eq"].plot(ax=ax, label="Strategy")
    bt_df["buy_hold_eq"].plot(ax=ax, label="Buy & Hold")
    ax.set_title("Equity Curve (Test)")
    ax.set_ylabel("Equity (normalized)")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)
