from __future__ import annotations
import pandas as pd

def time_range_split(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Split df (indexed by Date) into train, val, test by date ranges.
    train: up to train_end (inclusive)
    val:   (train_end, val_end]
    test:  (val_end, end]
    """
    assert df.index.name == "Date", "DataFrame must be indexed by Date."
    train = df.loc[:train_end]
    val   = df.loc[train_end:].loc[lambda d: (d.index > train_end) & (d.index <= val_end)]
    test  = df.loc[val_end:].loc[lambda d: d.index > val_end]
    return train, val, test
