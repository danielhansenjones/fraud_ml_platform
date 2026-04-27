from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, gap_days: int = 1):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        transaction_dt: pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        gap_seconds = self.gap_days * 86400
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        indices = np.arange(len(X))

        for train_idx, val_idx in tscv.split(indices):
            train_dt_max = transaction_dt.iloc[train_idx].max()
            val_mask = transaction_dt.iloc[val_idx] >= (train_dt_max + gap_seconds)
            purged_val = val_idx[val_mask.values]
            if len(purged_val) == 0:
                continue
            yield train_idx, purged_val
