from __future__ import annotations

import warnings
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

        skipped = 0
        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(indices)):
            train_dt_max = transaction_dt.iloc[train_idx].max()
            val_mask = transaction_dt.iloc[val_idx] >= (train_dt_max + gap_seconds)
            purged_val = val_idx[val_mask.values]
            if len(purged_val) == 0:
                # Caller's "5-fold CV" silently became 4-fold (or fewer).
                # Surface that so downstream metric averages can't lie about
                # how many folds they actually averaged over.
                warnings.warn(
                    f"PurgedTimeSeriesSplit: fold {fold_i} dropped (gap purged all "
                    f"validation rows); effective fold count is reduced",
                    RuntimeWarning,
                    stacklevel=2,
                )
                skipped += 1
                continue
            yield train_idx, purged_val
        if skipped == self.n_splits:
            warnings.warn(
                "PurgedTimeSeriesSplit: every fold was dropped; "
                "consider reducing gap_days or increasing the data window",
                RuntimeWarning,
                stacklevel=2,
            )
