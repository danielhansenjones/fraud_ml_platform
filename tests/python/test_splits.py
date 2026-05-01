from __future__ import annotations

import numpy as np
import pandas as pd

from training.src.splits import PurgedTimeSeriesSplit


def _make_ts_data(n: int = 300, gap_days: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    dt = pd.Series(np.sort(rng.integers(0, 86400 * 90, n)))
    X = pd.DataFrame({"feature": rng.standard_normal(n)}, index=range(n))
    y = pd.Series(rng.integers(0, 2, n), index=range(n))
    return X, y, dt


def test_gap_enforced():
    gap_days = 2
    splitter = PurgedTimeSeriesSplit(n_splits=3, gap_days=gap_days)
    X, y, dt = _make_ts_data(n=300, gap_days=gap_days)

    for tr_idx, val_idx in splitter.split(X, y, dt):
        tr_dt_max = dt.iloc[tr_idx].max()
        val_dt_min = dt.iloc[val_idx].min()
        assert val_dt_min >= tr_dt_max + gap_days * 86400, (
            f"Gap violated: train max={tr_dt_max}, val min={val_dt_min}"
        )


def test_no_overlap_between_folds():
    splitter = PurgedTimeSeriesSplit(n_splits=4, gap_days=1)
    X, y, dt = _make_ts_data(n=400)

    val_sets = []
    for _, val_idx in splitter.split(X, y, dt):
        val_sets.append(set(val_idx.tolist()))

    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            assert val_sets[i].isdisjoint(val_sets[j]), (
                f"Fold {i} and {j} have overlapping validation indices"
            )


def test_validation_folds_monotonically_later():
    splitter = PurgedTimeSeriesSplit(n_splits=3, gap_days=1)
    X, y, dt = _make_ts_data(n=300)

    fold_val_mins = []
    for _, val_idx in splitter.split(X, y, dt):
        fold_val_mins.append(dt.iloc[val_idx].min())

    for i in range(1, len(fold_val_mins)):
        assert fold_val_mins[i] >= fold_val_mins[i - 1], "Validation folds not monotonically later"


def test_train_always_before_val():
    splitter = PurgedTimeSeriesSplit(n_splits=3, gap_days=1)
    X, y, dt = _make_ts_data(n=300)

    for tr_idx, val_idx in splitter.split(X, y, dt):
        assert dt.iloc[tr_idx].max() < dt.iloc[val_idx].min() + 1


def test_yields_some_folds():
    splitter = PurgedTimeSeriesSplit(n_splits=5, gap_days=1)
    X, y, dt = _make_ts_data(n=300)
    folds = list(splitter.split(X, y, dt))
    assert len(folds) > 0
