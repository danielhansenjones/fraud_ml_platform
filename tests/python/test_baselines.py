"""Smoke test: baselines run on synthetic data without errors."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Force CPU for tests - no GPU required
os.environ.setdefault("XGB_DEVICE", "cpu")

from training.src.baselines import (
    run_lightgbm,
    run_logistic_regression,
    run_random_forest,
    run_xgboost_untuned,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(1)
    n = 600
    dt = pd.Series(np.sort(rng.integers(0, 86400 * 60, n)))
    X = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.uniform(0, 1, n),
        "f3": rng.standard_normal(n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y, dt


def _check_scores(scores: dict) -> None:
    for metric in ["pr_auc", "roc_auc", "f1", "brier"]:
        assert metric in scores
        assert "mean" in scores[metric]
        assert "std" in scores[metric]
        assert 0.0 <= scores[metric]["mean"] <= 1.0


def test_logistic_regression_smoke(synthetic_data):
    X, y, dt = synthetic_data
    scores = run_logistic_regression(X, y, dt)
    _check_scores(scores)


def test_random_forest_smoke(synthetic_data):
    X, y, dt = synthetic_data
    scores = run_random_forest(X, y, dt)
    _check_scores(scores)


def test_lightgbm_smoke(synthetic_data):
    X, y, dt = synthetic_data
    scores = run_lightgbm(X, y, dt)
    _check_scores(scores)


def test_xgboost_untuned_smoke(synthetic_data):
    X, y, dt = synthetic_data
    scores = run_xgboost_untuned(X, y, dt)
    _check_scores(scores)
