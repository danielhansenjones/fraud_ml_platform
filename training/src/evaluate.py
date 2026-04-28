from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import xgboost as xgb


class _IsotonicCalibrated:
    """Wraps an XGBClassifier with an isotonic calibration layer fitted on held-out data."""

    def __init__(self, clf: xgb.XGBClassifier, calibrator: IsotonicRegression):
        self.clf = clf
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])


def recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.95
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    mask = precision >= min_precision
    if mask.sum() == 0:
        return 0.0
    return float(recall[mask].max())


def precision_at_recall(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.50
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    mask = recall >= min_recall
    if mask.sum() == 0:
        return 0.0
    return float(precision[mask].max())


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "recall_at_95_precision": recall_at_precision(y_true, y_prob, 0.95),
        "precision_at_50_recall": precision_at_recall(y_true, y_prob, 0.50),
        "threshold": threshold,
    }


def calibrate(
    clf: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> _IsotonicCalibrated:
    val_probs = clf.predict_proba(X_val)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(val_probs, y_val.values)
    return _IsotonicCalibrated(clf, ir)
