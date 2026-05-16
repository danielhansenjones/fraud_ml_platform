from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from training.src.splits import PurgedTimeSeriesSplit


_METRICS_KEYS = ["pr_auc", "roc_auc", "f1", "brier"]
_XGB_DEVICE = os.environ.get("XGB_DEVICE", "cuda")
# LightGBM CUDA mishandles the baseline's max_depth=8 + num_leaves=63 + no-early-stop
# combination (verified: PR-AUC drops to ~0.04, vs 0.57 on CPU and 0.62 with the
# challenger's config on CUDA). Pinning the baseline to CPU; challenger keeps CUDA.
_LGB_DEVICE = os.environ.get("LGB_DEVICE", "cpu")

# sklearn LR and RF have no GPU path; subsample to keep baseline CV runtime reasonable.
_LR_RF_SAMPLE = int(os.environ.get("LR_RF_SAMPLE", "100000"))


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _cv_scores(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, gap_days=1)
    fold_metrics: list[dict[str, float]] = []

    for tr_idx, val_idx in splitter.split(X, y, transaction_dt):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]
        fold_metrics.append(_compute_metrics(y_val.values, probs))

    result: dict[str, dict[str, float]] = {}
    for k in _METRICS_KEYS:
        vals = [fm[k] for fm in fold_metrics]
        result[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return result


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    num = X.select_dtypes(include=[np.number])
    return num.columns[num.notna().any()].tolist()


def _stratified_sample(
    X: pd.DataFrame, y: pd.Series, transaction_dt: pd.Series, n: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(X) <= n:
        return X, y, transaction_dt
    # Stratified sample preserves fraud rate and temporal order so CV splits stay valid.
    idx = (
        y.reset_index(drop=True)
        .groupby(y.values, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), int(n * len(g) / len(y))), random_state=42))
        .index
    )
    idx = sorted(idx)
    return X.iloc[idx], y.iloc[idx], transaction_dt.iloc[idx]


def run_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
) -> dict[str, dict[str, float]]:
    X_num = X[_numeric_cols(X)].copy()
    X_s, y_s, dt_s = _stratified_sample(X_num, y, transaction_dt, _LR_RF_SAMPLE)
    print(f"  LR sample: {len(X_s):,} rows")
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return _cv_scores(clf, X_s, y_s, dt_s)


def run_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
) -> dict[str, dict[str, float]]:
    X_num = X[_numeric_cols(X)].copy()
    X_s, y_s, dt_s = _stratified_sample(X_num, y, transaction_dt, _LR_RF_SAMPLE)
    print(f"  RF sample: {len(X_s):,} rows")
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    return _cv_scores(clf, X_s, y_s, dt_s)


def run_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
) -> dict[str, dict[str, float]]:
    # LightGBM has native NaN handling; matches the final training and serving policy.
    pos = y.sum()
    neg = len(y) - pos
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=float(neg / max(pos, 1)),
        device_type=_LGB_DEVICE,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    return _cv_scores(clf, X, y, transaction_dt)


def run_xgboost_untuned(
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
) -> dict[str, dict[str, float]]:
    pos = y.sum()
    neg = len(y) - pos
    clf = xgb.XGBClassifier(
        tree_method="hist",
        device=_XGB_DEVICE,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(neg / max(pos, 1)),
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    return _cv_scores(clf, X, y, transaction_dt)
