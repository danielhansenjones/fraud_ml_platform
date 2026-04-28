from __future__ import annotations

import os

import pandas as pd
import xgboost as xgb


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    early_stopping_rounds: int = 50,
) -> xgb.XGBClassifier:
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    clf = xgb.XGBClassifier(
        tree_method="hist",
        device=os.environ.get("XGB_DEVICE", "cuda"),
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        verbosity=1,
        **params,
    )
    clf.fit(
        X_train.astype("float32"),
        y_train,
        eval_set=[(X_val.astype("float32"), y_val)],
        verbose=100,
    )
    # Move booster to CPU for predict_proba calls - avoids device mismatch warning
    # when input data arrives as a CPU pandas DataFrame
    clf.set_params(device="cpu")
    return clf
