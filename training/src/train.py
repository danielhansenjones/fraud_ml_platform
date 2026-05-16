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
    # scale_pos_weight and max_delta_step are now part of `params` (Optuna-tuned).
    clf = xgb.XGBClassifier(
        tree_method="hist",
        device=os.environ.get("XGB_DEVICE", "cuda"),
        eval_metric="aucpr",
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
    # predict_proba receives CPU pandas DataFrames; keep booster on CPU to avoid device mismatch.
    clf.set_params(device="cpu")
    return clf
