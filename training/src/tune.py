from __future__ import annotations

import os

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from training.src.splits import PurgedTimeSeriesSplit


def _xgb_device() -> str:
    return os.environ.get("XGB_DEVICE", "cuda")


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
    n_splits: int = 7,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # max_bin drives GPU histogram memory. Floor 512 keeps every trial
        # using the GPU heavily (target 12-14 GB on the 16 GB 5070 Ti).
        # Cap 768 still leaves headroom for max_depth=15 trees.
        "max_bin": trial.suggest_int("max_bin", 512, 768, step=64),
        # Class imbalance weighting: neg/pos ~= 27.5 was the v1 hardcoded value.
        # Optuna searches log-uniform around it.
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
        # Recommended by XGBoost docs whenever scale_pos_weight is high; caps
        # the per-step update size to stabilise gradients on the rare positives.
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
    }

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, gap_days=1)
    pr_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y, transaction_dt)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        clf = xgb.XGBClassifier(
            tree_method="hist",
            device=_xgb_device(),
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
            **params,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = clf.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val.values, probs))

        trial.report(float(np.mean(pr_aucs)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(pr_aucs))


def create_or_load_study(storage_path: str, study_name: str = "xgb_fraud_v2") -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=42)
    # Hyperband prunes weak trials at fold 1-2 instead of running the full
    # n_splits folds, freeing budget for promising configurations.
    # max_resource must match the number of fold reports in _objective.
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=7, reduction_factor=3)
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{storage_path}",
        study_name=study_name,
        load_if_exists=True,
    )


def run_study(
    study: optuna.Study,
    X: pd.DataFrame,
    y: pd.Series,
    transaction_dt: pd.Series,
    n_trials: int = 200,
    n_splits: int = 7,
) -> None:
    study.optimize(
        lambda trial: _objective(trial, X, y, transaction_dt, n_splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
