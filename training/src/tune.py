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
    n_splits: int = 5,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, gap_days=1)
    pr_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y, transaction_dt)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        clf = xgb.XGBClassifier(
            tree_method="hist",
            device=_xgb_device(),
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,
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


def create_or_load_study(storage_path: str, study_name: str = "xgb_fraud_v1") -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
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
    n_splits: int = 5,
) -> None:
    study.optimize(
        lambda trial: _objective(trial, X, y, transaction_dt, n_splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
