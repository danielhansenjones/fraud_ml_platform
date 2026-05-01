from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


EXCLUDE_COLS = {"TransactionDT", "TransactionID", "isFraud", "day"}


def _xgb_device() -> str:
    return os.environ.get("XGB_DEVICE", "cuda")


def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, pd.Series]:
    feature_cols = [
        c for c in train_df.columns
        if c not in EXCLUDE_COLS and not c.endswith("_encoded")
    ]
    encoded = [c + "_encoded" for c in feature_cols if c + "_encoded" in train_df.columns]
    raw_cats = [c for c in feature_cols if c + "_encoded" not in train_df.columns]
    use_cols = raw_cats + encoded

    train_X = train_df[use_cols].copy()
    test_X = test_df[use_cols].copy()

    X = pd.concat([train_X, test_X], ignore_index=True)
    y = np.array([1] * len(train_X) + [0] * len(test_X))

    X = X.fillna(-1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    importances = pd.Series(0.0, index=use_cols)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = xgb.XGBClassifier(
            tree_method="hist",
            device=_xgb_device(),
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        preds = clf.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y[val_idx], preds)
        aucs.append(auc)

        imp = pd.Series(clf.get_booster().get_score(importance_type="gain"))
        importances = importances.add(imp, fill_value=0)

    importances /= n_splits
    importances = importances.sort_values(ascending=False)
    return float(np.mean(aucs)), importances
