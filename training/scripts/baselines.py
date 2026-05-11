from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.src.baselines import (
    run_lightgbm,
    run_logistic_regression,
    run_random_forest,
    run_xgboost_untuned,
)
from training.src.constants import EXCLUDE_FROM_FEATURES

ARTIFACTS = Path("training/artifacts")


def _load_features(train_df: pd.DataFrame, pruned_features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # _encoded int versions carry the same info as Categorical cols and work with fillna(-1) across all backends
    feature_cols = [
        c for c in pruned_features
        if c in train_df.columns
        and c not in EXCLUDE_FROM_FEATURES
        and train_df[c].dtype.name != "category"
    ]
    X = train_df[feature_cols].copy()
    y = train_df["isFraud"]
    dt = train_df["TransactionDT"]
    return X, y, dt


def main() -> None:
    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")

    with open(ARTIFACTS / "pruned_features.json") as f:
        pruned_features = json.load(f)

    X, y, dt = _load_features(train_df, pruned_features)
    print(f"Training data: {X.shape}, fraud rate: {y.mean():.4f}")

    results = {}

    print("Running logistic regression...")
    results["logistic_regression"] = run_logistic_regression(X, y, dt)
    print("  PR-AUC:", results["logistic_regression"]["pr_auc"])

    print("Running random forest...")
    results["random_forest"] = run_random_forest(X, y, dt)
    print("  PR-AUC:", results["random_forest"]["pr_auc"])

    print("Running LightGBM...")
    results["lightgbm"] = run_lightgbm(X, y, dt)
    print("  PR-AUC:", results["lightgbm"]["pr_auc"])

    print("Running XGBoost (untuned)...")
    results["xgboost_untuned"] = run_xgboost_untuned(X, y, dt)
    print("  PR-AUC:", results["xgboost_untuned"]["pr_auc"])

    with open(ARTIFACTS / "baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBaseline summary (PR-AUC mean):")
    for name, scores in results.items():
        print(f"  {name}: {scores['pr_auc']['mean']:.4f} +/- {scores['pr_auc']['std']:.4f}")


if __name__ == "__main__":
    main()
