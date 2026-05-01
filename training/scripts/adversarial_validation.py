from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.src.adversarial import run_adversarial_validation

ARTIFACTS = Path("training/artifacts")
N_DRIFT_DROP = 10


def main() -> None:
    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")

    with open(ARTIFACTS / "feature_list.json") as f:
        feature_list = json.load(f)
    all_features = [e["name"] for e in feature_list]

    mean_auc, importances = run_adversarial_validation(train_df, test_df)
    print(f"Adversarial validation AUC: {mean_auc:.4f}")

    with open(ARTIFACTS / "adversarial_auc.json", "w") as f:
        json.dump({"mean_auc": mean_auc}, f, indent=2)

    drift_features = importances.head(30).to_dict()
    with open(ARTIFACTS / "drift_features.json", "w") as f:
        json.dump({str(k): float(v) for k, v in drift_features.items()}, f, indent=2)

    top_drift = list(importances.head(N_DRIFT_DROP).index)
    pruned = [f for f in all_features if f not in top_drift]
    with open(ARTIFACTS / "pruned_features.json", "w") as f:
        json.dump(pruned, f, indent=2)

    print(f"Pruned {len(top_drift)} drift features. Remaining: {len(pruned)}")
    print("Top drift features:", top_drift[:5])


if __name__ == "__main__":
    main()
