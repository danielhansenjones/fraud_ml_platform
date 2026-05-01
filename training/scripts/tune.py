from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from training.src.tune import create_or_load_study, run_study

ARTIFACTS = Path("training/artifacts")
OPTUNA_DIR = Path("training/optuna")
PLOTS = ARTIFACTS / "plots"
N_TRIALS = 200
EXCLUDE_FROM_FEATURES = {"TransactionID", "TransactionDT", "isFraud", "day"}


def main() -> None:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")

    with open(ARTIFACTS / "pruned_features.json") as f:
        pruned_features = json.load(f)

    feature_cols = [
        c for c in pruned_features
        if c in train_df.columns
        and c not in EXCLUDE_FROM_FEATURES
        and train_df[c].dtype.name != "category"
    ]
    X = train_df[feature_cols].fillna(-1)
    y = train_df["isFraud"]
    dt = train_df["TransactionDT"]

    study = create_or_load_study(str(OPTUNA_DIR / "study.sqlite"))
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - completed)
    print(f"Completed trials: {completed}, running {remaining} more")

    if remaining > 0:
        run_study(study, X, y, dt, n_trials=remaining)

    best_params = study.best_params
    best_value = study.best_value
    print(f"Best PR-AUC: {best_value:.4f}")
    print("Best params:", best_params)

    with open(ARTIFACTS / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(ARTIFACTS / "study_trials.csv", index=False)

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.get_figure().savefig(PLOTS / "optuna_history.png", dpi=150)
        plt.close("all")

        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.get_figure().savefig(PLOTS / "optuna_param_importances.png", dpi=150)
        plt.close("all")
    except Exception as e:
        print(f"Warning: could not render Optuna plots: {e}")


if __name__ == "__main__":
    main()
