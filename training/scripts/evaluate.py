from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve

from training.src.constants import EXCLUDE_FROM_FEATURES
from training.src.evaluate import (
    calibrate,
    compute_metrics,
    find_optimal_threshold,
)
from training.src.train import fit_xgboost

ARTIFACTS = Path("training/artifacts")
PLOTS = ARTIFACTS / "plots"
NAMED_FEATURES = {"TransactionAmt", "TransactionAmt_log", "TransactionAmt_decimal",
                  "card1", "card2", "card3", "card4", "card5", "card6",
                  "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                  "ProductCD", "hour", "day_of_week", "email_match",
                  "p_email_tld", "r_email_tld"}


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")

    with open(ARTIFACTS / "pruned_features.json") as f:
        pruned_features = json.load(f)
    with open(ARTIFACTS / "best_params.json") as f:
        best_params = json.load(f)

    feature_cols = [
        c for c in pruned_features
        if c in train_df.columns
        and c not in EXCLUDE_FROM_FEATURES
        and train_df[c].dtype.name != "category"
    ]

    n = len(train_df)
    val_start = int(n * 0.9)
    tr = train_df.iloc[:val_start]
    val = train_df.iloc[val_start:]

    # XGBoost handles NaN natively; the resulting ONNX TreeEnsembleClassifier
    # carries that branch metadata and ORT routes NaN through the same path at
    # inference. Filling here would create a train-serve skew with features/client.go.
    X_tr = tr[feature_cols].astype("float32")
    y_tr = tr["isFraud"]
    X_val = val[feature_cols].astype("float32")
    y_val = val["isFraud"]
    X_test = test_df[feature_cols].astype("float32")
    y_test = test_df["isFraud"]

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    clf = fit_xgboost(X_tr, y_tr, X_val, y_val, best_params)
    clf.save_model(str(ARTIFACTS / "final_model.json"))

    val_probs = clf.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val.values, val_probs)
    print(f"Optimal threshold (val): {threshold:.4f}")

    test_probs_uncal = clf.predict_proba(X_test)[:, 1]
    metrics_uncal = compute_metrics(y_test.values, test_probs_uncal, threshold)
    print("Uncalibrated test metrics:", metrics_uncal)

    # Calibration is computed in-memory only: the export pipeline serves the
    # uncalibrated model (see README "Calibration negative result"). Persisting
    # the calibrator to disk would invite a future caller to load the wrong
    # artifact.
    cal_clf = calibrate(clf, X_val, y_val)
    test_probs_cal = cal_clf.predict_proba(X_test)[:, 1]
    metrics_cal = compute_metrics(y_test.values, test_probs_cal, threshold)
    print("Calibrated test metrics:", metrics_cal)

    results = {
        "feature_cols": feature_cols,
        "optimal_threshold": threshold,
        "uncalibrated": metrics_uncal,
        "calibrated": metrics_cal,
    }
    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, label in [
        (ax1, test_probs_uncal, "Uncalibrated"),
        (ax2, test_probs_cal, "Calibrated"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=20)
        ax.plot(mean_pred, frac_pos, "s-", label=label)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"Reliability diagram: {label}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS / "calibration_before_after.png", dpi=150)
    plt.close(fig)

    sample_size = min(20000, len(X_test))
    X_shap = X_test.sample(sample_size, random_state=42)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_parquet(ARTIFACTS / "shap_values.parquet", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    fig.savefig(PLOTS / "shap_summary.png", dpi=150)
    plt.close(fig)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top10 = [feature_cols[i] for i in sorted_idx[:10]]
    named_first = sorted(top10, key=lambda c: (0 if c in NAMED_FEATURES else 1, top10.index(c)))

    for feat in named_first[:10]:
        feat_idx = feature_cols.index(feat)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(feat_idx, shap_values, X_shap, ax=ax, show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        fig.savefig(PLOTS / f"shap_dependence_{safe_name}.png", dpi=150)
        plt.close(fig)

    print(f"Final calibrated PR-AUC: {metrics_cal['pr_auc']:.4f}")
    print("Evaluate complete.")


if __name__ == "__main__":
    main()
