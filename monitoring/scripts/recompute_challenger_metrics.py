from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd

from monitoring.common.encoding import encode_categoricals
from training.src.evaluate import compute_metrics, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / "training" / "artifacts"
CHALLENGER_DIR = ARTIFACTS / "challenger"
OPTUNA_DIR = ROOT / "training" / "optuna"


def main() -> None:
    cat_mappings = json.loads((CHALLENGER_DIR / "lgbm_category_mappings.json").read_text())
    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")
    feature_cols = [c for c in test_df.columns if c not in ("isFraud", "TransactionID")]
    train_enc, _ = encode_categoricals(train_df[feature_cols + ["isFraud"]], mappings=cat_mappings)
    test_enc, _ = encode_categoricals(test_df[feature_cols + ["isFraud"]], mappings=cat_mappings)

    # val is in-sample for the saved booster; same twin scheme as train_challenger.py.
    best_params = json.loads((CHALLENGER_DIR / "lgbm_best_params.json").read_text())
    val_start = int(len(train_enc) * 0.9)
    tr = train_enc.iloc[:val_start]
    val = train_enc.iloc[val_start:]
    twin = lgb.LGBMClassifier(**{**best_params, "device": "cpu"})
    twin.fit(
        tr[feature_cols],
        tr["isFraud"],
        eval_set=[(val[feature_cols], val["isFraud"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    threshold = find_optimal_threshold(val["isFraud"].values, twin.predict_proba(val[feature_cols])[:, 1])

    booster = lgb.Booster(model_file=str(CHALLENGER_DIR / "lgbm_final_model.txt"))
    test_probs = booster.predict(test_enc[feature_cols].values)
    y_test = test_enc["isFraud"].values
    metrics = compute_metrics(y_test, test_probs, threshold=threshold)

    study = optuna.load_study(
        study_name="lgbm_fraud_v2",
        storage=f"sqlite:///{OPTUNA_DIR}/lgbm_study.sqlite",
    )

    results = {"uncalibrated": metrics, "best_cv_pr_auc": study.best_value}
    (CHALLENGER_DIR / "lgbm_results.json").write_text(json.dumps(results, indent=2))
    log.info(
        "test PR-AUC=%.4f ROC-AUC=%.4f brier=%.4f f1=%.4f thr=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["brier"],
        metrics["f1"],
        metrics["threshold"],
    )


if __name__ == "__main__":
    main()