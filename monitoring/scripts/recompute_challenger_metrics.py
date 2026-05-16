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
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")
    feature_cols = [c for c in test_df.columns if c not in ("isFraud", "TransactionID")]
    test_enc, _ = encode_categoricals(test_df[feature_cols + ["isFraud"]], mappings=cat_mappings)

    booster = lgb.Booster(model_file=str(CHALLENGER_DIR / "lgbm_final_model.txt"))
    test_probs = booster.predict(test_enc[feature_cols].values)
    y_test = test_enc["isFraud"].values

    threshold = find_optimal_threshold(y_test, test_probs)
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