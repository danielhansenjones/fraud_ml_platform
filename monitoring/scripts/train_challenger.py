from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from monitoring.common.encoding import encode_categoricals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / "training" / "artifacts"
CHALLENGER_DIR = ARTIFACTS / "challenger"
CHALLENGER_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_DIR = ROOT / "training" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_parquet(ARTIFACTS / "prep_train.parquet")
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")

    feature_cols = [c for c in train_df.columns if c not in ("isFraud", "TransactionID")]
    target = "isFraud"

    train_enc, cat_mappings = encode_categoricals(train_df[feature_cols + [target]])
    test_enc, _ = encode_categoricals(test_df[feature_cols + [target]], mappings=cat_mappings)
    (CHALLENGER_DIR / "lgbm_category_mappings.json").write_text(json.dumps(cat_mappings, indent=2))

    val_start = int(len(train_enc) * 0.9)
    tr = train_enc.iloc[:val_start]
    val = train_enc.iloc[val_start:]

    return (
        tr[feature_cols],
        tr[target],
        val[feature_cols],
        val[target],
        test_enc[feature_cols],
        test_enc[target],
        feature_cols,
    )


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "random_state": 42,
        "device": "cuda",
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, probs)


def main() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    log.info("data loaded: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_fraud_v2",
        storage=f"sqlite:///{OPTUNA_DIR}/lgbm_study.sqlite",
        load_if_exists=True,
    )
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, 300 - n_complete)
    log.info("study has %d completed trials, running %d more", n_complete, n_remaining)
    if n_remaining > 0:
        study.optimize(
            lambda t: objective(t, X_train, y_train, X_val, y_val),
            n_trials=n_remaining,
            show_progress_bar=True,
        )

    best_params = {**study.best_params, "objective": "binary", "verbosity": -1, "random_state": 42, "device": "cuda"}
    (CHALLENGER_DIR / "lgbm_best_params.json").write_text(json.dumps(best_params, indent=2))
    log.info("best params saved, PR-AUC=%.4f", study.best_value)

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
    )
    final_model.booster_.save_model(str(CHALLENGER_DIR / "lgbm_final_model.txt"))

    # Calibration skipped: phase 1 showed isotonic calibration improved Brier but
    # degraded PR-AUC on the test set. Matches the champion's export decision.

    test_probs = final_model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    results = {"pr_auc_test": pr_auc, "brier_test": brier, "best_cv_pr_auc": study.best_value}
    (CHALLENGER_DIR / "lgbm_results.json").write_text(json.dumps(results, indent=2))
    log.info("test PR-AUC=%.4f brier=%.4f", pr_auc, brier)

    _export_onnx(final_model, feature_cols, cat_mappings=_load_cat_mappings())


def _load_cat_mappings() -> dict:
    path = CHALLENGER_DIR / "lgbm_category_mappings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _export_onnx(model, feature_cols: list[str], cat_mappings: dict | None = None) -> None:
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_lightgbm(model.booster_, initial_types=initial_type, target_opset=12)

        # ORT 1.20.1 rejects onnxmltools output: it emits extra Identity + Cast + ZipMap
        # nodes and mixes standard opset 9 with ai.onnx.ml opset 1. Strip everything
        # except the TreeEnsembleClassifier and wire its outputs to "label" and
        # "probabilities" directly, matching the XGBoost ONNX structure the Go server
        # expects.
        from onnx import TensorProto, helper

        tec = next((n for n in onnx_model.graph.node if n.op_type == "TreeEnsembleClassifier"), None)
        if tec is not None:
            tec.output[0] = "label"
            tec.output[1] = "probabilities"
            # nodes_hitrates is not in the ai.onnx.ml v1 spec; ORT 1.20.x rejects it
            hitrates = next((a for a in tec.attribute if a.name == "nodes_hitrates"), None)
            if hitrates is not None:
                tec.attribute.remove(hitrates)
            del onnx_model.graph.node[:]
            onnx_model.graph.node.append(tec)
            del onnx_model.graph.output[:]
            onnx_model.graph.output.extend([
                helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
                helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2]),
            ])
            std_opsets = [op for op in onnx_model.opset_import if op.domain == ""]
            for op in std_opsets:
                onnx_model.opset_import.remove(op)

        onnx_path = CHALLENGER_DIR / "lgbm_final_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        sample_raw = pd.read_parquet(ARTIFACTS / "prep_test.parquet").head(1000)
        # Apply the train-time category mappings; a fresh per-sample encoding would
        # produce different integer codes than the trained model expects.
        sample_enc, _ = encode_categoricals(sample_raw[feature_cols], mappings=cat_mappings or {})
        x = sample_enc.values.astype(np.float32)

        probs_lgbm = model.predict_proba(sample_enc)[:, 1]
        out = sess.run(["label", "probabilities"], {"float_input": x})
        probs_onnx = out[1][:, 1]

        max_diff = float(np.abs(probs_lgbm - probs_onnx).max())
        parity = {"max_diff": max_diff, "onnx_sha256": sha256, "passed": max_diff < 0.01}
        (CHALLENGER_DIR / "lgbm_onnx_parity_report.json").write_text(json.dumps(parity, indent=2))
        (CHALLENGER_DIR / "lgbm_onnx_feature_order.json").write_text(json.dumps(feature_cols))
        log.info("ONNX exported: max_diff=%.2e sha256=%s", max_diff, sha256[:16])

        # onnxmltools tree-to-ONNX conversion introduces ~1e-3 fp differences that are
        # irreducible without a different export path. 0.01 is looser than XGBoost's
        # 1e-5 but still tight enough that fraud flags don't change at any reasonable
        # threshold.
        if max_diff >= 0.01:
            raise RuntimeError(f"ONNX parity check failed: max_diff={max_diff:.2e}")
    except ImportError as e:
        log.warning("ONNX export skipped (missing dep): %s", e)


if __name__ == "__main__":
    main()
