from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTIFACTS = Path("training/artifacts")


@pytest.mark.skipif(
    not (ARTIFACTS / "final_model.onnx").exists(),
    reason="ONNX artifact not present; run training pipeline first",
)
def test_onnx_parity():
    import onnxruntime as rt
    import xgboost as xgb

    with open(ARTIFACTS / "onnx_feature_order.json") as f:
        feature_order = json.load(f)

    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")
    X = test_df[feature_order].fillna(0).head(1000).astype(np.float32)

    clf = xgb.XGBClassifier()
    clf.load_model(str(ARTIFACTS / "final_model.json"))
    native_probs = clf.predict_proba(X)[:, 1]

    sess = rt.InferenceSession(str(ARTIFACTS / "final_model.onnx"))
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X.values})

    if isinstance(onnx_out[1], list):
        onnx_probs = np.array([d[1] for d in onnx_out[1]], dtype=np.float64)
    else:
        onnx_probs = onnx_out[1][:, 1].astype(np.float64)

    max_diff = float(np.abs(native_probs - onnx_probs).max())
    assert max_diff < 1e-5, f"ONNX parity exceeded tolerance: max_diff={max_diff:.2e}"
