from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from training.src.export import export_onnx, parity_check

ARTIFACTS = Path("training/artifacts")


def main() -> None:
    with open(ARTIFACTS / "results.json") as f:
        results = json.load(f)
    feature_cols = results["feature_cols"]

    clf = xgb.XGBClassifier()
    clf.load_model(str(ARTIFACTS / "final_model.json"))

    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")
    X_test = test_df[feature_cols].fillna(-1).astype("float32")

    onnx_path = ARTIFACTS / "final_model.onnx"
    export_onnx(clf, feature_cols, onnx_path)
    print(f"ONNX model written to {onnx_path}")

    onnx_size_mb = onnx_path.stat().st_size / 1e6
    print(f"ONNX file size: {onnx_size_mb:.1f} MB")
    assert onnx_size_mb < 50, f"ONNX model too large: {onnx_size_mb:.1f} MB"

    with open(ARTIFACTS / "onnx_feature_order.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    parity = parity_check(clf, onnx_path, X_test)
    print(f"Parity check: max_diff={parity['max_diff']:.2e}, mean_diff={parity['mean_diff']:.2e}")

    with open(ARTIFACTS / "onnx_parity_report.json", "w") as f:
        json.dump(parity, f, indent=2)

    print("ONNX export complete.")


if __name__ == "__main__":
    main()
