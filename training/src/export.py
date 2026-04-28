from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxmltools
import onnxruntime as rt
import pandas as pd
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType


def _remove_zipmap(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ZipMap post-processor so probabilities output is a plain float32 tensor.

    onnxmltools converts classifiers with a ZipMap node that turns the probability
    tensor into a sequence of maps. onnxruntime_go cannot consume sequence outputs,
    so we strip ZipMap and rewire directly to the float tensor.
    """
    graph = model.graph
    zipmap_nodes = [n for n in graph.node if n.op_type == "ZipMap"]
    if not zipmap_nodes:
        return model

    zipmap = zipmap_nodes[0]
    # zipmap input[0] is the raw float32 prob tensor; zipmap output[0] is the map
    raw_prob_name = zipmap.input[0]
    map_prob_name = zipmap.output[0]

    # Rewire all consumers of the map output to use the raw tensor directly
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == map_prob_name:
                node.input[i] = raw_prob_name

    # Fix the graph output declaration
    for out in graph.output:
        if out.name == map_prob_name:
            out.name = raw_prob_name
            # Update type to float tensor [None, n_classes]
            out.type.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    raw_prob_name, onnx.TensorProto.FLOAT, [None, 2]
                ).type
            )

    graph.node.remove(zipmap)
    return model


def export_onnx(
    clf: xgb.XGBClassifier,
    feature_cols: list[str],
    output_path: Path,
    opset: int = 15,
) -> None:
    n_features = len(feature_cols)
    # onnxmltools requires feature names matching 'f%d' - rename booster internals
    booster = clf.get_booster()
    original_names = booster.feature_names
    booster.feature_names = [f"f{i}" for i in range(n_features)]
    try:
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_type, target_opset=opset)
        onnx_model = _remove_zipmap(onnx_model)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    finally:
        booster.feature_names = original_names


def parity_check(
    clf: xgb.XGBClassifier,
    onnx_path: Path,
    X: pd.DataFrame,
    n_samples: int = 1000,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    X_sample = X.head(n_samples).copy().fillna(0).astype(np.float32)

    native_probs = clf.predict_proba(X_sample)[:, 1]

    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_sample.values})
    # After ZipMap removal, output[1] is a float32 tensor [N, 2]
    onnx_probs = np.asarray(onnx_out[1])[:, 1]

    diffs = np.abs(native_probs - onnx_probs.astype(np.float64))
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())

    if max_diff >= tolerance:
        raise ValueError(f"ONNX parity check failed: max_diff={max_diff:.2e} >= {tolerance:.2e}")

    return {"max_diff": max_diff, "mean_diff": mean_diff, "n_samples": n_samples}
