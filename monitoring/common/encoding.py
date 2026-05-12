from __future__ import annotations

import pandas as pd


def encode_categoricals(df: pd.DataFrame, mappings: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """`mappings=None` builds per-column category-to-int maps from the input
    (train mode) and returns them. Passing the returned `mappings` back in
    (val/test) reuses those maps so unseen categories encode to -1, matching
    the categorical_encoder.json semantics from the champion preprocessing -
    a fresh per-call encoding would produce different integer codes than the
    trained model expects.
    """
    df = df.copy()
    out_mappings: dict = {} if mappings is None else mappings
    for col in df.columns:
        if not hasattr(df[col], "cat"):
            continue
        if mappings is None:
            mapping = {str(v): i for i, v in enumerate(df[col].cat.categories)}
            out_mappings[col] = mapping
        else:
            mapping = mappings.get(col, {})
        df[col] = df[col].astype("string").map(mapping).fillna(-1).astype("float32")
    return df, out_mappings
