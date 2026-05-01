from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.src.data import load_and_merge
from training.src.features import (
    apply_categorical_encoder,
    build_categorical_encoder,
    drop_high_missing,
    engineer_amount_features,
    engineer_email_features,
    engineer_time_features,
)

DATA_DIR = Path("data/ieee_cis")
ARTIFACTS = Path("training/artifacts")

AUDIT_COLS = {"TransactionID", "TransactionDT", "isFraud"}


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(DATA_DIR)
    print(f"Raw shape: {df.shape}")

    df, dropped = drop_high_missing(df, threshold=0.75)
    with open(ARTIFACTS / "dropped_columns.json", "w") as f:
        json.dump(dropped, f, indent=2)
    print(f"Dropped {len(dropped)} high-missing columns")

    df = engineer_time_features(df)
    df = engineer_amount_features(df)
    df = engineer_email_features(df)

    obj_cols = [c for c in df.columns if df[c].dtype.kind == "O"]
    for col in obj_cols:
        cats = sorted(df[col].dropna().unique().tolist())
        df[col] = pd.Categorical(df[col], categories=cats, ordered=False)

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    encoder = build_categorical_encoder(df, obj_cols)
    df = apply_categorical_encoder(df, encoder)

    with open(ARTIFACTS / "categorical_encoder.json", "w") as f:
        json.dump(encoder, f, indent=2)

    n = len(df)
    train_end = int(n * 0.8)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    assert train_df["TransactionDT"].max() < test_df["TransactionDT"].min(), (
        "Temporal split leakage detected"
    )

    train_df.to_parquet(ARTIFACTS / "prep_train.parquet", index=False)
    test_df.to_parquet(ARTIFACTS / "prep_test.parquet", index=False)

    feature_cols = [c for c in df.columns if c not in AUDIT_COLS]
    feature_list = [{"name": c, "dtype": str(df[c].dtype)} for c in feature_cols]
    with open(ARTIFACTS / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_list)}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
