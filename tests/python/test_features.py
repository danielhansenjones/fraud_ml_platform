from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.src.features import (
    apply_categorical_encoder,
    build_categorical_encoder,
    drop_high_missing,
    engineer_amount_features,
    engineer_email_features,
    engineer_time_features,
)


def test_drop_high_missing_removes_column(sample_df):
    result, dropped = drop_high_missing(sample_df, threshold=0.75)
    assert "high_miss_col" in dropped
    assert "high_miss_col" not in result.columns


def test_drop_high_missing_keeps_low_missing(sample_df):
    result, dropped = drop_high_missing(sample_df, threshold=0.75)
    assert "TransactionAmt" not in dropped
    assert "TransactionAmt" in result.columns


def test_drop_high_missing_threshold_boundary():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4]})
    # 'a' is 75% missing - exactly at threshold, not above
    result, dropped = drop_high_missing(df, threshold=0.75)
    assert "a" not in dropped
    df2 = pd.DataFrame({"a": [np.nan, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4]})
    result2, dropped2 = drop_high_missing(df2, threshold=0.75)
    assert "a" in dropped2


def test_engineer_time_features_values():
    df = pd.DataFrame({"TransactionDT": [3600, 86400, 86400 * 2 + 7200]})
    result = engineer_time_features(df)
    assert result["hour"].iloc[0] == 1       # 3600s = 1 hour
    assert result["day_of_week"].iloc[1] == 1  # day 1
    assert result["day"].iloc[2] == 2         # day 2


def test_engineer_time_features_no_mutation(sample_df):
    original_cols = set(sample_df.columns)
    result = engineer_time_features(sample_df)
    assert set(sample_df.columns) == original_cols
    assert {"hour", "day_of_week", "day"}.issubset(result.columns)


def test_engineer_amount_features_log1p():
    df = pd.DataFrame({"TransactionAmt": [0.0, 1.0, 99.0]})
    result = engineer_amount_features(df)
    np.testing.assert_allclose(result["TransactionAmt_log"], np.log1p([0.0, 1.0, 99.0]))


def test_engineer_amount_features_decimal():
    df = pd.DataFrame({"TransactionAmt": [100.75, 50.0, 9.99]})
    result = engineer_amount_features(df)
    np.testing.assert_allclose(result["TransactionAmt_decimal"], [0.75, 0.0, 0.99], atol=1e-6)


def test_engineer_email_features_match():
    df = pd.DataFrame({
        "P_emaildomain": ["gmail.com", "yahoo.com", np.nan, "gmail.com"],
        "R_emaildomain": ["gmail.com", "gmail.com", "gmail.com", np.nan],
    })
    result = engineer_email_features(df)
    assert result["email_match"].iloc[0] == 1   # same
    assert result["email_match"].iloc[1] == 0   # different
    assert result["email_match"].iloc[2] == 0   # NaN P
    assert result["email_match"].iloc[3] == 0   # NaN R


def test_engineer_email_features_tld():
    df = pd.DataFrame({
        "P_emaildomain": ["gmail.com", "yahoo.co.uk", np.nan],
        "R_emaildomain": ["gmail.com", "hotmail.com", "gmail.com"],
    })
    result = engineer_email_features(df)
    assert result["p_email_tld"].iloc[0] == "com"
    assert result["p_email_tld"].iloc[1] == "uk"
    assert pd.isna(result["p_email_tld"].iloc[2])


def test_build_categorical_encoder_deterministic():
    df = pd.DataFrame({"col": ["b", "a", "c", "a", None]})
    enc1 = build_categorical_encoder(df, ["col"])
    enc2 = build_categorical_encoder(df, ["col"])
    assert enc1 == enc2
    # sorted order: a=0, b=1, c=2
    assert enc1["col"]["a"] == 0
    assert enc1["col"]["b"] == 1
    assert enc1["col"]["c"] == 2


def test_apply_categorical_encoder_unknown_maps_to_minus1():
    df = pd.DataFrame({"col": ["a", "b", "unknown_value"]})
    encoder = {"col": {"a": 0, "b": 1}}
    result = apply_categorical_encoder(df, encoder)
    assert result["col_encoded"].iloc[0] == 0
    assert result["col_encoded"].iloc[1] == 1
    assert result["col_encoded"].iloc[2] == -1


def test_apply_categorical_encoder_missing_maps_to_minus1():
    df = pd.DataFrame({"col": ["a", np.nan, "b"]})
    encoder = {"col": {"a": 0, "b": 1}}
    result = apply_categorical_encoder(df, encoder)
    assert result["col_encoded"].iloc[1] == -1
