from __future__ import annotations


import numpy as np
import pandas as pd


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.75) -> tuple[pd.DataFrame, list[str]]:
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    return df.drop(columns=drop_cols), drop_cols


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["TransactionDT"]
    df = df.copy()
    df["hour"] = (dt // 3600) % 24
    df["day_of_week"] = (dt // 86400) % 7
    df["day"] = dt // 86400
    return df


def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = df["TransactionAmt"] - np.floor(df["TransactionAmt"])
    return df


def engineer_email_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _tld(series: pd.Series) -> pd.Series:
        return series.str.split(".").str[-1]

    df["p_email_tld"] = _tld(df["P_emaildomain"])
    if "R_emaildomain" in df.columns:
        df["r_email_tld"] = _tld(df["R_emaildomain"])
    # Skip r_email_tld when R_emaildomain was dropped; an all-NaN column carries no signal.
    p = df["P_emaildomain"].fillna("__missing__")
    r = df["R_emaildomain"].fillna("__missing__") if "R_emaildomain" in df.columns else pd.Series(
        ["__missing__"] * len(df), index=df.index
    )
    df["email_match"] = (p == r).astype(int)
    p_null = df["P_emaildomain"].isnull()
    r_null = df["R_emaildomain"].isnull() if "R_emaildomain" in df.columns else pd.Series(False, index=df.index)
    df.loc[p_null | r_null, "email_match"] = 0
    return df


def build_categorical_encoder(
    df: pd.DataFrame, cols: list[str]
) -> dict[str, dict[str, int]]:
    encoder: dict[str, dict[str, int]] = {}
    for col in cols:
        cats = sorted(df[col].dropna().astype(str).unique())
        encoder[col] = {cat: idx for idx, cat in enumerate(cats)}
    return encoder


def apply_categorical_encoder(
    df: pd.DataFrame, encoder: dict[str, dict[str, int]]
) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in encoder.items():
        if col not in df.columns:
            continue
        df[col + "_encoded"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df
