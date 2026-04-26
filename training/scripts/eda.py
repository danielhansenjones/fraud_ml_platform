"""EDA on IEEE-CIS train data. Produces summary JSON and plots. No downstream script depends on this."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.src.data import load_and_merge

DATA_DIR = Path("data/ieee_cis")
ARTIFACTS = Path("training/artifacts")
PLOTS = ARTIFACTS / "plots"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(DATA_DIR)
    print(f"Merged shape: {df.shape}")

    # Class distribution
    fraud_counts = df["isFraud"].value_counts().to_dict()
    fraud_frac = float(df["isFraud"].mean())
    print(f"Fraud fraction: {fraud_frac:.4f}")

    # Temporal coverage
    dt_min = int(df["TransactionDT"].min())
    dt_max = int(df["TransactionDT"].max())
    dt_span_days = (dt_max - dt_min) / 86400

    # Missing values
    missing = df.isnull().mean().sort_values(ascending=False)
    top30_missing = missing.head(30)

    # Categorical cardinality
    obj_cols = [c for c in df.columns if df[c].dtype.kind == "O"]
    cardinality = {c: int(df[c].nunique()) for c in obj_cols}

    # Email domain
    p_email_vc = df["P_emaildomain"].value_counts().head(20).to_dict()

    # ProductCD
    product_vc = df["ProductCD"].value_counts().to_dict()

    summary = {
        "shape": list(df.shape),
        "fraud_fraction": fraud_frac,
        "fraud_counts": {str(k): int(v) for k, v in fraud_counts.items()},
        "transaction_dt_min": dt_min,
        "transaction_dt_max": dt_max,
        "transaction_dt_span_days": dt_span_days,
        "n_missing_cols_over_50pct": int((missing > 0.5).sum()),
        "categorical_cardinality": cardinality,
        "p_emaildomain_top20": {str(k): int(v) for k, v in p_email_vc.items()},
        "productcd_dist": {str(k): int(v) for k, v in product_vc.items()},
    }

    with open(ARTIFACTS / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: missing top 30
    fig, ax = plt.subplots(figsize=(12, 8))
    top30_missing.plot.barh(ax=ax)
    ax.set_xlabel("Missing fraction")
    ax.set_title("Top 30 columns by missing fraction")
    plt.tight_layout()
    fig.savefig(PLOTS / "missing_top30.png", dpi=150)
    plt.close(fig)

    # Plot: class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Not Fraud", "Fraud"], [fraud_counts.get(0, 0), fraud_counts.get(1, 0)])
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    plt.tight_layout()
    fig.savefig(PLOTS / "class_dist.png", dpi=150)
    plt.close(fig)

    # Plot: TransactionAmt log histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(df["TransactionAmt"].dropna()), bins=100, edgecolor="none")
    ax.set_xlabel("log1p(TransactionAmt)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction amount distribution (log scale)")
    plt.tight_layout()
    fig.savefig(PLOTS / "amount_log.png", dpi=150)
    plt.close(fig)

    # Plot: temporal transaction volume (daily)
    daily = df.groupby(df["TransactionDT"] // 86400).size()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values)
    ax.set_xlabel("Day (from reference)")
    ax.set_ylabel("Transactions")
    ax.set_title("Daily transaction volume")
    plt.tight_layout()
    fig.savefig(PLOTS / "temporal_volume.png", dpi=150)
    plt.close(fig)

    print("EDA complete. Artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
