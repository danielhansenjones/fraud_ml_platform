from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_and_merge(data_dir: Path) -> pd.DataFrame:
    txn = pd.read_csv(data_dir / "train_transaction.csv")
    idt = pd.read_csv(data_dir / "train_identity.csv")
    return txn.merge(idt, on="TransactionID", how="left")


def load_test(data_dir: Path) -> pd.DataFrame:
    txn = pd.read_csv(data_dir / "test_transaction.csv")
    idt = pd.read_csv(data_dir / "test_identity.csv")
    return txn.merge(idt, on="TransactionID", how="left")
