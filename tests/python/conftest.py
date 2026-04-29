from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 86400 * 180, n)),
        "TransactionAmt": rng.exponential(100, n),
        "isFraud": rng.integers(0, 2, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", np.nan], n),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
        "card1": rng.integers(1000, 9999, n).astype(float),
        "card2": rng.integers(100, 999, n).astype(float),
        "V1": rng.uniform(0, 1, n),
        "V2": rng.uniform(0, 1, n),
        "high_miss_col": [np.nan] * n,  # 100% missing - should be dropped
    })
