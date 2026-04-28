"""Populate Redis feature cache from prep_test.parquet."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import redis

ARTIFACTS = Path("training/artifacts")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
FEATURE_TTL = int(os.environ.get("FEATURE_TTL_SECONDS", "86400"))
BATCH_SIZE = 1000
PROGRESS_EVERY = 50_000


def main() -> None:
    test_df = pd.read_parquet(ARTIFACTS / "prep_test.parquet")

    with open(ARTIFACTS / "onnx_feature_order.json") as f:
        feature_order = json.load(f)

    print(f"Loading {len(test_df)} transactions into Redis at {REDIS_HOST}:{REDIS_PORT}")
    print(f"Feature count: {len(feature_order)}")

    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=10,
        health_check_interval=10,
    )
    r.ping()

    # Pre-extract as list of dicts - orders of magnitude faster than iterrows
    txn_ids = test_df["TransactionID"].to_numpy(dtype=np.int64)
    feature_df = test_df[feature_order].copy()
    records = feature_df.to_dict("records")

    total = 0
    pipe = r.pipeline(transaction=False)

    for txn_id, feat_dict in zip(txn_ids, records):
        # Replace NaN with None for JSON serialization
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in feat_dict.items()}
        pipe.set(f"fraud:features:{txn_id}", json.dumps(clean), ex=FEATURE_TTL)
        total += 1

        if total % BATCH_SIZE == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)

        if total % PROGRESS_EVERY == 0:
            print(f"  Loaded {total:,} / {len(test_df):,}")

    if total % BATCH_SIZE != 0:
        pipe.execute()

    print(f"Loaded {total:,} feature vectors into Redis")

    # Sanity check
    print("Sanity check: reading 5 random keys...")
    sample_ids = random.sample(txn_ids.tolist(), 5)
    for txn_id in sample_ids:
        key = f"fraud:features:{txn_id}"
        val = r.get(key)
        assert val is not None, f"Missing key: {key}"
        features = json.loads(val)
        assert len(features) == len(feature_order), (
            f"Feature count mismatch: got {len(features)}, expected {len(feature_order)}"
        )
        assert set(features.keys()) == set(feature_order), "Feature key mismatch"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
