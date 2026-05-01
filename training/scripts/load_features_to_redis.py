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

    # Store all features, not just the champion's subset. Each model service reads only
    # the columns it needs via its own feature order file. Category columns are encoded
    # to integer codes so the Go client can deserialize them as float32 uniformly across
    # champion and challenger.
    all_feature_cols = [c for c in test_df.columns if c != "TransactionID"]
    feature_df = test_df[all_feature_cols].copy()
    for col in feature_df.columns:
        if hasattr(feature_df[col], "cat"):
            feature_df[col] = feature_df[col].cat.codes.astype("float64")

    print(f"Loading {len(test_df)} transactions into Redis at {REDIS_HOST}:{REDIS_PORT}")
    print(f"Feature count: {len(all_feature_cols)}")

    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=10,
        health_check_interval=10,
    )
    r.ping()

    # to_dict("records") is orders of magnitude faster than iterrows for this volume.
    txn_ids = test_df["TransactionID"].to_numpy(dtype=np.int64)
    records = feature_df.to_dict("records")

    total = 0
    pipe = r.pipeline(transaction=False)

    for txn_id, feat_dict in zip(txn_ids, records):
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

    print("Sanity check: reading 5 random keys...")
    sample_ids = random.sample(txn_ids.tolist(), 5)
    for txn_id in sample_ids:
        key = f"fraud:features:{txn_id}"
        val = r.get(key)
        assert val is not None, f"Missing key: {key}"
        features = json.loads(val)
        assert len(features) == len(all_feature_cols), (
            f"Feature count mismatch: got {len(features)}, expected {len(all_feature_cols)}"
        )
        assert set(features.keys()) == set(all_feature_cols), "Feature key mismatch"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
