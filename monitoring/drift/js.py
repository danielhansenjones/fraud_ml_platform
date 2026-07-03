from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_js_on_hashes(reference_hashes: list[str], recent_hashes: list[str]) -> float:
    """Buckets by the first byte of each hash (256 buckets). SHA-256 buckets distinct
    inputs uniformly, so this cannot see feature drift; it spikes on repeated identical
    vectors or missing hashes. Pipeline-failure alarm, not a drift detector.
    """
    n_buckets = 256
    eps = 1e-6

    def bucket_counts(hashes: list[str]) -> np.ndarray:
        counts = np.zeros(n_buckets, dtype=float)
        for h in hashes:
            if h and len(h) >= 2:
                bucket = int(h[:2], 16)
                counts[bucket] += 1
        total = counts.sum()
        if total == 0:
            return np.ones(n_buckets) / n_buckets
        return counts / total

    ref_dist = bucket_counts(reference_hashes)
    rec_dist = bucket_counts(recent_hashes)

    ref_dist = np.where(ref_dist == 0, eps, ref_dist)
    rec_dist = np.where(rec_dist == 0, eps, rec_dist)

    ref_dist /= ref_dist.sum()
    rec_dist /= rec_dist.sum()

    return float(jensenshannon(ref_dist, rec_dist))
