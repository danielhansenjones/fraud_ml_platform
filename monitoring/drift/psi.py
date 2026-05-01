from __future__ import annotations

import numpy as np


def compute_psi(reference: np.ndarray, recent: np.ndarray, n_bins: int = 10) -> float:
    """Standard thresholds: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant.
    These are industry convention, not statistically derived.
    """
    # Quantile bins give equal reference mass per bin. Equal-width bins would collapse
    # ~96% of fraud scores (3.5% fraud rate) into [0, 0.1], blinding PSI to mid-range drift.
    edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)

    eps = 1e-6

    ref_counts, _ = np.histogram(reference, bins=edges)
    rec_counts, _ = np.histogram(recent, bins=edges)

    ref_pct = ref_counts / len(reference)
    rec_pct = rec_counts / len(recent)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    rec_pct = np.where(rec_pct == 0, eps, rec_pct)

    return float(np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct)))