from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class KSResult:
    statistic: float
    pvalue: float
    flagged: bool


def compute_ks(
    reference: np.ndarray,
    recent: np.ndarray,
    pvalue_threshold: float = 0.01,
    statistic_threshold: float = 0.05,
) -> KSResult:
    """Both thresholds must be exceeded to flag. At large N, any tiny shift produces
    p < 0.01 - the statistic threshold prevents noise from triggering alerts.
    """
    result = stats.ks_2samp(reference, recent)
    flagged = result.pvalue < pvalue_threshold and result.statistic > statistic_threshold
    return KSResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        flagged=flagged,
    )
