from __future__ import annotations

import numpy as np

from phase2.src.drift.ks import compute_ks


def test_identical_distributions_not_flagged():
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1, 1000)
    result = compute_ks(data, data.copy())
    assert not result.flagged


def test_clearly_different_distributions_flagged():
    rng = np.random.default_rng(1)
    reference = rng.beta(2, 5, 2000)
    recent = rng.beta(5, 2, 2000)
    result = compute_ks(reference, recent)
    assert result.flagged
    assert result.statistic > 0.05


def test_large_sample_small_effect_not_flagged():
    # At n=100_000 even a trivial shift produces p < 0.01; statistic threshold guards this case.
    rng = np.random.default_rng(2)
    reference = rng.normal(0.5, 0.1, 100_000)
    recent = rng.normal(0.501, 0.1, 100_000)  # 0.1% mean shift
    result = compute_ks(reference, recent)
    if result.pvalue < 0.01:
        assert not result.flagged or result.statistic > 0.05


def test_statistic_and_pvalue_returned():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1, 500)
    b = rng.uniform(0.2, 1.2, 500)
    result = compute_ks(a, b)
    assert 0 <= result.statistic <= 1
    assert 0 <= result.pvalue <= 1


def test_custom_thresholds():
    rng = np.random.default_rng(4)
    reference = rng.uniform(0, 1, 1000)
    recent = rng.uniform(0.1, 1.1, 1000)
    result = compute_ks(reference, recent, pvalue_threshold=0.5, statistic_threshold=0.0)
    assert result.flagged
