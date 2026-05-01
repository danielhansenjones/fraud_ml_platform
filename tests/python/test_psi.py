from __future__ import annotations

import numpy as np

from monitoring.drift.psi import compute_psi


def test_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1, 1000)
    psi = compute_psi(data, data.copy())
    assert psi < 0.01, f"identical distributions should yield PSI near 0, got {psi:.4f}"


def test_completely_disjoint_distributions():
    reference = np.zeros(500)
    recent = np.ones(500)
    psi = compute_psi(reference, recent)
    assert psi > 1.0, f"fully disjoint distributions should yield large PSI, got {psi:.4f}"


def test_warning_threshold():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0, 0.5, 1000)
    recent = rng.uniform(0.3, 0.8, 1000)
    psi = compute_psi(reference, recent)
    assert psi > 0.1, f"expected PSI > 0.1 for moderate shift, got {psi:.4f}"


def test_empty_bucket_no_division_error():
    reference = np.full(500, 0.05)
    recent = np.concatenate([np.full(400, 0.05), np.full(100, 0.95)])
    psi = compute_psi(reference, recent)
    assert np.isfinite(psi)


def test_single_sample_each():
    psi = compute_psi(np.array([0.5]), np.array([0.5]))
    assert np.isfinite(psi)


def test_stable_distribution_below_threshold():
    rng = np.random.default_rng(2)
    reference = rng.beta(2, 5, 2000)
    recent = rng.beta(2.1, 5.1, 2000)
    psi = compute_psi(reference, recent)
    assert psi < 0.1, f"near-identical beta distributions should be stable, got {psi:.4f}"
