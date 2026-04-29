from __future__ import annotations

import numpy as np

from phase2.src.drift.js import compute_js_on_hashes


def _make_hashes_from_bucket(bucket: int, n: int) -> list[str]:
    prefix = f"{bucket:02x}"
    return [f"{prefix}{'00' * 31}" for _ in range(n)]


def test_identical_distributions_near_zero():
    hashes = [f"{i % 256:02x}{'00' * 31}" for i in range(1000)]
    js = compute_js_on_hashes(hashes, hashes[:])
    assert js < 0.01, f"identical hash distributions should yield JS near 0, got {js:.4f}"


def test_disjoint_distributions_nonzero():
    ref = _make_hashes_from_bucket(0, 500)
    rec = _make_hashes_from_bucket(255, 500)
    js = compute_js_on_hashes(ref, rec)
    assert js > 0.5, f"fully disjoint should yield large JS, got {js:.4f}"


def test_all_hashes_same_bucket_no_divide_by_zero():
    ref = _make_hashes_from_bucket(42, 500)
    rec = _make_hashes_from_bucket(42, 500)
    js = compute_js_on_hashes(ref, rec)
    assert np.isfinite(js)
    assert js < 0.01


def test_empty_reference_list():
    rec = _make_hashes_from_bucket(10, 100)
    js = compute_js_on_hashes([], rec)
    assert np.isfinite(js)


def test_js_symmetric():
    rng = np.random.default_rng(0)
    buckets_a = rng.integers(0, 256, 500)
    buckets_b = rng.integers(0, 256, 500)
    hashes_a = [f"{b:02x}{'00' * 31}" for b in buckets_a]
    hashes_b = [f"{b:02x}{'00' * 31}" for b in buckets_b]
    js_ab = compute_js_on_hashes(hashes_a, hashes_b)
    js_ba = compute_js_on_hashes(hashes_b, hashes_a)
    assert abs(js_ab - js_ba) < 0.01, f"JS should be symmetric: {js_ab:.4f} vs {js_ba:.4f}"


def test_returns_float_in_range():
    hashes = [f"{i % 256:02x}{'00' * 31}" for i in range(200)]
    js = compute_js_on_hashes(hashes[:100], hashes[100:])
    assert 0.0 <= js <= 1.0
