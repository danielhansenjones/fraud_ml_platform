from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from monitoring.shadow.comparator import compute_shadow_summary


def _make_pool(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn.execute.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool


def _make_rows(
    n: int,
    correlation: float = 0.9,
    disagreement_rate: float = 0.05,
) -> list[tuple]:
    rng = np.random.default_rng(42)
    champ = rng.uniform(0.1, 0.9, n)
    noise = rng.normal(0, 0.05 * (1 - correlation), n)
    chall = np.clip(champ + noise, 0.01, 0.99)

    champ_flag = champ > 0.5
    chall_flag = champ_flag.copy()
    flip_idx = rng.choice(n, size=int(n * disagreement_rate), replace=False)
    chall_flag[flip_idx] = ~chall_flag[flip_idx]

    return [(float(c), float(d), bool(cf), bool(df)) for c, d, cf, df in zip(champ, chall, champ_flag, chall_flag)]


def test_returns_none_on_empty_window():
    pool = _make_pool([])
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is None


def test_correlation_near_one_for_identical_scores():
    rows = [(0.8, 0.8, True, True)] * 100 + [(0.2, 0.2, False, False)] * 100
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["correlation"] == pytest.approx(1.0, abs=0.01)


def test_zero_disagreement_when_flags_match():
    rows = [(0.8, 0.7, True, True)] * 50 + [(0.2, 0.3, False, False)] * 50
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["disagreement_rate"] == pytest.approx(0.0, abs=0.01)


def test_full_disagreement_when_flags_all_differ():
    rows = [(0.8, 0.8, True, False)] * 100
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["disagreement_rate"] == pytest.approx(1.0, abs=0.01)


def test_abs_diff_percentiles_ordered():
    rows = _make_rows(500, correlation=0.8, disagreement_rate=0.1)
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["abs_diff_p50"] <= result["abs_diff_p95"] <= result["abs_diff_p99"]


def test_confusion_matrix_sums_to_n():
    rows = [(0.8, 0.8, True, True)] * 40 + \
           [(0.8, 0.2, True, False)] * 10 + \
           [(0.2, 0.8, False, True)] * 5 + \
           [(0.2, 0.2, False, False)] * 45
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    c = result["confusion"]
    assert c["tp"] == 40
    assert c["fn"] == 10
    assert c["fp"] == 5
    assert c["tn"] == 45
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 100


def test_n_comparisons_matches_row_count():
    rows = _make_rows(200)
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["n_comparisons"] == 200


def test_summary_id_is_deterministic_per_window():
    """Re-running the comparator on the same window must produce the same
    summary_id so the INSERT ... ON CONFLICT no-ops instead of duplicating.
    Mirrors the drift detector's idempotency contract.
    """
    rows = _make_rows(50)
    pool = _make_pool(rows)
    result1 = compute_shadow_summary(pool, window_hours=1)
    result2 = compute_shadow_summary(pool, window_hours=1)
    assert result1 is not None
    assert result2 is not None
    # Two back-to-back calls fall into different microsecond windows, so a
    # direct id1 == id2 check cannot prove determinism. Recompute the recipe
    # directly against the first window's timestamps instead.
    import hashlib
    import uuid as _uuid
    key = f"{result1['window_start'].isoformat()}|{result1['window_end'].isoformat()}"
    expected = str(_uuid.UUID(hashlib.md5(key.encode()).hexdigest()))
    assert result1["summary_id"] == expected


def test_single_row_no_crash():
    rows = [(0.6, 0.4, True, False)]
    pool = _make_pool(rows)
    result = compute_shadow_summary(pool, window_hours=1)
    assert result is not None
    assert result["n_comparisons"] == 1
    assert result["correlation"] is None  # corrcoef requires n > 1
