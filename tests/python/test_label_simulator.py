from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock


from monitoring.labels.simulator import simulate_label_arrivals


def _make_pool(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn.execute.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=None)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool


def test_deterministic_zero_delay_inserts_all():
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    rows = [
        (1001, one_hour_ago),
        (1002, one_hour_ago),
        (1003, one_hour_ago),
    ]
    pool = _make_pool(rows)

    ground_truth = {1001: True, 1002: False, 1003: True}
    def zero_delay(): return timedelta(seconds=0)

    n = simulate_label_arrivals(pool, ground_truth, zero_delay, max_inserts_per_run=100)

    assert n == 3, f"expected 3 inserts, got {n}"


def test_future_delay_inserts_nothing():
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)

    rows = [(1001, now)]
    pool = _make_pool(rows)

    ground_truth = {1001: True}
    def far_future_delay(): return timedelta(days=365)

    n = simulate_label_arrivals(pool, ground_truth, far_future_delay, max_inserts_per_run=100)

    assert n == 0, f"expected 0 inserts (delay not elapsed), got {n}"


def test_idempotent_with_no_candidates():
    pool = _make_pool([])
    ground_truth = {1001: True}
    n = simulate_label_arrivals(pool, ground_truth, lambda: timedelta(0), max_inserts_per_run=100)
    assert n == 0


def test_respects_max_inserts_per_run():
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    past = now - timedelta(hours=2)

    rows = [(i, past) for i in range(1000, 1200)]
    pool = _make_pool(rows)

    ground_truth = {i: i % 2 == 0 for i in range(1000, 1200)}
    n = simulate_label_arrivals(pool, ground_truth, lambda: timedelta(0), max_inserts_per_run=50)

    assert n <= 50


def test_only_known_transaction_ids_inserted():
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    past = now - timedelta(hours=1)

    rows = [(1001, past), (9999, past)]
    pool = _make_pool(rows)

    ground_truth = {1001: True}  # 9999 is not in ground truth
    n = simulate_label_arrivals(pool, ground_truth, lambda: timedelta(0))

    assert n == 1, f"expected 1 insert (only known tx), got {n}"
