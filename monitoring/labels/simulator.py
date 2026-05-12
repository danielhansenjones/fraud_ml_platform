from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import numpy as np
from psycopg_pool import ConnectionPool


def default_delay_distribution(mu: float = np.log(86400), sigma: float = 0.7, max_days: int = 14) -> Callable[[], timedelta]:
    max_seconds = max_days * 86400

    def sample() -> timedelta:
        seconds = min(np.random.lognormal(mu, sigma), max_seconds)
        return timedelta(seconds=float(seconds))

    return sample


def simulate_label_arrivals(
    db_pool: ConnectionPool,
    ground_truth: dict[int, bool],
    delay_distribution: Callable[[], timedelta],
    max_inserts_per_run: int = 5000,
) -> int:
    """Delay distribution is configurable but not calibrated to real chargeback timing.
    Real label latency is bursty, correlated with fraud category, and subject to manual
    review queues that this function does not model.
    """
    now = datetime.now(tz=timezone.utc)

    # Pull unlabeled predictions and filter to known IDs in Python instead of
    # shipping the full ground_truth keyset (potentially 500k+ ints) as a SQL
    # parameter on every interval.
    with db_pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT p.transaction_id, p.created_at
            FROM predictions p
            LEFT JOIN labels l ON l.transaction_id = p.transaction_id
            WHERE l.transaction_id IS NULL
            LIMIT %s
            """,
            (max_inserts_per_run * 5,),
        ).fetchall()

    if not rows:
        return 0

    to_insert = []
    for transaction_id, created_at in rows:
        if transaction_id not in ground_truth:
            continue
        delay = delay_distribution()
        if created_at + delay <= now:
            to_insert.append((transaction_id, ground_truth[transaction_id]))
        if len(to_insert) >= max_inserts_per_run:
            break

    if not to_insert:
        return 0

    with db_pool.connection() as conn:
        with conn.transaction():
            for transaction_id, is_fraud in to_insert:
                conn.execute(
                    """
                    INSERT INTO labels (transaction_id, is_fraud, label_source, available_at)
                    VALUES (%s, %s, 'simulator', NOW())
                    ON CONFLICT (transaction_id) DO NOTHING
                    """,
                    (transaction_id, is_fraud),
                )

    return len(to_insert)
