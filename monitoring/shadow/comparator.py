from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
from psycopg_pool import ConnectionPool


def compute_shadow_summary(
    db_pool: ConnectionPool,
    window_hours: int = 1,
) -> dict | None:
    now = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(hours=window_hours)

    with db_pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT champion_probability, challenger_probability,
                   champion_flagged, challenger_flagged
            FROM shadow_comparisons
            WHERE created_at BETWEEN %s AND %s
            """,
            (window_start, now),
        ).fetchall()

    if not rows:
        return None

    champ_probs = np.array([r[0] for r in rows])
    chall_probs = np.array([r[1] for r in rows])
    champ_flags = np.array([r[2] for r in rows], dtype=bool)
    chall_flags = np.array([r[3] for r in rows], dtype=bool)

    abs_diffs = np.abs(champ_probs - chall_probs)
    disagreement_rate = float(np.mean(champ_flags != chall_flags))
    correlation = float(np.corrcoef(champ_probs, chall_probs)[0, 1]) if len(rows) > 1 else None

    tp = int(np.sum(champ_flags & chall_flags))
    fp = int(np.sum(~champ_flags & chall_flags))
    fn = int(np.sum(champ_flags & ~chall_flags))
    tn = int(np.sum(~champ_flags & ~chall_flags))

    summary = {
        "summary_id": str(uuid.uuid4()),
        "window_start": window_start,
        "window_end": now,
        "n_comparisons": len(rows),
        "correlation": correlation,
        "disagreement_rate": disagreement_rate,
        "abs_diff_p50": float(np.percentile(abs_diffs, 50)),
        "abs_diff_p95": float(np.percentile(abs_diffs, 95)),
        "abs_diff_p99": float(np.percentile(abs_diffs, 99)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    with db_pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO shadow_summaries (
                summary_id, window_start, window_end, n_comparisons,
                correlation, disagreement_rate,
                abs_diff_p50, abs_diff_p95, abs_diff_p99,
                confusion
            ) VALUES (%(summary_id)s, %(window_start)s, %(window_end)s, %(n_comparisons)s,
                      %(correlation)s, %(disagreement_rate)s,
                      %(abs_diff_p50)s, %(abs_diff_p95)s, %(abs_diff_p99)s,
                      %(confusion)s)
            """,
            {**summary, "confusion": json.dumps(summary["confusion"])},
        )

    return summary
