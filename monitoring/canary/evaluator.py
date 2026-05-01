from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from psycopg_pool import ConnectionPool
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class ModelMetrics:
    n_predictions: int
    pr_auc: float
    roc_auc: float
    brier_score: float
    flag_rate: float
    p95_latency_ms: float


@dataclass
class CanaryDecision:
    decision: Literal["promote", "rollback", "continue", "extend"]
    reason: str
    metrics: dict
    window_start: datetime
    window_end: datetime
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_flag: np.ndarray, y_latency_ms: np.ndarray
) -> ModelMetrics:
    return ModelMetrics(
        n_predictions=len(y_true),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        flag_rate=float(y_flag.mean()),
        p95_latency_ms=float(np.percentile(y_latency_ms, 95)),
    )


def evaluate_canary(
    db_pool: ConnectionPool,
    champion_version: str,
    challenger_version: str,
    window_hours: int,
    config,
    prior_decisions: list[str] | None = None,
) -> CanaryDecision:
    """prior_decisions: recent decision values for the consecutive-run promotion check.
    If None, queries the DB.
    """
    now = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(hours=window_hours)
    window_end = now

    with db_pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT p.model_version, p.fraud_probability, p.flagged, l.is_fraud, p.total_ms
            FROM predictions p
            JOIN labels l ON l.transaction_id = p.transaction_id
            WHERE p.created_at BETWEEN %s AND %s
              AND l.available_at <= %s
              AND p.model_version IN (%s, %s)
            """,
            (window_start, window_end, now, champion_version, challenger_version),
        ).fetchall()

    champ_rows = [(r[1], r[2], r[3], r[4]) for r in rows if r[0] == champion_version]
    chall_rows = [(r[1], r[2], r[3], r[4]) for r in rows if r[0] == challenger_version]

    def to_arrays(rs):
        probs = np.array([r[0] for r in rs])
        flags = np.array([r[1] for r in rs])
        labels = np.array([r[2] for r in rs], dtype=float)
        latencies_ms = np.array([r[3] for r in rs])
        return probs, flags, labels, latencies_ms

    metrics_payload: dict = {}

    if len(champ_rows) < config.canary_min_labeled_predictions or len(chall_rows) < config.canary_min_labeled_predictions:
        metrics_payload = {
            "champion_n": len(champ_rows),
            "challenger_n": len(chall_rows),
            "min_required": config.canary_min_labeled_predictions,
        }
        return CanaryDecision(
            decision="continue",
            reason="insufficient_data",
            metrics=metrics_payload,
            window_start=window_start,
            window_end=window_end,
        )

    champ_probs, champ_flags, champ_labels, champ_latencies = to_arrays(champ_rows)
    chall_probs, chall_flags, chall_labels, chall_latencies = to_arrays(chall_rows)

    champ_m = _compute_metrics(champ_labels, champ_probs, champ_flags, champ_latencies)
    chall_m = _compute_metrics(chall_labels, chall_probs, chall_flags, chall_latencies)

    metrics_payload = {
        "champion": {
            "n": champ_m.n_predictions,
            "pr_auc": champ_m.pr_auc,
            "roc_auc": champ_m.roc_auc,
            "brier_score": champ_m.brier_score,
            "flag_rate": champ_m.flag_rate,
            "p95_latency_ms": champ_m.p95_latency_ms,
        },
        "challenger": {
            "n": chall_m.n_predictions,
            "pr_auc": chall_m.pr_auc,
            "roc_auc": chall_m.roc_auc,
            "brier_score": chall_m.brier_score,
            "flag_rate": chall_m.flag_rate,
            "p95_latency_ms": chall_m.p95_latency_ms,
        },
    }

    pr_delta = chall_m.pr_auc - champ_m.pr_auc
    brier_delta = chall_m.brier_score - champ_m.brier_score

    if pr_delta < -config.canary_pr_auc_rollback_delta:
        return CanaryDecision(
            decision="rollback",
            reason=f"challenger pr_auc worse by {abs(pr_delta):.4f} (threshold {config.canary_pr_auc_rollback_delta})",
            metrics=metrics_payload,
            window_start=window_start,
            window_end=window_end,
        )

    if brier_delta > config.canary_brier_rollback_delta:
        return CanaryDecision(
            decision="rollback",
            reason=f"challenger brier_score worse by {brier_delta:.4f} (threshold {config.canary_brier_rollback_delta})",
            metrics=metrics_payload,
            window_start=window_start,
            window_end=window_end,
        )

    if champ_m.p95_latency_ms > 0:
        latency_ratio = chall_m.p95_latency_ms / champ_m.p95_latency_ms
        if latency_ratio > config.canary_latency_p95_rollback_ratio:
            return CanaryDecision(
                decision="rollback",
                reason=(
                    f"challenger p95 latency {chall_m.p95_latency_ms:.1f}ms is "
                    f"{latency_ratio:.1f}x champion ({champ_m.p95_latency_ms:.1f}ms)"
                ),
                metrics=metrics_payload,
                window_start=window_start,
                window_end=window_end,
            )

    if pr_delta > config.canary_pr_auc_promote_delta:
        consecutive = _count_consecutive_improvements(
            db_pool, champion_version, challenger_version, pr_delta, prior_decisions
        )
        if consecutive >= config.canary_required_consecutive_runs - 1:
            return CanaryDecision(
                decision="promote",
                reason=f"challenger pr_auc better by {pr_delta:.4f} for {consecutive + 1} consecutive runs",
                metrics=metrics_payload,
                window_start=window_start,
                window_end=window_end,
            )

    return CanaryDecision(
        decision="continue",
        reason="no_clear_signal",
        metrics=metrics_payload,
        window_start=window_start,
        window_end=window_end,
    )


def _count_consecutive_improvements(
    db_pool: ConnectionPool,
    champion_version: str,
    challenger_version: str,
    current_delta: float,
    prior_decisions: list[str] | None,
) -> int:
    if prior_decisions is not None:
        count = 0
        for d in reversed(prior_decisions):
            if d == "continue":
                count += 1
            else:
                break
        return count

    with db_pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT decision FROM canary_decisions
            WHERE champion_version = %s AND challenger_version = %s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (champion_version, challenger_version),
        ).fetchall()

    count = 0
    for (d,) in rows:
        if d == "continue":
            count += 1
        else:
            break
    return count
