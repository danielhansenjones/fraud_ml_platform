from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from psycopg_pool import ConnectionPool

from monitoring.drift.js import compute_js_on_hashes
from monitoring.drift.ks import compute_ks
from monitoring.drift.psi import compute_psi

log = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    alert_id: str
    detector_name: str
    metric_name: str
    metric_value: float
    threshold: float
    window_start: datetime
    window_end: datetime
    reference_window_start: datetime
    reference_window_end: datetime
    severity: str
    payload: dict = field(default_factory=dict)


def _deterministic_alert_id(detector_name: str, window_start: datetime, window_end: datetime) -> str:
    """Stable UUID so ON CONFLICT prevents double-insertion on crash-restart within the same window."""
    key = f"{detector_name}|{window_start.isoformat()}|{window_end.isoformat()}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def run_drift_check(
    db_pool: ConnectionPool,
    reference_start: datetime,
    reference_end: datetime,
    recent_window_hours: int,
    min_samples: int,
    psi_warning_threshold: float,
    psi_critical_threshold: float,
    ks_pvalue_threshold: float,
    ks_statistic_threshold: float,
    js_threshold: float,
) -> list[DriftAlert] | None:
    """Returns None if skipped (insufficient samples), [] if no detector fired,
    or a non-empty list when one or more detectors fired.
    """
    now = datetime.now(tz=timezone.utc)
    recent_start = now - timedelta(hours=recent_window_hours)
    recent_end = now

    with db_pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT fraud_probability, features_hash, created_at
            FROM predictions
            WHERE created_at BETWEEN %s AND %s
               OR created_at BETWEEN %s AND %s
            ORDER BY created_at
            """,
            (reference_start, reference_end, recent_start, recent_end),
        ).fetchall()

    ref_probs = np.array([r[0] for r in rows if reference_start <= r[2] <= reference_end])
    rec_probs = np.array([r[0] for r in rows if recent_start <= r[2] <= recent_end])
    ref_hashes = [r[1] for r in rows if reference_start <= r[2] <= reference_end]
    rec_hashes = [r[1] for r in rows if recent_start <= r[2] <= recent_end]

    if len(ref_probs) < min_samples or len(rec_probs) < min_samples:
        log.info(
            "skipped: insufficient samples (reference=%d, recent=%d, min=%d)",
            len(ref_probs), len(rec_probs), min_samples,
        )
        return None

    alerts: list[DriftAlert] = []

    psi_val = compute_psi(ref_probs, rec_probs)
    if psi_val >= psi_warning_threshold:
        severity = "critical" if psi_val >= psi_critical_threshold else "warning"
        alerts.append(DriftAlert(
            alert_id=_deterministic_alert_id("psi", recent_start, recent_end),
            detector_name="psi",
            metric_name="psi",
            metric_value=psi_val,
            threshold=psi_warning_threshold,
            window_start=recent_start,
            window_end=recent_end,
            reference_window_start=reference_start,
            reference_window_end=reference_end,
            severity=severity,
            payload={"n_reference": len(ref_probs), "n_recent": len(rec_probs)},
        ))

    ks = compute_ks(ref_probs, rec_probs, ks_pvalue_threshold, ks_statistic_threshold)
    if ks.flagged:
        alerts.append(DriftAlert(
            alert_id=_deterministic_alert_id("ks", recent_start, recent_end),
            detector_name="ks",
            metric_name="ks_statistic",
            metric_value=ks.statistic,
            threshold=ks_statistic_threshold,
            window_start=recent_start,
            window_end=recent_end,
            reference_window_start=reference_start,
            reference_window_end=reference_end,
            severity="warning",
            payload={"ks_statistic": ks.statistic, "pvalue": ks.pvalue},
        ))

    js_val = compute_js_on_hashes(ref_hashes, rec_hashes)
    if js_val > js_threshold:
        alerts.append(DriftAlert(
            alert_id=_deterministic_alert_id("js", recent_start, recent_end),
            detector_name="js",
            metric_name="js_divergence",
            metric_value=js_val,
            threshold=js_threshold,
            window_start=recent_start,
            window_end=recent_end,
            reference_window_start=reference_start,
            reference_window_end=reference_end,
            severity="warning",
            payload={"n_reference_hashes": len(ref_hashes), "n_recent_hashes": len(rec_hashes)},
        ))

    if alerts:
        _write_alerts(db_pool, alerts)

    return alerts


def _write_alerts(db_pool: ConnectionPool, alerts: list[DriftAlert]) -> None:
    with db_pool.connection() as conn:
        with conn.transaction():
            for a in alerts:
                conn.execute(
                    """
                    INSERT INTO drift_alerts (
                        alert_id, detector_name, metric_name, metric_value, threshold,
                        window_start, window_end, reference_window_start, reference_window_end,
                        severity, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (alert_id) DO NOTHING
                    """,
                    (
                        a.alert_id, a.detector_name, a.metric_name, a.metric_value, a.threshold,
                        a.window_start, a.window_end, a.reference_window_start,
                        a.reference_window_end, a.severity, json.dumps(a.payload),
                    ),
                )
