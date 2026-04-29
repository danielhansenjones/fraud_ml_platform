from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from phase2.src.common.config import DriftConfig
from phase2.src.common.db import make_pool
from phase2.src.drift.runner import run_drift_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    cfg = DriftConfig()

    if not cfg.reference_window_start or not cfg.reference_window_end:
        raise RuntimeError("REFERENCE_WINDOW_START and REFERENCE_WINDOW_END must be set")

    ref_start = datetime.fromisoformat(cfg.reference_window_start).replace(tzinfo=timezone.utc)
    ref_end = datetime.fromisoformat(cfg.reference_window_end).replace(tzinfo=timezone.utc)

    pool = make_pool(cfg.postgres_dsn)
    log.info("drift detector starting")

    while True:
        try:
            alerts = run_drift_check(
                db_pool=pool,
                reference_start=ref_start,
                reference_end=ref_end,
                recent_window_hours=cfg.recent_window_hours,
                min_samples=cfg.drift_min_samples,
                psi_warning_threshold=cfg.psi_warning_threshold,
                psi_critical_threshold=cfg.psi_critical_threshold,
                ks_pvalue_threshold=cfg.ks_pvalue_threshold,
                ks_statistic_threshold=cfg.ks_statistic_threshold,
                js_threshold=cfg.js_threshold,
            )
            if alerts is not None:
                for a in alerts:
                    log.info("drift alert: %s %s=%.4f severity=%s", a.detector_name, a.metric_name, a.metric_value, a.severity)
                log.info("drift check done: %d alert(s)", len(alerts))
        except Exception:
            log.exception("drift check error")

        time.sleep(cfg.drift_run_interval_minutes * 60)


if __name__ == "__main__":
    main()