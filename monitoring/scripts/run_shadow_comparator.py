from __future__ import annotations

import logging
import time

from monitoring.common.config import ShadowConfig
from monitoring.common.db import make_pool
from monitoring.shadow.comparator import compute_shadow_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    cfg = ShadowConfig()
    pool = make_pool(cfg.postgres_dsn)
    log.info("shadow comparator starting")

    while True:
        try:
            summary = compute_shadow_summary(pool, window_hours=cfg.shadow_window_hours)
            if summary:
                log.info(
                    "shadow summary: n=%d disagreement_rate=%.3f correlation=%.3f",
                    summary["n_comparisons"],
                    summary["disagreement_rate"],
                    summary["correlation"] if summary["correlation"] is not None else float("nan"),
                )
            else:
                log.info("shadow comparator: no comparisons in window")
        except Exception:
            log.exception("shadow comparator error")

        time.sleep(cfg.shadow_interval_minutes * 60)


if __name__ == "__main__":
    main()
