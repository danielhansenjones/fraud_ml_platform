from __future__ import annotations

import logging
import time

import numpy as np
import pyarrow.parquet as pq

from monitoring.common.config import LabelConfig
from monitoring.common.db import make_pool
from monitoring.labels.simulator import default_delay_distribution, simulate_label_arrivals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_ground_truth(path: str) -> dict[int, bool]:
    table = pq.read_table(path, columns=["TransactionID", "isFraud"])
    df = table.to_pandas()
    return dict(zip(df["TransactionID"].astype(int), df["isFraud"].astype(bool)))


def main() -> None:
    cfg = LabelConfig()

    if not cfg.ground_truth_path:
        raise RuntimeError("GROUND_TRUTH_PATH must be set")

    ground_truth = load_ground_truth(cfg.ground_truth_path)
    log.info("loaded %d ground truth labels", len(ground_truth))

    pool = make_pool(cfg.postgres_dsn)
    delay_dist = default_delay_distribution(
        mu=np.log(cfg.label_delay_mu),
        sigma=cfg.label_delay_sigma,
        max_days=cfg.label_delay_max_days,
    )
    log.info("label joiner starting")

    while True:
        try:
            n = simulate_label_arrivals(pool, ground_truth, delay_dist, cfg.max_inserts_per_run)
            log.info("label joiner: inserted %d labels", n)
        except Exception:
            log.exception("label joiner error")

        time.sleep(cfg.label_joiner_interval_minutes * 60)


if __name__ == "__main__":
    main()