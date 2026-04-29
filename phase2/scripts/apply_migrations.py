from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"

PHASE2_MIGRATIONS = [
    "002_labels_table.sql",
    "003_models_table.sql",
    "004_drift_alerts_table.sql",
    "005_canary_decisions_table.sql",
    "006_shadow_comparisons_table.sql",
    "007_shadow_summaries_table.sql",
]


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud")
    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Postgres ({dsn}). "
            "Start it first: docker compose up -d postgres"
        ) from exc
    try:
        for filename in PHASE2_MIGRATIONS:
            path = MIGRATIONS_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"migration not found: {path}")
            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
            log.info("applied %s", filename)
    finally:
        conn.close()
    log.info("all phase 2 migrations applied")


if __name__ == "__main__":
    main()
