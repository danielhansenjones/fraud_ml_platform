from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"

MIGRATIONS = [
    "001_predictions_table.sql",
    "002_labels_table.sql",
    "003_models_table.sql",
    "004_drift_alerts_table.sql",
    "005_canary_decisions_table.sql",
    "006_shadow_comparisons_table.sql",
    "007_shadow_summaries_table.sql",
    "008_canary_decisions_outcome.sql",
]

SCHEMA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud")
    try:
        conn = psycopg.connect(dsn, autocommit=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Postgres ({dsn}). "
            "Start it first: docker compose up -d postgres"
        ) from exc
    try:
        conn.execute(SCHEMA_TABLE_SQL)

        for filename in MIGRATIONS:
            already = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE filename = %s", (filename,)
            ).fetchone()
            if already:
                log.info("skip %s (already recorded)", filename)
                continue
            path = MIGRATIONS_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"migration not found: {path}")
            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)", (filename,)
                )
            log.info("applied %s", filename)
    finally:
        conn.close()
    log.info("all monitoring migrations applied")


if __name__ == "__main__":
    main()
