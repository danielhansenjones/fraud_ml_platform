from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import requests

from monitoring.canary.evaluator import CanaryDecision, evaluate_canary
from monitoring.common.config import CanaryConfig
from monitoring.common.db import make_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_router_state(cfg) -> dict:
    try:
        resp = requests.get(f"{cfg.router_url}/admin/state", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.error("could not fetch router state: %s; assuming canary inactive", exc)
        return {"canary_enabled": "false"}


def _set_router_canary(cfg, enabled: bool, challenger_pct: int, shadow_pct: int) -> None:
    try:
        resp = requests.post(
            f"{cfg.router_url}/admin/canary",
            json={"enabled": enabled, "challenger_traffic_percent": challenger_pct, "shadow_percent": shadow_pct},
            headers={"X-Admin-Token": cfg.admin_token},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error("failed to update router canary state: %s", exc)


def _write_decision(pool, decision: CanaryDecision, cfg) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO canary_decisions (
                decision_id, champion_version, challenger_version,
                decision, reason, metrics, window_start, window_end
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                decision.decision_id,
                cfg.champion_version,
                cfg.challenger_version,
                decision.decision,
                decision.reason,
                json.dumps(decision.metrics),
                decision.window_start,
                decision.window_end,
            ),
        )


def apply_decision(decision: CanaryDecision, pool, cfg) -> None:
    if decision.decision == "promote":
        with pool.connection() as conn:
            conn.execute("UPDATE models SET role='retired', retired_at=NOW() WHERE role='champion'")
            conn.execute(
                "UPDATE models SET role='champion', promoted_at=NOW() WHERE model_version=%s",
                (cfg.challenger_version,),
            )
        _set_router_canary(cfg, enabled=False, challenger_pct=0, shadow_pct=100)
        log.info("promoted challenger %s to champion", cfg.challenger_version)
    elif decision.decision == "rollback":
        with pool.connection() as conn:
            conn.execute(
                "UPDATE models SET role='retired', retired_at=NOW() WHERE model_version=%s",
                (cfg.challenger_version,),
            )
        _set_router_canary(cfg, enabled=False, challenger_pct=0, shadow_pct=100)
        log.info("rolled back challenger %s", cfg.challenger_version)


def main() -> None:
    cfg = CanaryConfig()

    if not cfg.champion_version or not cfg.challenger_version:
        raise RuntimeError("CHAMPION_VERSION and CHALLENGER_VERSION must be set")

    pool = make_pool(cfg.postgres_dsn)
    log.info("canary evaluator starting")

    while True:
        try:
            router_state = _get_router_state(cfg)
            canary_active = router_state.get("canary_enabled", "false").lower() == "true"

            if not canary_active:
                now = datetime.now(tz=timezone.utc)
                decision = CanaryDecision(
                    decision="continue",
                    reason="canary_inactive",
                    metrics={},
                    window_start=now,
                    window_end=now,
                )
            else:
                decision = evaluate_canary(
                    db_pool=pool,
                    champion_version=cfg.champion_version,
                    challenger_version=cfg.challenger_version,
                    window_hours=cfg.canary_window_hours,
                    config=cfg,
                )

            _write_decision(pool, decision, cfg)
            apply_decision(decision, pool, cfg)
            log.info("canary decision: %s (%s)", decision.decision, decision.reason)
        except Exception:
            log.exception("canary evaluator error")

        time.sleep(cfg.canary_interval_minutes * 60)


if __name__ == "__main__":
    main()
