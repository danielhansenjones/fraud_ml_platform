from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np

from monitoring.canary.evaluator import evaluate_canary


def _make_config(**overrides):
    defaults = {
        "postgres_dsn": "postgresql://x",
        "champion_version": "champ-v1",
        "challenger_version": "chall-v1",
        "canary_window_hours": 24,
        "canary_min_labeled_predictions": 10,
        "canary_pr_auc_rollback_delta": 0.05,
        "canary_brier_rollback_delta": 0.02,
        "canary_pr_auc_promote_delta": 0.01,
        "canary_required_consecutive_runs": 3,
        "canary_latency_p95_rollback_ratio": 3.0,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_pool(champ_rows: list[tuple], chall_rows: list[tuple]) -> MagicMock:
    """SQL-blind pool: every conn.execute(...).fetchall() returns the same
    `all_rows`. That works because:
    - The window-fetch query in evaluate_canary is the only fetchall caller
      that uses these rows directly.
    - _count_consecutive_improvements is bypassed in every test below by
      passing `prior_outcomes=[...]`, so its DB query path is never taken.
    Keep that invariant when adding new tests; the alternative is testcontainers
    against a real Postgres, which is tracked separately.
    """
    all_rows = [("champ-v1", *r) for r in champ_rows] + [("chall-v1", *r) for r in chall_rows]

    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = all_rows
    conn.execute.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool


def _fraud_rows(
    n_total: int,
    fraud_rate: float,
    prob_fraud: float = 0.8,
    prob_legit: float = 0.2,
    latency_ms: float = 5.0,
) -> list[tuple]:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_total):
        is_fraud = rng.random() < fraud_rate
        prob = prob_fraud if is_fraud else prob_legit
        prob += rng.normal(0, 0.05)
        prob = float(np.clip(prob, 0.01, 0.99))
        rows.append((prob, prob > 0.5, is_fraud, latency_ms))
    return rows


def test_insufficient_data_returns_continue():
    cfg = _make_config(canary_min_labeled_predictions=100)
    pool = _make_pool(_fraud_rows(5, 0.3), _fraud_rows(5, 0.3))
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.decision == "continue"
    assert decision.reason == "insufficient_data"


def test_catastrophic_pr_auc_divergence_rollback():
    cfg = _make_config()
    # Champion: high discrimination; challenger: near-random
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.9, prob_legit=0.1)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.5, prob_legit=0.45)
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.decision == "rollback"
    assert "pr_auc" in decision.reason


def test_catastrophic_brier_divergence_rollback():
    cfg = _make_config(canary_pr_auc_rollback_delta=99.0)  # disable PR-AUC gate
    # Force Brier score divergence by making challenger wildly overconfident
    champ_rows = _fraud_rows(200, 0.1, prob_fraud=0.7, prob_legit=0.3)
    chall_rows = [(0.99, True, False, 5.0)] * 180 + [(0.99, True, True, 5.0)] * 20
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.decision == "rollback"
    assert "brier" in decision.reason


def test_sustained_improvement_three_runs_promotes():
    cfg = _make_config(canary_required_consecutive_runs=3)
    # Champion near-random, challenger near-perfect: guaranteed large delta
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.52, prob_legit=0.48)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.95, prob_legit=0.05)
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=["improving", "improving"])
    assert decision.decision == "promote"


def test_sustained_improvement_only_two_runs_continues():
    cfg = _make_config(canary_required_consecutive_runs=3)
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.52, prob_legit=0.48)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.95, prob_legit=0.05)
    pool = _make_pool(champ_rows, chall_rows)
    # Only one prior 'continue' - not enough
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=["improving"])
    assert decision.decision == "continue"


def test_no_signal_returns_continue():
    cfg = _make_config()
    # Equal models
    rows = _fraud_rows(200, 0.3, prob_fraud=0.75, prob_legit=0.25)
    pool = _make_pool(rows, rows[:])
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.decision == "continue"
    assert decision.outcome == "no_signal"
    assert decision.reason == "no_clear_signal"
    # Equal inputs must yield equal metrics; pr_auc delta is exactly zero.
    assert decision.metrics["champion"]["pr_auc"] == decision.metrics["challenger"]["pr_auc"]
    assert decision.metrics["champion"]["brier_score"] == decision.metrics["challenger"]["brier_score"]


def test_metrics_payload_has_per_model_section_when_data_sufficient():
    cfg = _make_config(canary_min_labeled_predictions=10)
    rows = _fraud_rows(50, 0.2, latency_ms=7.5)
    pool = _make_pool(rows, rows[:])
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])

    assert "min_required" not in decision.metrics
    expected_keys = {"n", "pr_auc", "roc_auc", "brier_score", "flag_rate", "p95_total_ms"}
    for side in ("champion", "challenger"):
        section = decision.metrics[side]
        assert set(section.keys()) == expected_keys, side
        assert section["n"] == 50
        # Sanity-bound the probabilistic metrics so a regression that emits
        # NaN, inf, or out-of-range values fails loudly here instead of
        # silently corrupting a downstream decision.
        for k in ("pr_auc", "roc_auc", "brier_score", "flag_rate"):
            v = section[k]
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0, f"{side}.{k}={v}"
        # p95 of a constant array must equal the constant.
        assert section["p95_total_ms"] == 7.5


def test_metrics_payload_carries_min_required_when_insufficient():
    cfg = _make_config(canary_min_labeled_predictions=1000)
    rows = _fraud_rows(50, 0.2)
    pool = _make_pool(rows, rows[:])
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.outcome == "insufficient_data"
    assert decision.metrics["min_required"] == 1000
    assert decision.metrics["champion_n"] == 50
    assert decision.metrics["challenger_n"] == 50
    assert "champion" not in decision.metrics


def test_inactive_history_does_not_count_toward_promotion():
    """Regression: 'continue' decisions from `outcome='inactive'` or
    `outcome='insufficient_data'` runs must not feed the consecutive-improvement
    counter. Only `outcome='improving'` should accumulate.
    """
    cfg = _make_config(canary_required_consecutive_runs=3)
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.52, prob_legit=0.48)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.95, prob_legit=0.05)
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(
        pool, "champ-v1", "chall-v1", 24, cfg,
        prior_outcomes=["inactive", "insufficient_data", "no_signal"],
    )
    assert decision.decision == "continue"
    assert decision.outcome == "improving"


def test_improving_outcome_emitted_when_delta_positive_but_runs_insufficient():
    cfg = _make_config(canary_required_consecutive_runs=3)
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.52, prob_legit=0.48)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.95, prob_legit=0.05)
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=[])
    assert decision.decision == "continue"
    assert decision.outcome == "improving"


def test_high_latency_challenger_rolls_back():
    cfg = _make_config(canary_latency_p95_rollback_ratio=2.0)
    champ_rows = _fraud_rows(200, 0.3, prob_fraud=0.52, prob_legit=0.48, latency_ms=5.0)
    chall_rows = _fraud_rows(200, 0.3, prob_fraud=0.95, prob_legit=0.05, latency_ms=50.0)
    pool = _make_pool(champ_rows, chall_rows)
    decision = evaluate_canary(pool, "champ-v1", "chall-v1", 24, cfg, prior_outcomes=["improving", "improving"])
    assert decision.decision == "rollback"
    assert "p95" in decision.reason
