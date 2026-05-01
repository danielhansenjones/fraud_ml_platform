from __future__ import annotations

import os
import types


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def DriftConfig() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        postgres_dsn=_get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"),
        reference_window_start=_get("REFERENCE_WINDOW_START", ""),
        reference_window_end=_get("REFERENCE_WINDOW_END", ""),
        recent_window_hours=int(_get("RECENT_WINDOW_HOURS", "1")),
        drift_run_interval_minutes=int(_get("DRIFT_RUN_INTERVAL_MINUTES", "15")),
        drift_min_samples=int(_get("DRIFT_MIN_SAMPLES", "200")),
        psi_warning_threshold=float(_get("PSI_WARNING_THRESHOLD", "0.1")),
        psi_critical_threshold=float(_get("PSI_CRITICAL_THRESHOLD", "0.25")),
        ks_pvalue_threshold=float(_get("KS_PVALUE_THRESHOLD", "0.01")),
        ks_statistic_threshold=float(_get("KS_STATISTIC_THRESHOLD", "0.05")),
        js_threshold=float(_get("JS_THRESHOLD", "0.05")),
    )


def LabelConfig() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        postgres_dsn=_get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"),
        ground_truth_path=_get("GROUND_TRUTH_PATH", ""),
        label_joiner_interval_minutes=int(_get("LABEL_JOINER_INTERVAL_MINUTES", "5")),
        label_delay_mu=float(_get("LABEL_DELAY_MU", "86400.0")),
        label_delay_sigma=float(_get("LABEL_DELAY_SIGMA", "0.7")),
        label_delay_max_days=int(_get("LABEL_DELAY_MAX_DAYS", "14")),
        max_inserts_per_run=int(_get("MAX_INSERTS_PER_RUN", "5000")),
    )


def CanaryConfig() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        postgres_dsn=_get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"),
        router_url=_get("ROUTER_URL", "http://localhost:8081"),
        admin_token=_get("ADMIN_TOKEN", ""),
        champion_version=_get("CHAMPION_VERSION", ""),
        challenger_version=_get("CHALLENGER_VERSION", ""),
        canary_interval_minutes=int(_get("CANARY_INTERVAL_MINUTES", "60")),
        canary_window_hours=int(_get("CANARY_WINDOW_HOURS", "24")),
        canary_min_labeled_predictions=int(_get("CANARY_MIN_LABELED_PREDICTIONS", "500")),
        canary_pr_auc_rollback_delta=float(_get("CANARY_PR_AUC_ROLLBACK_DELTA", "0.05")),
        canary_brier_rollback_delta=float(_get("CANARY_BRIER_ROLLBACK_DELTA", "0.02")),
        canary_pr_auc_promote_delta=float(_get("CANARY_PR_AUC_PROMOTE_DELTA", "0.01")),
        canary_required_consecutive_runs=int(_get("CANARY_REQUIRED_CONSECUTIVE_RUNS", "3")),
        canary_latency_p95_rollback_ratio=float(_get("CANARY_LATENCY_P95_ROLLBACK_RATIO", "3.0")),
    )


def ShadowConfig() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        postgres_dsn=_get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"),
        shadow_interval_minutes=int(_get("SHADOW_INTERVAL_MINUTES", "30")),
        shadow_window_hours=int(_get("SHADOW_WINDOW_HOURS", "1")),
    )