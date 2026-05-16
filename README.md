# Fraud ML Platform

[![ci](https://github.com/danielhansenjones/fraud_ml_platform/actions/workflows/ci.yml/badge.svg)](https://github.com/danielhansenjones/fraud_ml_platform/actions/workflows/ci.yml)

Most ML portfolio projects stop at model training.
This one builds the production system around the model: ONNX inference in Go, a canary router with automated rollback, drift detection on rolling prediction windows, and a late-label evaluation loop that simulates the delay between a transaction and its fraud label.

The XGBoost champion scores PR-AUC 0.523 on the held-out test set. A LightGBM challenger scores 0.573 on the same split and runs behind the router on 100% shadow traffic while the canary evaluator decides whether to promote it.

## The Interesting Parts

**Temporal correctness in training.** Standard k-fold CV leaks future data into training folds on a time-ordered dataset.
I used purged time-series splits and ran adversarial validation to identify the features driving distribution shift between train and test, then pruned them before training.

**ONNX parity across a language boundary.** The Go serving layer loads an ONNX model exported from Python and runs inference via ONNX Runtime.
XGBoost exports are bit-identical.
LightGBM via onnxmltools required stripping ZipMap output nodes, removing a non-standard `nodes_hitrates` attribute, and patching the opset import before ORT 1.20.1 would accept the graph.
The resulting max diff is 1.6e-3 - acceptable for any fraud flag threshold.

**Safe model rollout.** The challenger receives 100% shadow traffic from the moment it is deployed, with zero impact on responses.
The canary evaluator scores both models on labeled predictions, applies PR-AUC and Brier guardrails, and promotes after 3 consecutive improving evaluation windows.
A single bad window triggers immediate rollback.

**Calibration negative result.** Isotonic calibration fit on the most recent validation window did not improve Brier on the held-out test set and slightly degraded PR-AUC.
The step function collapses ranges of scores into ties, which loses ranking information when the test distribution sits on different breakpoints than the validation window the calibrator was fit on.
The uncalibrated model is served.

## Results

### Baselines (5-fold purged time-series CV, train set only)

| Model                       | PR-AUC | ROC-AUC | Notes                          |
|-----------------------------|--------|---------|--------------------------------|
| Logistic Regression         | 0.271  | 0.807   | 100k stratified subsample      |
| Random Forest               | 0.404  | 0.836   | 100k stratified subsample      |
| LightGBM baseline           | 0.532  | 0.880   | full train, native NaN         |
| XGBoost untuned             | 0.539  | 0.890   | full train, native NaN         |

LR and RF use a 100k stratified subsample to keep the baseline sweep within a fixed time budget; LightGBM and XGBoost run on the full train set.

### Champion vs challenger (held-out test set)

The same evaluation script, same NaN policy, same categorical encoding applied to test using train-time mappings.

| Model                       | PR-AUC    | ROC-AUC   |
|-----------------------------|-----------|-----------|
| XGBoost tuned (champion)    | 0.523     | **0.906** |
| LightGBM tuned (challenger) | **0.573** | -         |

The challenger beats the champion on test PR-AUC by 0.05. This is the canary system's actual job: shadow it on live traffic and let the evaluator decide whether the advantage holds across multiple windows before promoting. ROC-AUC is not recorded in the challenger's results artifact; only PR-AUC and Brier (0.0223) are saved.

| Load Test                      | p95     | Error Rate | Result          |
|--------------------------------|---------|------------|-----------------|
| Steady-state 100 RPS, 10 min   | 3.83ms  | 0.00%      | pass            |
| Ramp to 500 RPS, 22 min        | 32.45ms | 0.00%      | no breakdown    |
| Champion paused 30s at 100 RPS | -       | 9.95%      | pass (SLO <10%) |

## Stack

Python, Go, Redis, Postgres, ONNX Runtime, Prometheus, Grafana, k6, Docker Compose

## Caveats

- The majority of the features are anonymized Vesta fields.
  SHAP identifies which ones matter; understanding why requires their private feature dictionary.
- The label simulator captures the delay structure of real fraud labels (log-normal, ~1 day median) but not the real arrival process: no chargeback processing bursts, no label flipping, no manual review queues.
- Load test numbers are from a single developer machine.
  They confirm the system handles the stated load, not production capacity.

## How to Run

```bash
uv sync
cp .env.example .env  # edit CHAMPION_VERSION / CHALLENGER_VERSION after monitoring setup

# Phase 1: train and serve
docker compose up -d redis postgres
uv run python main.py            # runs training pipeline through load_features_to_redis
docker compose up --build

# Phase 2 monitoring: challenger model, router, drift/label/canary/shadow jobs
# (the challenger Optuna search dominates wall time on first run; reusing the
# saved study on subsequent runs is fast.)
uv run python monitoring/main.py
# edit .env: set REFERENCE_WINDOW_START/END and CHAMPION/CHALLENGER_VERSION
docker compose --profile monitoring up --build -d
```

See [PRODUCTION.md](PRODUCTION.md) for the full architecture, drift methodology, canary walkthrough, ONNX negative result, and load test results.
