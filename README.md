# Fraud ML Platform

Most ML portfolio projects stop at model training.
This one builds the production system around the model: ONNX inference in Go, a canary router with automated rollback, drift detection on rolling prediction windows, and a late-label evaluation loop that simulates the delay between a transaction and its fraud label.

The XGBoost champion scores PR-AUC 0.505 on the held-out test set.
A LightGBM challenger at 0.584 runs behind the router on 100% shadow traffic while the canary evaluator decides whether to promote it.

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

| Model                       | PR-AUC    | ROC-AUC   | Eval          |
|-----------------------------|-----------|-----------|---------------|
| Logistic Regression         | 0.272     | 0.807     | CV mean       |
| Random Forest               | 0.405     | 0.836     | CV mean       |
| LightGBM baseline           | 0.527     | 0.878     | CV mean       |
| XGBoost untuned             | 0.540     | 0.892     | CV mean       |
| XGBoost tuned (champion)    | **0.505** | **0.899** | held-out test |
| LightGBM tuned (challenger) | **0.584** | -         | held-out test |

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
cp .env.example .env  # edit CHAMPION_VERSION / CHALLENGER_VERSION after phase 2 setup

# Phase 1: train and serve
uv run python main.py
docker compose up -d redis postgres
uv run python training/scripts/load_features_to_redis.py
docker compose up --build

# Phase 2: challenger, router, canary, monitoring (LightGBM training ~60 min)
docker compose up -d postgres redis
uv run python phase2/main.py
# edit .env: set REFERENCE_WINDOW_START/END and CHAMPION/CHALLENGER_VERSION
docker compose --profile phase2-canary up --build -d
```

See [PRODUCTION.md](PRODUCTION.md) for the full architecture, drift methodology, canary walkthrough, ONNX negative result, and load test results.
