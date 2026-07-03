# Production ML Lifecycle

Layered onto the core XGBoost serving stack.
Adds drift detection, shadow deployment, canary rollout with automatic rollback, late-label evaluation, load testing, and Grafana dashboards.

## What this adds

Drift detection using PSI, KS, and Jensen-Shannon divergence on the prediction distribution and feature hashes.
Computed on rolling windows against a fixed reference period.
Alerts written to Postgres, exposed on a Grafana dashboard.

Shadow mode: 100% of champion-served traffic is duplicated to the challenger with no impact on the response.
Score pairs are written to `shadow_comparisons` and aggregated by the shadow comparator.

Canary deployment with configurable traffic split between champion and challenger.
Automated rollback on metric divergence, automated promotion after sustained improvement over N consecutive evaluation runs.

Late-label evaluation: a simulator draws labels from the IEEE-CIS ground truth with a log-normal delay distribution, inserting them into the `labels` table as if they arrived from a real chargeback system.

Load testing with k6: steady-state, ramp, and failure injection scripts.
SLOs declared and measured.

Four Grafana dashboards covering serving overview, drift, canary comparison, and shadow mode.

## Architecture

Every request hits the router, which forwards to the champion or challenger based on the active traffic split.
Both model services pull features from Redis and write predictions to Postgres asynchronously.

```
client -> router -> champion (Go, ONNX, XGBoost v1)
               \--> challenger (Go, ONNX, LightGBM v1)
                        |
                     Redis               Postgres
                     (features)          (predictions, labels, drift_alerts, ...)
```

Four Python jobs run as containers on a schedule:

- drift_detector (every 15 min): queries recent predictions, writes drift_alerts
- label_joiner (every 5 min): simulates chargeback label arrivals, populates the labels table
- canary_evaluator (every 60 min): scores champion vs challenger on labeled predictions, writes canary_decisions
- shadow_comparator (every 30 min): aggregates shadow_comparisons into shadow_summaries

Prometheus scrapes the router, both model services, and all four jobs.
Grafana reads Prometheus and Postgres.

## What we did not build

No Kafka.
Drift detection is a SQL query over rolling windows of the prediction table.
If volume outgrows a single Postgres instance, the consumer becomes a stream consumer and the schema does not change.

No model registry framework.
The `models` table in Postgres is the registry.
Version, sha256, and role are stored there.
MLflow would add friction without adding signal here.

No Kubernetes.
Docker Compose continues.
The router + champion + challenger pattern is the production pattern; the orchestrator is irrelevant.

No real-time streaming features.
Features stay preloaded in Redis from parquet.
Drift work is on prediction distributions and feature hashes.

No A/B testing on user populations.
This is fraud detection.
The canary splits traffic between two models on the same population, measured on aggregate metrics.

## Drift detection

### Population Stability Index (PSI)

Computed on `fraud_probability` bucketed into 10 quantile-based bins from the reference distribution.
Each bin holds equal reference mass, giving uniform sensitivity across score levels.
Equal-width bins would collapse ~96.5% of scores into `[0, 0.1]` at a 3.5% fraud rate, blinding PSI to mid-range drift.
Falls back to 10 equal-width bins when the reference is degenerated.

Thresholds: <0.1 stable, 0.1-0.25 warning, >0.25 critical.
Industry convention, not statistically derived.
The right thresholds for a real deployment depend on the cost of false alarms vs. missed drift.

PSI catches bulk distribution shift.
A feature drifting in isolation only registers if it moves the predicted probability.
For per-feature shift, use KS or JS.

### Two-sample Kolmogorov-Smirnov (KS)

`scipy.stats.ks_2samp` on `fraud_probability`.
Flagged when p-value < 0.01 AND statistic > 0.05.

Both thresholds are required.
At >10k predictions, even a 0.001 mean shift produces p < 0.01.
The statistic floor ensures detected shifts are large enough to matter.

Catches distribution shape changes that PSI's binning misses.
Misses bulk shifts that PSI catches.
They're complementary.

### Jensen-Shannon divergence on feature hashes

`features_hash` (SHA-256 of the feature vector) bucketed by first byte (256 buckets).
JS divergence between reference and recent bucket distributions.

Honest limitation: SHA-256 spreads distinct inputs uniformly across buckets, so this metric cannot see feature drift.
A drifted stream of distinct vectors buckets the same as a healthy one.
What moves it: repeated identical vectors (a stuck pipeline serving one cached vector, a flood of defaults) or missing and malformed hashes.
Treat it as a pipeline-failure alarm, not a drift detector.
PSI and KS carry the drift signal.

## Label simulator caveats

The `labels` table is populated by a simulator.
On each run it finds predictions without a label, draws a delay from a log-normal distribution (mu=ln(86400) ~= 1 day, sigma=0.7, capped at 14 days) seeded per transaction ID so repeated runs draw the same delay, and inserts the label with `available_at = created_at + delay` once that time has passed.
The label value comes from the IEEE-CIS `isFraud` column for that TransactionID.

This mimics real label latency in structure only.
Real chargeback labels arrive in bursts tied to processing cycles, correlate with the fraud category, have multi-week delays for manual review, and can flip after investigation.
The delayed-evaluation infrastructure is real; the label arrival process is synthetic.

## Canary methodology

Rollback fires if challenger PR-AUC is worse than champion by >0.05 OR challenger Brier score is worse by >0.02.
Either condition alone is enough.

Promotion fires if challenger PR-AUC is better than champion by >0.01 sustained for 3 consecutive evaluator runs.

Minimum data per window: 500 labeled predictions per model.
Below that the decision is `continue/insufficient_data`.

### Promotion walkthrough

```
Champion: XGBoost v1, PR-AUC=0.505 (test set, phase 1)
Challenger: LightGBM v1, PR-AUC=0.52 (test set, phase 2)

Run 1 (24h window, 600 champion / 180 challenger labeled predictions):
  champion PR-AUC=0.498, challenger=0.511, delta=+0.013 > promote_delta(0.01)
  consecutive_runs=0 < required(3) -> continue/no_clear_signal

Run 2 (next 24h):
  challenger=0.518, delta=+0.017
  consecutive_runs=1 < 3 -> continue

Run 3:
  delta=+0.014, consecutive_runs=2 -> promote
  old champion -> retired, challenger -> champion
  router: canary off, full traffic to promoted champion
```

### Rollback walkthrough

```
LightGBM challenger at 30% canary.

Run 1 (after a feature pipeline incident):
  champion PR-AUC=0.499, challenger=0.441, delta=-0.058 < -rollback_delta(-0.05)
  -> rollback
  challenger -> retired, router back to champion only
```

### Operator notes

The monitoring jobs (drift, label, canary, shadow) each run as a single container with a `time.sleep(interval)` loop. There is no leader election. Scaling any of them to multiple replicas would produce duplicate writes (canary decisions, drift alerts) and double the work. Boring beats clever at this scale; if you need to scale, add a leader lease or move the schedule into an external orchestrator.

### Scope decisions

Promotion is gated on prediction-quality metrics (PR-AUC, Brier) plus a latency guardrail. The challenger trips a rollback if its p95 exceeds the champion's by more than `CANARY_LATENCY_P95_ROLLBACK_RATIO` (default 3x), wide enough to tolerate a heavier model under partial canary load without flapping. Upstream error rate is collected in Prometheus for dashboards rather than wired into the decision so the gate stays driven by labeled outcomes.

Retraining is operator-triggered. The drift detector writes to `drift_alerts`; when sustained drift warrants it, rerun `monitoring/main.py --only train_challenger` and restart with the new challenger artifact. Auto-retraining off unlabeled drift is deliberately left out: promotion is gated on labeled outcomes, and feeding drift signals into model lifecycle changes the failure mode in ways that need their own evaluation.

## Load test results

Single developer machine, not a production environment. These numbers confirm the system handles the stated load on this hardware; they do not establish production capacity.

Hardware: AMD Ryzen 9 9950X3D 16-core, 30GiB RAM, no Docker resource limits.

### Steady-state (100 RPS, 10 minutes)

60,001 requests, 0 errors.

| Metric      | Value  | SLO    |
|-------------|--------|--------|
| p95 latency | 3.83ms | <50ms  |
| p99 latency | 7.08ms | <200ms |
| error rate  | 0.00%  | <0.5%  |
| result      | pass   |        |

p95 is 13x inside the SLO.
The Go serving stack (ONNX inference + Redis feature lookup + async Postgres write) is not the bottleneck at this load.

### Breakdown ramp

100 to 5000 RPS planned over 4 stages with a 5% error-rate auto-abort. 183,741 requests over 5m18s before abort.

| Metric              | Value     | SLO             |
|---------------------|-----------|-----------------|
| p95 at abort        | 64ms      | <200ms          |
| p99 at abort        | 125ms     | -               |
| error rate at abort | 5.30%     | abort threshold |
| peak RPS reached    | ~1100     |                 |
| breakdown point     | ~1000 RPS |                 |
| failure mode        | hard 503s |                 |

Champion ONNX inference is the bottleneck. The champion container is already consuming ~16 host cores at 311 RPS; router stays below 10% CPU and Redis below 2% throughout. Latency stays inside the p95 SLO through the cliff; failures are explicit 503s from the upstream client, not context-deadline timeouts. See [load/README.md](load/README.md) for the per-component diagnosis.

### Failure injection (champion paused 30s)

Two 30s pause windows during a 10-minute 100 RPS run.
60,000 requests total.

| Metric                   | Value                        |
|--------------------------|------------------------------|
| error rate during pause  | ~100% (503s)                 |
| overall error rate       | 9.95%                        |
| threshold                | pass (rate < 0.1)            |
| recovery after unpause   | immediate (<1 request cycle) |
| p95 on good requests     | 4.73ms                       |
| crash/data loss          | none                         |

Router returned 503 during the pause, not silent failure.
Latency returned to baseline on the first request after unpause.

## SLOs

- Latency: p95 < 50ms warm, p99 < 200ms warm, single-node compose
- Throughput: 100 RPS without queue growth
- Error rate: under 0.5% steady state

## LightGBM ONNX export

onnxmltools converts LightGBM boosters to ONNX via a custom converter.
At num_leaves=500, parity between native predict and ORT 1.20.1:

```
max_diff: 3.37e-04
passed:   true (threshold 0.01)
```

3.4e-4 is a 0.03 percentage point probability disagreement.
At any real fraud threshold (0.3-0.7) this never changes a flag decision.
XGBoost export agrees to within 6e-7 (0.00006 percentage points); the LightGBM path keeps a larger float gap (1e-4 to 1e-3 across exports) from how onnxmltools represents the tree ensemble in ONNX opset.

Three fixes were needed to produce a graph ORT 1.20.1 accepts:

1. Strip ZipMap.
   onnxmltools emits a ZipMap node that converts the probability array to a list of dicts.
   ORT rejects this for output binding.
   Fix: remove ZipMap, rename the raw float tensor to "probabilities".
2. Remove nodes_hitrates.
   ORT 1.20.1 rejects this attribute, which is not part of the ai.onnx.ml v1 spec.
   Fix: drop it from TreeEnsembleClassifier before serializing.
3. Remove standard opset import.
   onnxmltools emits both standard opset 9 and ai.onnx.ml opset 1.
   ORT rejects the mixed import for this graph.
   Fix: drop the standard opset entry, keep only ai.onnx.ml v1.

## How to run

Phase 1 must be complete: training done, XGBoost ONNX exported, Redis loaded.

### Monitoring setup

Start Postgres and Redis first:

```bash
docker compose up -d postgres redis
uv run python monitoring/main.py
```

Runs three steps: apply migrations 002-007, train the LightGBM challenger (200 Optuna trials, ~60 min), register both models in Postgres, and write `training/artifacts/model_versions.json`.

### Bring up the serving stack

```bash
docker compose --profile monitoring up --build -d
```

With observability (Prometheus + Grafana + node-exporter):

```bash
docker compose --profile monitoring --profile observability up --build -d
```

Phase 1 only (champion + router, no challenger):

```bash
docker compose up -d
```

### Load features into Redis

```bash
uv run python training/scripts/load_features_to_redis.py
```

### Configure .env

Before starting the stack, fill in `.env` (copy from `.env.example`):

```bash
REFERENCE_WINDOW_START=<first hour with >=200 predictions>
REFERENCE_WINDOW_END=<end of that window>

# from training/artifacts/model_versions.json after running monitoring/main.py
CHAMPION_VERSION=<sha>
CHALLENGER_VERSION=<sha>
```

The four monitoring jobs (drift-detector, label-joiner, canary-evaluator, shadow-comparator) start as containers when `docker compose --profile monitoring up` runs.

### Smoke test

```bash
ROUTER_URL=http://localhost:8081 bash ops/phase2_smoke_test.sh
```

### Load tests

k6 must be installed (`go install go.k6.io/k6@latest`)
Generate transaction IDs once:

```bash
uv run python -c "import json, pandas as pd; df = pd.read_parquet('training/artifacts/prep_test.parquet', columns=['TransactionID']); json.dump(df.TransactionID.astype(int).tolist(), open('load/k6/transaction_ids.json','w'))"
```

```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_steady.js
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_ramp.js
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_with_failures.js
```

### Tests

```bash
uv run pytest tests/python/ -v
cd router && go test ./...
```
