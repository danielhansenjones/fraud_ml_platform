# Load Testing

k6 load tests for the router service.

## Prerequisites

- k6 installed: https://k6.io/docs/getting-started/installation/
- Router running and reachable
- `load/k6/transaction_ids.json` generated (see below)
- Redis populated with features for those IDs (see pre-flight check below)

## Generating transaction_ids.json

The IDs must come from the same parquet that `training/scripts/load_features_to_redis.py` reads (currently `prep_test.parquet`) so requests hit the populated feature set in Redis. Mismatched IDs produce upstream 404s and a flat 503 response from the router.

```bash
cd /path/to/fraud_ml_platform
uv run python -c "
import json
import pandas as pd
df = pd.read_parquet('training/artifacts/prep_test.parquet', columns=['TransactionID'])
ids = df['TransactionID'].astype(int).tolist()
json.dump(ids, open('load/k6/transaction_ids.json', 'w'))
print(f'wrote {len(ids)} transaction IDs')
"
```

## Pre-flight check

Run before every load session. Catches empty/stale Redis, mismatched IDs, and routing-layer regressions in under a second.

```bash
# 1. Redis has features
docker exec fraud_ml_platform-redis-1 redis-cli DBSIZE
# Expect a non-zero count (~118k for full test parquet). If 0, run:
#   REDIS_HOST=localhost REDIS_PORT=6379 uv run python -m training.scripts.load_features_to_redis

# 2. A real ID from the IDs file scores cleanly through the router
TID=$(uv run python -c "import json; print(json.load(open('load/k6/transaction_ids.json'))[0])")
curl -s -X POST http://localhost:8081/score \
  -H 'Content-Type: application/json' \
  -d "{\"transaction_id\":$TID}" \
  -w "\nHTTP %{http_code}\n"
# Expect HTTP 200 and a JSON body with fraud_probability + model_version.
# HTTP 503 + body "upstream error" means the champion is returning non-200 (most
# commonly 404 from a feature miss). Do not proceed; fix Redis state first.
```

## Running tests

### Steady-state (100 RPS, 10 minutes)

```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_steady.js
```

### Ramp (10 -> 500 RPS over 22 minutes)

```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_ramp.js
```

### Breakdown ramp (100 -> 5000 RPS over 17 minutes)

Pushes until p95 or error rate cross SLO. Capture Prometheus snapshots during the run to attribute the bottleneck (router CPU, Redis op latency, ONNX inference duration).

```bash
ROUTER_URL=http://localhost:8081 k6 run --summary-export=load/results/breakdown_$(date +%Y%m%d).json load/k6/score_breakdown.js
```

While the test is running, in another terminal:

```bash
# router + champion + challenger CPU/mem over time
docker stats --no-stream router champion-model challenger-model redis > load/results/docker_stats_$(date +%Y%m%d).txt

# Grafana: open the inference latency board and screenshot the cliff
```

### Failure injection (100 RPS with champion paused at t=5m)

In one terminal:
```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_with_failures.js
```

In another terminal, at t=5m:
```bash
docker compose pause champion-model && sleep 30 && docker compose unpause champion-model
```

## SLOs

Declared in PRODUCTION.md. Measured on a developer laptop (see results below for hardware spec).

## Results

Hardware: AMD Ryzen 9 9950X3D 16-Core (32 threads), 30GiB RAM, Docker with no resource limits.

### Steady-state (2026-04-30)

100 RPS constant for 10 minutes. 60,001 requests.

| Metric     | Value  | SLO    |
|------------|--------|--------|
| p50        | 0.56ms | -      |
| p95        | 3.83ms | <50ms  |
| p99        | 7.08ms | <200ms |
| error rate | 0.00%  | <0.5%  |
| result     | PASS   |        |

### Breakdown ramp (2026-05-16)

100 to 5000 RPS planned over 4 stages, auto-abort at 5% error rate. 183,741 requests over 5m18s before abort. Run against the v2 retrain with the `Transport` fix in `router/internal/upstream/client.go` applied.

| Metric              | Value                  | SLO             |
|---------------------|------------------------|-----------------|
| p95 at abort        | 64.44ms                | <200ms          |
| p99 at abort        | 125.30ms               | -               |
| max                 | 298.86ms               | -               |
| error rate at abort | 5.30%                  | abort threshold |
| peak RPS reached    | ~1118                  |                 |
| breakdown point     | ~1000 RPS              |                 |
| failure mode        | hard 503s, no queueing |                 |
| result              | cliff identified       |                 |

Champion ONNX inference is the bottleneck. At 311 RPS in stage 1 the champion container already consumed ~16 host cores (1645% docker stats CPU). Linear extrapolation puts saturation at ~600-700 RPS sustained; observed cliff onset is consistent with that, after which 503s climb as the kernel's accept queue and Go scheduler thrash.

Latency stays under the 200ms p95 SLO all the way through the cliff. Failures are explicit 503s from the router's upstream client, not ctx timeouts (max latency 299ms, well under the 500ms upstream deadline). This is the desired degradation pattern: fail fast rather than pile up.

Router CPU stayed below 10% throughout (Go scheduler + JSON marshalling, lightweight). Redis stayed at ~1-2% CPU and 428MB RAM (the loaded test feature set). Neither is close to saturating; capacity work belongs at the model layer.

### Notes on the earlier ramp (2026-04-30)

A 10 to 500 RPS ramp was previously recorded with 0% errors and p95 32ms. That result is still valid for its declared scope but did not push the system to breakdown - 500 RPS was well below the cliff identified above. The breakdown ramp supersedes it.

### Failure injection (2026-04-30)

100 RPS for 10 minutes, champion container paused twice for 30s each (~7 minutes in). 60,000 requests.

| Metric                     | Value         | SLO               |
|----------------------------|---------------|-------------------|
| overall error rate         | 9.95%         | <10%              |
| error behaviour            | 503 (correct) | no silent failure |
| p95 on successful requests | 4.73ms        | -                 |
| recovery after unpause     | immediate     | -                 |
| crash/data loss            | none          | none              |
| result                     | PASS          |                   |

Router returned 503 during pause windows (not timeouts or silent drops). Recovery was instant on unpause.
