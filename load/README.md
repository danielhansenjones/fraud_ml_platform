# Load Testing

k6 load tests for the router service.

## Prerequisites

- k6 installed: https://k6.io/docs/getting-started/installation/
- Router running and reachable
- `load/k6/transaction_ids.json` generated (see below)

## Generating transaction_ids.json

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

## Running tests

### Steady-state (100 RPS, 10 minutes)

```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_steady.js
```

### Ramp (10 -> 500 RPS over 22 minutes)

```bash
ROUTER_URL=http://localhost:8081 k6 run load/k6/score_ramp.js
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

| Metric | Value | SLO |
|--------|-------|-----|
| p50 | 0.56ms | - |
| p95 | 3.83ms | <50ms |
| p99 | 7.08ms | <200ms |
| error rate | 0.00% | <0.5% |
| result | PASS | |

### Ramp (2026-04-30)

10 to 500 RPS over 3 stages, 22 minutes. 409,200 requests.

| Metric | Value | SLO |
|--------|-------|-----|
| p95 at peak | 32.45ms | <200ms |
| error rate | 0.00% | <1% |
| breakdown point | not reached at 500 RPS | |
| result | PASS | |

No breakdown found at the maximum tested load. The system handled 500 RPS within SLO thresholds.

### Failure injection (2026-04-30)

100 RPS for 10 minutes, champion container paused twice for 30s each (~7 minutes in). 60,000 requests.

| Metric | Value | SLO |
|--------|-------|-----|
| overall error rate | 9.95% | <10% |
| error behaviour | 503 (correct) | no silent failure |
| p95 on successful requests | 4.73ms | - |
| recovery after unpause | immediate | - |
| crash/data loss | none | none |
| result | PASS | |

Router returned 503 during pause windows (not timeouts or silent drops). Recovery was instant on unpause.
