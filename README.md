# Fraud ML Platform

XGBoost fraud classifier trained on IEEE-CIS. The training pipeline covers leakage-aware temporal CV, adversarial validation, Optuna tuning, calibration, and SHAP. A Go service reads pre-encoded features from Redis, runs ONNX inference, and logs every prediction to Postgres.

The focus is modeling correctness. The Go layer exists to show the ONNX model survives a language boundary with verified parity.

---

## Results

Baselines are 5-fold `PurgedTimeSeriesSplit` CV means. Tuned XGBoost is evaluated on the held-out test set (last 20% by time). Source of truth: `training/artifacts/results.json` and `training/artifacts/baseline_scores.json`.

| Model                             | PR-AUC    | ROC-AUC   | Eval              |
|-----------------------------------|-----------|-----------|-------------------|
| Logistic Regression               | 0.272     | 0.807     | CV mean           |
| Random Forest                     | 0.405     | 0.836     | CV mean           |
| LightGBM                          | 0.527     | 0.878     | CV mean           |
| XGBoost untuned                   | 0.540     | 0.892     | CV mean           |
| XGBoost tuned (200 Optuna trials) | 0.568     | -         | CV best           |
| **XGBoost tuned (test set)**      | **0.505** | **0.899** | **Held-out test** |

Optimal threshold (tuned on validation): 0.596. F1 at threshold: 0.510.

Adversarial validation AUC: 0.823. The train and test pools are distinguishable, confirming temporal drift. Top 10 drift-heavy features pruned before training.

Pruned feature count rationale: the top 10 drift-heavy features are dropped. These ranked high in the adversarial classifier (train-vs-test distinguishability) but contributed little to fraud detection. Dropping them improved CV stability with negligible PR-AUC impact.

---

## Architecture

```
[Kaggle data] -> [Python training pipeline] -> [ONNX artifact]
                                                    |
                        [Redis feature cache] <-> [Go HTTP server] -> [Postgres prediction log]
                               |
                    [Python loader script]
```

Prediction path per request:
1. Go service receives `POST /score {transaction_id}`
2. Feature lookup: `GET fraud:features:{id}` from Redis (JSON-encoded float vector)
3. ONNX Runtime inference: features ordered by `onnx_feature_order.json`
4. Response returned immediately with fraud probability and flag
5. Prediction record written asynchronously to Postgres via a buffered channel

---

## What this project does NOT include

- **No feature store (Feast).** The parquet column list is the schema; `onnx_feature_order.json` enforces it at serving time. Feast's value lands in phase 2 when point-in-time correctness is needed.
- **No Kafka.** Predictions go directly to Postgres. Phase 2's drift detector is a SQL query, not a Kafka consumer. The swap is straightforward if the volume outgrows a single instance.
- **No load testing.** Latency (p95 < 10ms warm) is from a single local request, not measured under load.
- **No drift detection, shadow mode, or canary deployment.** These are in the phase 2 design doc.

---

## Honest caveats

- IEEE-CIS features V1..V339, C1..C14, D1..D15, M1..M9 are anonymized. SHAP shows which features matter but cannot explain why without Vesta's private feature dictionary. Analysis focuses on named columns (TransactionAmt, card1-6, addr1-2, P_emaildomain, ProductCD, engineered time/email features).
- Labels in IEEE-CIS come from Vesta's definition: chargebacks within 120 days plus manual review. Real production fraud detection deals with label latency and ambiguity that this dataset papers over.
- The serving layer architecture is production-grade; the deployment scale is single-machine demo. Real-traffic scaling is asserted, not measured.
- Unknown categories in the categorical encoder map to code -1 at serving time. This is documented behavior; the Go service loads the encoder at startup for future raw-feature API support.

---

## Negative result: isotonic calibration on a single window

Isotonic calibration fit on the validation window (latest 10% of the training pool) did not improve test-set metrics and slightly degraded PR-AUC.

The fitted step function collapses ranges of scores into ties, which loses ranking information when the test distribution sits on different breakpoints than the validation window the calibrator was fit on.

The uncalibrated model is used for the ONNX export and served predictions. The calibrated pkl is retained for reference.

---

## Cost notes

- GPU training time: 50 min on a single CUDA GPU. Breakdown: EDA, preprocess, adversarial: 5 min; baselines: 10 min; Optuna 200 trials (5-fold CV each): 33 min; evaluate, export: 2 min.
- Inference cost per prediction: ONNX Runtime on CPU under 1ms; Redis GET and Postgres async write add 2-5ms in local Docker. Smoke test round-trip under 10ms.
- Projected at 1k QPS: a single Go instance handles it; the async Postgres buffer absorbs write spikes. Redis is not the bottleneck at this rate.
- Projected at 10k QPS: the Postgres write path needs pgBouncer or a Kafka swap. Redis and the Go ONNX inference path scale horizontally without coordination.

---

## How to run

### Prerequisites

- Python 3.11+, uv
- Go 1.22+
- Docker + Docker Compose
- GPU with CUDA 12.x (CPU fallback: remove `device='cuda'` from training scripts)
- Kaggle CLI configured

### Training

```bash
# Download data
mkdir -p data/ieee_cis
kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis/
cd data/ieee_cis && unzip ieee-fraud-detection.zip && cd ../..

# Install Python deps
uv sync

# Run training pipeline (sequential)
uv run python training/scripts/eda.py
uv run python training/scripts/preprocess.py
uv run python training/scripts/adversarial_validation.py
uv run python training/scripts/baselines.py
uv run python training/scripts/tune.py
uv run python training/scripts/evaluate.py
uv run python training/scripts/export_onnx.py
```

### Serving

```bash
# Start Redis and Postgres
docker compose up -d redis postgres

# Load feature cache
uv run python training/scripts/load_features_to_redis.py

# Build and start serving
docker compose up --build serving

# Smoke test
bash scripts/smoke_test.sh
```

### Testing

```bash
# Python tests
uv run pytest tests/python/ -v

# Go tests
cd serving && go test ./...
```

### Linting

```bash
uv run ruff check training/
cd serving && go vet ./...
cd serving && gofmt -l .
```
