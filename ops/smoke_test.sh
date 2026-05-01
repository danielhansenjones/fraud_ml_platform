#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
FLUSH_WAIT_MS="${PREDICTION_FLUSH_INTERVAL_MS:-1000}"
POSTGRES_DSN="${POSTGRES_DSN:-postgres://fraud:fraud@localhost:5432/fraud?sslmode=disable}"

echo "Waiting for /health..."
for i in $(seq 1 30); do
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "Service is up."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: service did not come up in time"
        exit 1
    fi
    sleep 1
done

# Pick a known transaction_id from the test parquet
TXN_ID=$(python3 -c "
import pandas as pd, json
df = pd.read_parquet('training/artifacts/prep_test.parquet')
print(int(df['TransactionID'].iloc[0]))
")

echo "Scoring transaction_id=${TXN_ID}..."
RESPONSE=$(curl -sf -X POST "${BASE_URL}/score" \
    -H "Content-Type: application/json" \
    -d "{\"transaction_id\": ${TXN_ID}}")

echo "Response: ${RESPONSE}"

# Assert fraud_probability field exists
echo "${RESPONSE}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
assert 'fraud_probability' in r, 'missing fraud_probability'
assert 'flagged' in r, 'missing flagged'
assert 'model_version' in r, 'missing model_version'
print('Response shape OK')
"

# Wait for prediction flush
WAIT_SECS=$(( (FLUSH_WAIT_MS + 500) / 1000 + 1 ))
echo "Waiting ${WAIT_SECS}s for prediction flush..."
sleep "${WAIT_SECS}"

# Check Postgres
echo "Checking Postgres for prediction row..."
ROW_COUNT=$(psql "${POSTGRES_DSN}" -t -c \
    "SELECT COUNT(*) FROM predictions WHERE transaction_id = ${TXN_ID};")
ROW_COUNT=$(echo "${ROW_COUNT}" | tr -d ' ')

if [ "${ROW_COUNT}" -ge 1 ]; then
    echo "Prediction row found in Postgres. Smoke test PASSED."
else
    echo "ERROR: prediction row not found in Postgres for transaction_id=${TXN_ID}"
    exit 1
fi
