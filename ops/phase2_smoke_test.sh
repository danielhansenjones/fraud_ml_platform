#!/usr/bin/env bash
set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://localhost:8081}"
POSTGRES_DSN="${POSTGRES_DSN:-postgres://fraud:fraud@localhost:5432/fraud}"

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; exit 1; }

echo "=== phase 2 smoke test ==="

echo "1. Router health..."
status=$(curl -s -o /dev/null -w "%{http_code}" "$ROUTER_URL/health")
[ "$status" = "200" ] && pass "router /health" || fail "router /health returned $status"

echo "2. Sending 100 /score requests..."
TX_IDS_FILE="${TX_IDS_FILE:-load/k6/transaction_ids.json}"
if [ ! -f "$TX_IDS_FILE" ]; then
    fail "transaction id file not found: $TX_IDS_FILE (generate it via the k6 setup step in PRODUCTION.md)"
fi
# Reuse the persisted ID list rather than re-shelling pandas on every smoke run.
TRANSACTION_IDS=($(python3 -c "import json; print('\n'.join(map(str, json.load(open('$TX_IDS_FILE'))[:100])))"))
if [ ${#TRANSACTION_IDS[@]} -eq 0 ]; then
    fail "no transaction IDs available in $TX_IDS_FILE"
fi

ok_count=0
for tx_id in "${TRANSACTION_IDS[@]}"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$ROUTER_URL/score" \
        -H "Content-Type: application/json" \
        -d "{\"transaction_id\": $tx_id}")
    [ "$code" = "200" ] && ok_count=$((ok_count + 1))
done
[ "$ok_count" -eq "${#TRANSACTION_IDS[@]}" ] && pass "all ${#TRANSACTION_IDS[@]} /score requests returned 200" || fail "$ok_count/${#TRANSACTION_IDS[@]} succeeded"

echo "3. Checking predictions table..."
count=$(psql "$POSTGRES_DSN" -t -c "SELECT count(*) FROM predictions;" 2>/dev/null | tr -d ' ')
[ "$count" -gt 0 ] && pass "predictions table has $count rows" || fail "predictions table is empty"

echo "4. Checking shadow_comparisons..."
shadow_count=$(psql "$POSTGRES_DSN" -t -c "SELECT count(*) FROM shadow_comparisons;" 2>/dev/null | tr -d ' ')
[ "$shadow_count" -gt 0 ] && pass "shadow_comparisons has $shadow_count rows" || echo "[WARN] shadow_comparisons is empty (shadow may be disabled)"

echo "5. Waiting for drift detector (up to 60s)..."
for i in $(seq 1 12); do
    drift_count=$(psql "$POSTGRES_DSN" -t -c "SELECT count(*) FROM drift_alerts;" 2>/dev/null | tr -d ' ')
    if [ "$drift_count" -gt 0 ]; then
        pass "drift_alerts has $drift_count rows after $((i * 5))s"
        break
    fi
    sleep 5
done
if [ "$drift_count" -eq 0 ]; then
    echo "[WARN] no drift alerts yet (may be insufficient samples - this is expected on first run)"
fi

echo "6. Checking labels table..."
label_count=$(psql "$POSTGRES_DSN" -t -c "SELECT count(*) FROM labels;" 2>/dev/null | tr -d ' ')
[ "$label_count" -gt 0 ] && pass "labels table has $label_count rows" || echo "[WARN] labels table empty (label joiner may not have run yet)"

echo "=== smoke test complete ==="
