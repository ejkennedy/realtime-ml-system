#!/usr/bin/env bash
# End-to-end smoke test: send a transaction to Ray Serve, verify response shape and latency

set -euo pipefail

HOST="${RAY_SERVE_HOST:-http://localhost:8000}"
MAX_LATENCY_MS=200

echo "=== Smoke Test: Fraud Detection Serving ==="
echo "Target: $HOST"
echo ""

PAYLOAD='{
  "event_id": "smoke-test-001",
  "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
  "card_id": "card_000001",
  "merchant_id": "merch_0001",
  "merchant_category": "online",
  "amount": 149.99,
  "currency": "USD",
  "country_code": "US",
  "pos_type": "online",
  "mcc": 5411,
  "velocity": {
    "tx_count_1m": 1, "tx_count_5m": 2, "tx_count_1h": 5, "tx_count_24h": 12,
    "amount_sum_1h": 300.0, "amount_avg_1h": 60.0, "amount_max_1h": 149.99,
    "amount_sum_24h": 850.0, "distinct_merchants_1h": 2, "distinct_countries_1h": 1
  },
  "card_risk_score": 0.1, "merchant_fraud_rate_30d": 0.01,
  "merchant_avg_amount": 55.0, "card_avg_spend_30d": 65.0
}'

START=$(date +%s%N)
RESPONSE=$(curl -s -X POST "$HOST/score" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")
END=$(date +%s%N)
LATENCY_MS=$(( (END - START) / 1000000 ))

echo "Response: $RESPONSE"
echo ""
echo "Client-side latency: ${LATENCY_MS}ms"

# Validate response shape
echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
assert 'fraud_score' in r, 'Missing fraud_score'
assert 'is_fraud' in r, 'Missing is_fraud'
assert 'model_version' in r, 'Missing model_version'
assert 'inference_latency_ms' in r, 'Missing inference_latency_ms'
assert 0 <= r['fraud_score'] <= 1, f'fraud_score out of range: {r[\"fraud_score\"]}'
print(f'✓ Response valid: fraud_score={r[\"fraud_score\"]:.4f}, is_fraud={r[\"is_fraud\"]}, model={r[\"model_version\"]}')
print(f'✓ Server-reported latency: {r[\"inference_latency_ms\"]}ms')
"

if [ "$LATENCY_MS" -gt "$MAX_LATENCY_MS" ]; then
  echo "⚠ WARNING: Client latency ${LATENCY_MS}ms > ${MAX_LATENCY_MS}ms (includes network)"
else
  echo "✓ Latency within bounds"
fi

echo ""
echo "=== Smoke test passed ==="
