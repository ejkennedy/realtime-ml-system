#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports

wait_for_server() {
  local attempts=0
  while (( attempts < 60 )); do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "http://localhost:8000/" \
      -H "Content-Type: application/json" \
      -d '{
        "event_id": "benchmark-probe",
        "timestamp": "2026-03-31T00:00:00Z",
        "timestamp_unix_ms": 1774915200000,
        "hour_of_day": 0,
        "day_of_week": 0,
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
        "velocity_tx_count_1m": 1, "velocity_tx_count_5m": 2, "velocity_tx_count_1h": 5, "velocity_tx_count_24h": 12,
        "velocity_amount_sum_1h": 300.0, "velocity_amount_avg_1h": 60.0, "velocity_amount_max_1h": 149.99,
        "velocity_amount_sum_24h": 850.0, "velocity_distinct_merchants_1h": 2, "velocity_distinct_countries_1h": 1,
        "card_risk_score": 0.1, "merchant_fraud_rate_30d": 0.01,
        "merchant_avg_amount": 55.0, "card_avg_spend_30d": 65.0
      }' || true)
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    attempts=$((attempts + 1))
    sleep 1
  done
  return 1
}

run_case() {
  local name="$1"
  local scorer_replicas="$2"
  local pool_size="$3"
  local quantized="${4:-false}"
  local log_path="reports/${name}_serve.log"
  local html_path="reports/${name}.html"

  ray stop --force >/dev/null 2>&1 || true

  mkdir -p ".ray_tmp/${name}"
  RAY_TMPDIR="$ROOT_DIR/.ray_tmp/${name}" \
  SHADOW_ENABLED=false \
  VERSION_MANAGER_ENABLED=false \
  ONLINE_UPDATES_ENABLED=false \
  SERVE_USE_ROUTER=false \
  SERVE_USE_QUANTIZED_MODEL="$quantized" \
  SERVE_SCORER_REPLICAS="$scorer_replicas" \
  SERVE_ROUTER_REPLICAS=1 \
  SERVE_ROUTER_MAX_ONGOING_REQUESTS=32 \
  ONNX_SESSION_POOL_SIZE="$pool_size" \
  ONNX_INTRA_OP_THREADS=1 \
  ONNX_INTER_OP_THREADS=1 \
  uv run --package serving python -m serving.app >"$log_path" 2>&1 &
  local serve_pid=$!

  cleanup() {
    kill "$serve_pid" >/dev/null 2>&1 || true
    wait "$serve_pid" >/dev/null 2>&1 || true
    ray stop --force >/dev/null 2>&1 || true
  }
  trap cleanup RETURN

  wait_for_server

  LOAD_TEST_LABEL="$name" \
  LOAD_TEST_ARTIFACT_STEM="$name" \
  LOAD_TEST_STAGE1_DURATION_S=5 \
  LOAD_TEST_STAGE1_USERS=10 \
  LOAD_TEST_STAGE1_SPAWN_RATE=2 \
  LOAD_TEST_STAGE2_DURATION_S=25 \
  LOAD_TEST_STAGE2_USERS=30 \
  LOAD_TEST_STAGE2_SPAWN_RATE=5 \
  LOAD_TEST_STAGE3_DURATION_S=30 \
  LOAD_TEST_STAGE3_USERS=0 \
  LOAD_TEST_STAGE3_SPAWN_RATE=10 \
  uv run --package load-testing locust \
    -f services/load-testing/src/load_testing/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    --html "$html_path"
}

run_case "benchmark_r1_s1" 1 1
run_case "benchmark_r1_s2" 1 2
run_case "benchmark_r2_s2" 2 2
if [[ -f "models/registry/fraud_detector/latest/model.int8.onnx" ]]; then
  run_case "benchmark_q_r1_s1" 1 1 true
fi

{
  echo "# Benchmark Matrix"
  echo
  echo "| Config | Total requests | Failures | Failure rate | Avg RPS | p50 | p95 | p99 | Mean | SLA |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
  for name in benchmark_r1_s1 benchmark_r1_s2 benchmark_r2_s2 benchmark_q_r1_s1; do
    file="reports/load_test_summary_${name}.md"
    [[ -f "$file" ]] || continue
    total_requests=$(awk -F'|' '/Total requests/ {gsub(/ /, "", $3); print $3}' "$file")
    total_failures=$(awk -F'|' '/Total failures/ {gsub(/ /, "", $3); print $3}' "$file")
    failure_rate=$(awk -F'|' '/Failure rate/ {gsub(/ /, "", $3); print $3}' "$file")
    avg_rps=$(awk -F'|' '/Average RPS/ {gsub(/ /, "", $3); print $3}' "$file")
    p50=$(awk -F'|' '/\| p50 / {gsub(/ /, "", $3); print $3}' "$file")
    p95=$(awk -F'|' '/\| p95 / {gsub(/ /, "", $3); print $3}' "$file")
    p99=$(awk -F'|' '/\| p99 / {gsub(/ /, "", $3); print $3}' "$file")
    mean=$(awk -F'|' '/\| Mean / {gsub(/ /, "", $3); print $3}' "$file")
    sla=$(awk -F'`' '/Result:/ {print $2 " " $4}' "$file")
    echo "| ${name#benchmark_} | $total_requests | $total_failures | $failure_rate | $avg_rps | $p50 | $p95 | $p99 | $mean | $sla |"
  done
} > reports/benchmark_matrix.md

echo "Wrote reports/benchmark_matrix.md"
