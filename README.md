# Real-Time ML System — Streaming Fraud Detection

Real-time fraud scoring with streaming features, ONNX serving, shadow deployment, concept drift detection, and online learning. The architecture target is sub-50ms p95 on tuned infrastructure; local laptop runs are primarily for functional validation and relative performance comparisons.

## Architecture

```
Transactions (synthetic / live)
        │
        ▼
  Redpanda (Kafka)  ←── transactions-raw (12 partitions)
        │
        ▼
  Apache Flink  ──── Velocity features (RocksDB, exactly-once)
        │         └── Feast enrichment (Redis pipeline)
        ▼
  transactions-enriched topic
        │
        ▼
  Ray Serve ──────── FraudRouter
        │                 ├── FraudScorer [primary]  → ONNX Runtime (pooled sessions)
        │                 └── FraudScorer [shadow]   → fire-and-forget
        │
        ▼
  transactions-scored topic  ──► Iceberg (offline store)
        │
        ▼
  Drift Detector (Evidently)  ──► Retraining trigger
        │
        ▼
  XGBoost training  ──► ONNX export  ──► MLflow registry  ──► shadow deploy  ──► promote
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Streaming | Redpanda, Apache Flink (PyFlink 1.19), exactly-once semantics |
| Feature Store | Feast 0.40, Redis (online), Apache Iceberg (offline) |
| Serving | Ray Serve 2.32, ONNX Runtime 1.18, session pool, GC tuning |
| MLOps | MLflow 2.14, shadow deployment, one-click rollback |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Online Learning | SGD micro-batches (scikit-learn), LinUCB contextual bandit |
| Infra | Docker Compose (local), Kubernetes + Helm + Terraform (prod) |

## Quick Start

### Prerequisites
- Docker Desktop with 8GB+ RAM allocated
- Python 3.12 (`uv` manages this automatically)
- `uv` installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. Install Python dependencies

```bash
cp .env.example .env
uv sync
```

### 2. Start infrastructure

```bash
make up
```

This starts: Redpanda, Flink, Ray, Redis, MLflow, MinIO, Prometheus, Grafana.

### 3. Train initial model

```bash
make train
```

Generates 500k synthetic transactions, trains XGBoost, exports FP32 ONNX, attempts
an int8 quantized ONNX variant when the graph supports ORT quantization, and
registers in MLflow.

### 4. Apply feature store

```bash
make seed-data      # generate synthetic Feast feature tables
make feast-apply    # register feature views
make feast-materialize  # load features into Redis
```

### 5. Start serving

```bash
make serve          # or make serve-docker if stack is running
```

`make serve` is a long-running foreground process. The API is ready once you see
`Deployed app 'fraud-detection' successfully` in the logs. The later 30-minute
`Shadow period ongoing` messages are background model-promotion checks, not
startup work.

### 6. Smoke test

```bash
make smoke-test
```

Run `make smoke-test` from a second terminal while `make serve` is still running.

### 7. Start the streaming path

```bash
make flink-job      # submit the PyFlink feature pipeline
make produce        # emit synthetic transactions into Redpanda
```

### 8. Run load tests

```bash
make load-test-local   # recommended local baseline
make load-test         # aggressive stress profile
```

Results are saved to `reports/load_test_*.html`.
Latency distribution PNGs are generated alongside the HTML report in `reports/`.
Each run also writes a compact Markdown summary to `reports/load_test_summary_*.md`.
By default, slow 200 responses are kept as successes so the report reflects
actual HTTP failure rate. To make latency breaches fail the run, set
`LOAD_TEST_LATENCY_FAIL_THRESHOLD_MS`, for example:
`LOAD_TEST_LATENCY_FAIL_THRESHOLD_MS=200 make load-test`.

For laptop-friendly benchmarking, use:

```bash
make serve-perf
make load-test-local
```

`make serve-perf` disables shadow traffic and the version manager, and uses fewer
Ray replicas / ONNX sessions to reduce local CPU oversubscription. It also uses a
short dedicated Ray temp directory, bypasses the router when shadowing is off, and
disables online-update writes so you can measure the scoring path rather than
background MLOps overhead.

To benchmark the quantized model path:

```bash
make serve-perf-quantized
make load-test-local
```

To emit ONNX Runtime profiling traces during a run:
`ONNX_PROFILE_ENABLED=true make serve-perf`

## Key Design Decisions

### Sub-50ms p95 Latency

Five patterns work together to approach the SLA:

1. **Pre-warmed ONNX session pool** — sessions created at startup, never on request path
2. **Typed request decoding** — `msgspec` decodes request bytes into a fixed schema, avoiding repeated dict lookups and JSON object churn
3. **Direct perf path** — local perf mode binds HTTP directly to `FraudScorer` when shadowing is off
4. **Bounded concurrency** — local defaults use one scorer replica / one ONNX session to avoid CPU oversubscription
5. **Background GC + off-path updates** — ONNX sessions avoid stop-the-world GC, and online updates are queued off the request path
6. **Optional int8 ONNX path** — training emits a quantized model variant for benchmarking CPU inference tradeoffs

### Exactly-Once Semantics

Flink checkpointing with `EXACTLY_ONCE` + RocksDB incremental snapshots:
- 10s checkpoint interval, 25h state TTL
- Kafka transactional producer (`EXACTLY_ONCE` delivery guarantee)
- Tolerates 1 consecutive checkpoint failure before failing the job

### Shadow Deployment

```
FraudRouter
  ├── primary: always awaited → returned to caller
  └── shadow: asyncio.create_task() → fire-and-forget → Kafka topic
              hard 100ms timeout, exceptions swallowed
```

Shadow results flow to `shadow-results` topic for offline AUC comparison. Promotion only happens after 30 minutes with no metric regression (>1% tolerance).
This shadow window begins after serving is already live; it does not delay requests.

### Drift Detection → Retraining

Evidently AI checks every 15 minutes:
- Feature drift (KS test): trigger if >30% of features drift
- Target drift (PSI): trigger if PSI >0.25
- Critical features: trigger if `amount`, `tx_count_1h`, `card_risk_score` all drift

New model goes to shadow deployment first — no auto-promotion without human approval.

### Online Learning

Two layers update without full retraining:

1. **SGD adapter** — `SGDClassifier.partial_fit()` on 100-item micro-batches from Redis queue. Updates every ~0.5s, serialised back to Redis. All Ray workers reload periodically.

2. **LinUCB bandit** — adapts the fraud decision threshold per `(merchant_category, hour_bucket)` context. Learns from delayed labels (actual fraud outcomes) with asymmetric rewards: false negatives penalised 4× more than false positives.

## Load Testing Modes

- `make load-test-local`: lighter profile for laptop benchmarking and code-change comparisons
- `make load-test`: aggressive stress profile intended to saturate the local stack
- `bash scripts/benchmark_matrix.sh`: compares replica / session configurations side by side
- if `models/registry/fraud_detector/latest/model.int8.onnx` exists, the benchmark matrix also includes a quantized `q_r1_s1` case

Treat the local benchmark as a relative signal, not a production claim. A full
local stack with Ray, Redis, MLflow, Flink, and Locust running on one machine is
not expected to reproduce a production latency envelope.

## Services

| Service | URL | Purpose |
|---------|-----|---------|
| Ray Serve | http://localhost:8000 | Inference endpoint |
| Ray Dashboard | http://localhost:8265 | Serving health |
| Flink UI | http://localhost:8081 | Stream job monitoring |
| MLflow | http://localhost:5001 | Model registry |
| Grafana | http://localhost:3000 | Latency dashboards |
| Prometheus | http://localhost:9090 | Metrics |
| Redpanda Console | http://localhost:8080 | Kafka topics |
| MinIO | http://localhost:9001 | Object storage |
| Evidently | http://localhost:8085 | Drift reports |

## Notes

- `make serve` is expected to keep running until you stop it with `Ctrl+C`.
- `make serve-perf` is the recommended path for local latency work.
- `make serve-docker` only restarts the `ray-head` container from the Docker stack started by `make up`.
- `make smoke-test` validates serving only; it does not require the Flink job or producer.
- The full streaming demo needs both `make flink-job` and `make produce`.

## Prometheus Alerts

- `InferenceLatencyHigh` — p95 > 50ms for 2m
- `InferenceLatencyCritical` — p95 > 100ms for 1m
- `FraudModelDriftHigh` — drift_share > 30% for 10m
- `FraudModelDriftCritical` — drift_share > 40% for 5m
- `ServingErrorRateHigh` — error rate > 1% for 2m

## MLOps Operations

### Rollback model

```bash
make rollback
```

Reverts to the previous `production` alias in MLflow and hot-swaps the ONNX session pool. Zero downtime.

### Check shadow deployment status

```bash
make shadow-status
```

### Trigger drift check manually

```bash
make drift-check
```

## Project Structure

```
realtime-ml-system/
├── services/
│   ├── streaming/      PyFlink feature pipeline (velocity + enrichment)
│   ├── serving/        Ray Serve + ONNX Runtime + shadow deployment
│   ├── feature-store/  Feast definitions, Redis materialisation
│   ├── training/       XGBoost + ONNX export + online learning
│   ├── monitoring/     Evidently drift detection + retraining trigger
│   └── load-testing/   Locust 10k req/s + latency distribution plots
├── infra/
│   ├── docker/         docker-compose + Dockerfiles + Prometheus config
│   ├── k8s/helm/       Helm chart with HPA + PDB
│   └── terraform/      EKS + MSK + ElastiCache + S3 modules
├── scripts/            Bootstrap, smoke test, rollback, producer
└── data/               Synthetic data generation
```
