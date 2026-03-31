# Operations Runbook

## Local Development

### First-time Setup

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Install Python dependencies
uv sync

# 3. Start infrastructure (Redpanda, Flink, Ray, Redis, MLflow, MinIO, Prometheus, Grafana)
make up
# Wait ~30 seconds for all services to initialise

# 4. Train the initial model
make train
# → models/registry/fraud_detector/latest/model.onnx

# 5. Generate synthetic feature data and materialise to Redis
make seed-data
make feast-apply
make feast-materialize

# 6. Start serving in a separate terminal
make serve
# Wait for: "Deployed app 'fraud-detection' successfully."

# 7. Verify the serving layer is up
make smoke-test

# 8. Optional: run the streaming path
make flink-job
make produce
```

### Daily Development Workflow

```bash
make up            # start all services
make produce       # start synthetic transaction producer (optional)
make flink-job     # submit Flink feature pipeline (optional)
make serve         # start Ray Serve locally in the foreground
make smoke-test    # sanity check
make down          # stop everything
```

For latency-focused work, prefer:

```bash
make serve-perf
make load-test-local
```

This path disables shadow traffic, version-manager polling, and online-update
Redis writes, binds HTTP directly to the scorer, and uses a short dedicated Ray temp
directory to avoid `/tmp/ray` spill warnings on space-constrained machines.

To test the quantized model path locally:

```bash
make serve-perf-quantized
make load-test-local
```

## Training

### Full Training Run

```bash
make train
# Uses 500k synthetic samples
# Exports to ONNX, registers in MLflow as "staging"
# Does NOT auto-promote — goes through shadow deployment
```

### Quick Training (50k samples, ~30s)

```bash
make train-quick
```

### Training with Iceberg Offline Data (production)

```bash
uv run --package training python -m training.pipeline --iceberg
```

### What Happens During Training

1. Loads data (synthetic or Iceberg)
2. 5-fold stratified cross-validation (XGBoost, 100 trees per fold) — logged to MLflow
3. Final model trained on 90% of data, validated on 10%
4. Metrics logged: AUC, AUCPR, precision, recall, F1
5. Model exported to ONNX via onnxmltools, validated against original predictions
6. Optional int8 quantized ONNX variant emitted and validated when the exported graph supports ORT quantization
7. Registered in MLflow under `fraud-detector` model name with `staging` alias
8. Version manager picks up the `staging` alias and loads into `ShadowScorer`

`make train` finishes when the model is logged and registered. The later
30-minute shadow window belongs to serving-side promotion logic and does not
block training completion.

## Model Promotion and Rollback

### Shadow Deployment Flow

When a new model is trained:
1. It is registered in MLflow with `staging` alias
2. `VersionManager` polls MLflow every 60s and detects the new staging model
3. The model is loaded into the `ShadowScorer` deployment (never affects primary)
4. After 30 minutes, shadow comparison metrics are evaluated
5. If new model AUC >= baseline - 1%: auto-promote to `production`
6. If regression detected: hold promotion, log warning, alert

This happens after serving is already available. The `Shadow period ongoing`
log lines are background status messages, not startup progress.

### Manual Promotion

```bash
# Promote staging to production immediately (bypasses shadow window)
uv run --package training python -m training.pipeline --auto-promote
```

### One-Click Rollback

```bash
make rollback
```

This calls `VersionManager.rollback()` which:
1. Queries MLflow for the previous `production` model version
2. Downloads its ONNX artifact
3. Hot-swaps the `OnnxSessionPool` (zero downtime, pool drains then refills)
4. Updates the `production` alias in MLflow

### Check Current Model Versions

```bash
# Open MLflow UI
open http://localhost:5001

# Or via CLI
uv run python -c "
import mlflow
client = mlflow.MlflowClient(tracking_uri='http://localhost:5001')
versions = client.search_model_versions(\"name='fraud-detector'\")
for v in versions[:5]:
    print(f'v{v.version}: {v.aliases} | AUC: {v.run_id[:8]}')
"
```

## Monitoring

### Check Drift Manually

```bash
make drift-check
# Runs Evidently AI report against last 1 hour of scored transactions
# Saves HTML report to reports/
# Triggers retraining if thresholds exceeded
```

### Grafana Dashboards

Open http://localhost:3000 (admin/admin)

**Fraud Detection Overview** dashboard shows:
- p50/p95/p99 inference latency (time series)
- Request rate and error rate
- Fraud prediction rate
- Session pool wait time
- Feature drift share gauge

### Prometheus Alerts

| Alert | Threshold | Severity |
|-------|-----------|---------|
| `InferenceLatencyHigh` | p95 > 50ms for 2m | warning |
| `InferenceLatencyCritical` | p95 > 100ms for 1m | critical |
| `FraudModelDriftHigh` | drift_share > 30% for 10m | warning |
| `FraudModelDriftCritical` | drift_share > 40% for 5m | critical |
| `FraudRateSpikeAnomaly` | fraud_rate > 15% for 5m | warning |
| `ServingErrorRateHigh` | error_rate > 1% for 2m | critical |

### Investigating Latency Regression

If p95 exceeds 50ms:

1. Check session pool wait time in Grafana (`fraud_session_pool_wait_seconds`)
   - If high: the scorer is queueing on ONNX sessions → reduce concurrency or retune pool size
2. Check ONNX inference time (`fraud_onnx_duration_seconds`)
   - If high: model runtime or ORT thread count is the bottleneck
3. Check feature-prep and request-parse histograms
   - If high: the hot path is paying for JSON shape / timestamp parsing overhead
   - The fast path now accepts `hour_of_day`, `day_of_week`, and `timestamp_unix_ms`
4. Check whether online updates are enabled
   - For benchmarks, keep `ONLINE_UPDATES_ENABLED=false`
5. If you need deeper model-runtime evidence, enable ORT profiling:
   ```bash
   ONNX_PROFILE_ENABLED=true make serve-perf
   ```
   Profiles will be written under `reports/onnx_profiles/`
6. Capture a point-in-time scorer breakdown after the run:
   ```bash
   make perf-breakdown
   ```
   This writes `reports/perf_breakdown_*.md` with request-parse, feature-prep,
   pool-wait, ONNX-run, and response-build summaries.
7. Check GC pressure:
   ```bash
   docker stats ray-head  # watch memory usage patterns
   ```
8. Run load test to profile at different concurrency levels:
   ```bash
   make load-test-local
   # Compare reports/load_test_summary_*.md across configs
   ```

## Load Testing

### Local Baseline Load Test

```bash
make serve-perf
make load-test-local
```

Generates `reports/load_test_local_TIMESTAMP.html` with:
- p50/p95/p99/p99.9 latency
- Requests per second
- Error rate
- Latency distribution histogram + CDF plot

The load test also writes a latency distribution PNG into `reports/` before exiting
and a compact Markdown summary to `reports/load_test_summary_TIMESTAMP.md`.

After a run, write the live scorer timing snapshot with:

```bash
make perf-breakdown
```

### Stress Load Test

```bash
make load-test
```

Use this as a saturation / stress run, not as the default laptop benchmark.

If the latest training run produced `model.int8.onnx`, you can compare the
quantized path with:

```bash
make serve-perf-quantized
make load-test-local
```

To fail the run when successful responses exceed a latency threshold, set
`LOAD_TEST_LATENCY_FAIL_THRESHOLD_MS`, for example:

```bash
LOAD_TEST_LATENCY_FAIL_THRESHOLD_MS=200 make load-test
```

For more realistic laptop benchmarking, run the lighter local profile:

```bash
make serve-perf
make load-test-local
```

`make serve-perf` disables shadow traffic and version-manager polling and uses
smaller local Ray / ONNX settings to reduce contention.

### Load Test with Web UI

```bash
make load-test-ui
# Open http://localhost:8089 to control the test interactively
```

### Interpreting Load Test Results

**Interpretation guide:**
- `make load-test-local` is for comparing code and config changes on one machine
- `make load-test` is expected to saturate a laptop and is useful for tail-latency stress only
- Sub-50ms p95 is the architecture target, not a guaranteed local-laptop outcome

**Common failure modes:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High pool wait time | Too much concurrency for available CPU | Reduce replicas / pool size or use direct perf mode |
| High ONNX time with low pool wait | ORT thread count or model runtime | Set `ONNX_INTRA_OP_THREADS=1` and retest |
| High feature-prep time | Request parsing / timestamp parsing overhead | Send `hour_of_day`, `day_of_week`, `timestamp_unix_ms` |
| High variance under local stress | Full-stack machine saturation | Use `make load-test-local` for baseline, `make load-test` for stress |

## Feature Store Operations

### Materialise Features (sync offline → online Redis)

```bash
make feast-materialize
# Materialises all feature views with data newer than last materialisation
```

For a clean local setup, run `make seed-data` and `make feast-apply` once before
the first `make feast-materialize`.

### Add a New Feature

1. Define it in `services/feature-store/src/feature_store/repo/features.py`
2. Add the column to the offline parquet source
3. Run `make feast-apply && make feast-materialize`
4. Add the feature to `FEATURE_NAMES` in `services/serving/src/serving/models/onnx_runner.py`
5. Retrain the model with the new feature

### Check Feature Freshness

```bash
uv run --package feature-store feast -c services/feature-store/src/feature_store/repo feature-views list
```

## Online Learning

### SGD Adapter Status

```bash
uv run python -c "
import redis, json
r = redis.Redis()
metrics = r.hgetall('online_model:sgd:metrics')
print({k.decode(): v.decode() for k, v in metrics.items()})
"
```

### Reset Online Model (if diverged)

```bash
uv run python -c "
import redis
r = redis.Redis()
r.delete('online_model:sgd:current')
r.delete('online_model:sgd:scaler')
print('Online model reset. Will re-initialise on next batch.')
"
```

### Bandit Status

```bash
uv run python -c "
import sys
sys.path.insert(0, 'services/training/src')
import redis
from training.online_learning.bandit import LinUCBBandit
r = redis.Redis()
bandit = LinUCBBandit.from_redis(r)
import numpy as np
# Show which threshold is preferred per hour bucket
for hour in [9, 14, 20, 2]:  # morning, afternoon, evening, night
    t = bandit.select_threshold(0.5, 4, hour)  # online merchant
    print(f'Hour {hour:02d}: preferred threshold = {t:.2f}')
"
```

## Kubernetes Deployment

### Prerequisites

- `kubectl` configured for your cluster
- `helm` 3.x
- Docker images built and pushed to registry

### Deploy

```bash
cd infra/k8s/helm

# Install umbrella chart
helm install fraud-detection ./fraud-detection \
  -f fraud-detection/values.prod.yaml \
  --namespace fraud-detection --create-namespace \
  --set kafka.bootstrapServers="your-msk-endpoint:9092" \
  --set mlflow.backendStore="postgresql://..." \
  --set redis.enabled=false \  # use ElastiCache
  --wait

# Check rollout
kubectl -n fraud-detection get pods
kubectl -n fraud-detection get hpa
```

### Upgrade (zero downtime)

```bash
helm upgrade fraud-detection ./fraud-detection \
  -n fraud-detection \
  --set rayServe.image.tag="new-version"
```

Ray Serve performs a rolling update: new replicas are warmed up before old ones are terminated.

### Rollback Helm Release

```bash
helm rollback fraud-detection -n fraud-detection
```

### Scale Serving Manually

```bash
kubectl -n fraud-detection scale deployment/ray-serve-fraud-scorer --replicas=10
```

## Terraform

### First Run

```bash
cd infra/terraform

# Create S3 backend bucket first (one-time)
aws s3 mb s3://fraud-detection-tfstate --region us-east-1

terraform init
terraform workspace new prod
terraform plan -var-file=envs/prod/terraform.tfvars
terraform apply -var-file=envs/prod/terraform.tfvars
```

### Destroy Staging

```bash
terraform workspace select staging
terraform destroy -var-file=envs/staging/terraform.tfvars
```
