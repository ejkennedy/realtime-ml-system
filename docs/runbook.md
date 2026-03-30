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

# 4. Generate synthetic feature data and materialise to Redis
make seed-data
make feast-apply
make feast-materialize

# 5. Train initial model
make train
# → models/registry/fraud_detector/latest/model.onnx

# 6. Verify the serving layer is up
make smoke-test
```

### Daily Development Workflow

```bash
make up            # start all services
make produce       # start synthetic transaction producer (optional)
make flink-job     # submit Flink feature pipeline (optional)
make serve         # start Ray Serve locally (if not using docker)
make smoke-test    # sanity check
make down          # stop everything
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
6. Registered in MLflow under `fraud-detector` model name with `staging` alias
7. Version manager picks up the `staging` alias and loads into `ShadowScorer`

## Model Promotion and Rollback

### Shadow Deployment Flow

When a new model is trained:
1. It is registered in MLflow with `staging` alias
2. `VersionManager` polls MLflow every 60s and detects the new staging model
3. The model is loaded into the `ShadowScorer` deployment (never affects primary)
4. After 30 minutes, shadow comparison metrics are evaluated
5. If new model AUC >= baseline - 1%: auto-promote to `production`
6. If regression detected: hold promotion, log warning, alert

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
open http://localhost:5000

# Or via CLI
uv run python -c "
import mlflow
client = mlflow.MlflowClient(tracking_uri='http://localhost:5000')
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
   - If high: pool is exhausted → reduce `max_concurrent_queries` or increase pool size
2. Check Redis latency (`fraud_redis_duration_seconds`)
   - If high: Redis is slow → check `redis-cli --latency-history`
3. Check ONNX inference time (`fraud_onnx_duration_seconds`)
   - If high: model too large → consider pruning or quantisation
4. Check GC pressure:
   ```bash
   docker stats ray-head  # watch memory usage patterns
   ```
5. Run load test to profile at different concurrency levels:
   ```bash
   make load-test
   # Check reports/load_test_*.html for latency distribution plots
   ```

## Load Testing

### Standard Load Test (10k req/s, 5 min)

```bash
make load-test
```

Generates `reports/load_test_TIMESTAMP.html` with:
- p50/p95/p99/p99.9 latency
- Requests per second
- Error rate
- Latency distribution histogram + CDF plot

### Load Test with Web UI

```bash
make load-test-ui
# Open http://localhost:8089 to control the test interactively
```

### Interpreting Load Test Results

**Healthy system** at 10k req/s should show:
- p50 < 15ms
- p95 < 50ms (SLA)
- p99 < 80ms
- Error rate < 0.1%

**Common failure modes:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| p95 spike at ~50ms | Pool exhaustion | Increase `ONNX_SESSION_POOL_SIZE` |
| p99 spikes every ~30s | GC pause | Already mitigated; check memory leak |
| Errors at high QPS | Ray actor overloaded | Add replicas (`ray.serve.run` with `num_replicas`) |
| High variance in latency | NUMA effects on multi-socket | Set `intra_op_num_threads=1` in ONNX session opts |

## Feature Store Operations

### Materialise Features (sync offline → online Redis)

```bash
make feast-materialize
# Materialises all feature views with data newer than last materialisation
```

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
