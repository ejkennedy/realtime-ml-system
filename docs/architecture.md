# System Architecture

## Overview

This system implements real-time fraud detection with a sub-50ms p95 inference SLA at 10,000 events per second. It demonstrates production-grade MLOps patterns: streaming feature engineering, ONNX model serving, shadow deployment, concept drift detection, and online learning.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                                     │
│                                                                             │
│  Transaction Producer → Redpanda (transactions-raw, 12 partitions)         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING (Flink)                           │
│                                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │  VelocityOperator   │    │  FeastEnrichmentOperator                 │   │
│  │  (keyed by card_id) │    │  (Redis pipeline, ~1ms)                  │   │
│  │                     │    │                                          │   │
│  │  - tx_count_1m      │    │  - card_risk_score                       │   │
│  │  - tx_count_5m      │    │  - merchant_fraud_rate_30d               │   │
│  │  - tx_count_1h      │    │  - card_avg_spend_30d                    │   │
│  │  - amount_avg_1h    │    │  - merchant_avg_amount                   │   │
│  │  - distinct_merch.. │    │                                          │   │
│  │  RocksDB state      │    │  Gracefully degrades on Redis failure    │   │
│  └─────────────────────┘    └──────────────────────────────────────────┘   │
│                                                                             │
│  Checkpointing: 10s interval, incremental RocksDB, exactly-once semantics  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    transactions-enriched topic
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER (Ray Serve)                          │
│                                                                             │
│                        ┌──────────────┐                                    │
│  Request ──────────────► FraudRouter  │                                    │
│                        └──────┬───────┘                                    │
│                               │                                            │
│              ┌────────────────┼────────────────────┐                       │
│              │ (awaited)      │                     │ (fire-and-forget)    │
│              ▼                │                     ▼                      │
│  ┌─────────────────────┐      │      ┌────────────────────────────┐        │
│  │  FraudScorer        │      │      │  ShadowScorer              │        │
│  │  [primary]          │      │      │  [candidate model]         │        │
│  │                     │      │      │                            │        │
│  │  ONNX session pool  │      │      │  asyncio.wait_for(100ms)   │        │
│  │  (4 pre-warmed)     │      │      │  → shadow-results topic    │        │
│  │  numpy pre-alloc    │      │      └────────────────────────────┘        │
│  │  GC disabled        │      │                                            │
│  └────────┬────────────┘      │                                            │
│           │                   │                                            │
│  Response returned to caller  │                                            │
└───────────┼───────────────────────────────────────────────────────────────-┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE LEARNING LAYER                                    │
│                                                                             │
│  ┌─────────────────────────────┐   ┌────────────────────────────────────┐  │
│  │  SGD Adapter                │   │  LinUCB Contextual Bandit           │  │
│  │                             │   │                                    │  │
│  │  partial_fit() on 100-item  │   │  Adapts decision threshold per:    │  │
│  │  micro-batches from Redis   │   │  (merchant_category, hour_bucket)  │  │
│  │  queue. Serialised to Redis │   │                                    │  │
│  │  after each batch.          │   │  Arms: [0.3, 0.4, 0.5, 0.6, 0.7]  │  │
│  │  All workers reload every   │   │  Reward: -2 for missed fraud       │  │
│  │  30s.                       │   │  Reward: -0.5 for false positive   │  │
│  └─────────────────────────────┘   └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & MLOPS                                       │
│                                                                             │
│  Evidently AI (every 15min)                                                 │
│    → DataDriftPreset: KS test per feature (threshold: 30% drift)           │
│    → TargetDriftPreset: PSI on fraud rate (threshold: 0.25)                │
│    → DataQualityPreset: null rates, outliers                                │
│                                                                             │
│  If drift detected:                                                         │
│    → RetrainingTrigger.dispatch() → K8s Job (prod) / subprocess (local)    │
│    → Slack/PagerDuty alert                                                  │
│    → New model → MLflow staging → shadow deploy → compare 30min            │
│    → Promote if AUC >= baseline - 1%                                        │
│                                                                             │
│  Prometheus scrapes:                                                        │
│    → fraud_inference_duration_seconds (histogram, 21 buckets around 50ms)  │
│    → fraud_feature_drift_share (gauge)                                      │
│    → fraud_requests_total (counter by status/path)                          │
│    → fraud_session_pool_wait_seconds (histogram)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture

### Redpanda over Kafka

Redpanda is a Kafka-compatible broker written in C++ (no JVM) that achieves lower latency and simpler operation for the streaming layer. In production at 10k events/sec with 12 partitions, it uses approximately 200MB RAM vs Kafka's 1-2GB. It is fully compatible with the Flink Kafka connector — no code changes required to switch to MSK in production.

### PyFlink with RocksDB State

Velocity features (transaction counts over 1m/5m/1h/24h windows) require maintaining per-card state across time. Options considered:

| Approach | Exactly-once | Fault tolerant | Complexity |
|----------|-------------|----------------|------------|
| Plain Kafka consumer + Redis | No (manual) | Requires custom logic | High |
| Kafka Streams | Yes | Yes | Medium (JVM only) |
| PyFlink + RocksDB | Yes | Yes (checkpoints) | Medium |
| Flink + Java | Yes | Yes | Low (Java expertise needed) |

PyFlink is chosen because it provides exactly-once semantics via checkpointing without requiring the application to manage offset commits and Redis writes atomically. RocksDB state backend with incremental checkpoints reduces checkpoint latency from ~2s (default heap-based) to ~200ms at steady state.

### ONNX Runtime over TensorFlow Serving / Triton

For an XGBoost model with 21 features:

| Runtime | p95 latency (local) | Memory | Cold start |
|---------|---------------------|--------|------------|
| Python XGBoost predict() | ~12ms | Low | ~50ms |
| ONNX Runtime (session pool) | ~5ms | Low | None (pre-warmed) |
| Triton Inference Server | ~8ms | High (CUDA optional) | ~200ms |
| TensorFlow Serving | ~20ms | High | ~500ms |

ONNX Runtime with a pre-warmed session pool achieves the best latency/memory tradeoff for CPU-based tree model inference. Triton is available as a secondary path (`services/serving/src/serving/triton/`) for when GPU acceleration is needed for deep learning models.

### Session Pool + Pre-allocated Numpy Array

The two patterns that matter most for p95 latency:

1. **Session pool**: ONNX Runtime sessions are not created on the request path. They are created at actor startup (`__init__`) and pooled. The pool size matches `max_concurrent_queries` to prevent internal Ray queueing from adding latency on top of pool wait time.

2. **Pre-allocated input array**: `self._input_buffer = np.zeros((1, 21), dtype=np.float32)` is allocated once at actor startup and reused every request. Without this, `np.array([...])` allocation at 10k req/s generates significant GC pressure, adding 5-15ms to p99 unpredictably.

### Feature Freshness Tradeoffs

| Feature type | Freshness | Source | Latency |
|--------------|-----------|--------|---------|
| Velocity (tx counts, amounts) | Sub-second (event-driven via Flink) | RocksDB → Redis | ~1ms |
| Card risk score | Daily batch (materialised from Iceberg) | Redis (Feast) | ~1ms |
| Merchant fraud rate | Daily batch | Redis (Feast) | ~1ms |
| Historical averages | 30-day rolling (nightly job) | Redis (Feast) | ~1ms |

The tradeoff: velocity features are highly fresh but require stateful Flink operators and RocksDB state, which adds operational complexity. Card/merchant features are stale (up to 24h) but trivially served from Redis. The model is trained with this mixed-freshness setup so it is robust to the feature lag.

## Latency Budget Breakdown

For a single inference request at p95:

| Component | Time | Notes |
|-----------|------|-------|
| Network (client → Ray Serve) | ~1ms | localhost |
| Ray Serve request parsing | ~1ms | FastAPI overhead |
| Feature preparation (dict → numpy) | ~0.5ms | Pre-allocated buffer |
| ONNX session acquisition (pool) | ~0.5ms | Non-contended at ≤4 concurrent |
| ONNX inference (XGBoost, 21 features) | ~5ms | Pre-warmed, graph-optimised |
| Response serialisation | ~0.5ms | JSONResponse |
| Network (Ray Serve → client) | ~1ms | |
| **Total p95** | **~10ms** | Well under 50ms SLA |

p99 adds 5-15ms for occasional GC pauses (mitigated by background GC thread). p999 spikes to 30-40ms under peak load due to Ray actor scheduling variance.

## Deployment Topology (Kubernetes)

```
Kubernetes Cluster
├── general node pool (m5.2xlarge × 3-10)
│   ├── flink-jobmanager (1 replica, 1 CPU, 2GB)
│   ├── flink-taskmanager (3 replicas, 2 CPU, 4GB each)
│   ├── redis (StatefulSet, r6g.large)
│   ├── mlflow (1 replica)
│   └── monitoring CronJob (drift check, every 15min)
│
└── serving node pool (c5.4xlarge × 4-40, HPA)
    └── ray-head + ray-workers (tainted: workload=serving)
        ├── FraudRouter (2 replicas, 0.5 CPU)
        ├── FraudScorer/primary (2-20 replicas, 1 CPU, HPA)
        └── FraudScorer/shadow (2 replicas, 1 CPU)
```

HPA scales `FraudScorer` replicas when CPU > 70% or memory > 80%, with a 60s scale-up cooldown and 300s scale-down cooldown to prevent oscillation.
