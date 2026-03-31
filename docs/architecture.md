# System Architecture

## Overview

This project implements a real-time fraud-scoring system with streaming feature
engineering, ONNX-based model serving, shadow deployment, model registry
integration, and drift-triggered retraining. The architectural target is
sub-50ms p95 inference on tuned infrastructure. Local laptop runs are useful for
functional validation and relative performance comparisons, but they should not
be treated as production-equivalent latency evidence.

## Data Flow

```text
Transaction Producer
  -> Redpanda (transactions-raw)
  -> Flink velocity features + Feast enrichment
  -> transactions-enriched
  -> Ray Serve
       -> FraudScorer [primary]
       -> FraudScorer [shadow, optional]
  -> transactions-scored / shadow-results
  -> MLflow + monitoring + drift detection
```

## Serving Paths

### Standard serving path

- HTTP requests enter Ray Serve.
- `FraudRouter` forwards to the primary scorer.
- If shadowing is enabled, the router also dispatches a fire-and-forget request to the shadow scorer.
- Shadow output is published asynchronously for comparison and promotion logic.

### Local perf path

- `make serve-perf` disables shadowing and version-manager polling.
- HTTP binds directly to `FraudScorer` with no router hop.
- Online-update Redis writes are disabled.
- Local perf defaults use one scorer replica, one ONNX session, and single-threaded ORT settings.

This direct path exists because the extra Ray hop and background MLOps work are
useful operationally, but they are not the lowest-latency serving path.

## Request Path

The current low-latency path does the following inside `FraudScorer`:

1. Decode request bytes with `msgspec` into a typed struct.
2. Reuse a pre-allocated `float32` numpy buffer for model features.
3. Fill features in place, preferring `hour_of_day`, `day_of_week`, and `timestamp_unix_ms` when present.
4. Acquire an ONNX Runtime session from a pre-warmed pool.
5. Run inference and return an `orjson` response.

Important design details:

- Typed request decoding removes a large amount of per-request Python dict churn.
- The ONNX pool records both pool-wait latency and ONNX runtime latency separately.
- Online updates are queued off the request path when enabled; they no longer perform synchronous Redis writes during inference.

## ONNX Runtime

The model runtime is centered around `OnnxSessionPool`:

- sessions are created at startup, never lazily on the request path
- sessions are pre-warmed with a dummy inference
- ORT thread counts are env-configurable
- profiling can be enabled with `ONNX_PROFILE_ENABLED=true`
- profile files are written to `reports/onnx_profiles/`

The current local perf defaults are intentionally conservative because the repo
benchmarks showed CPU oversubscription hurts both latency and throughput on a
laptop.

## Quantized Model Path

Training now attempts to emit both:

- `model.onnx`
- `model.int8.onnx`

The quantized path is optional. In practice, the current XGBoost-exported ONNX
graph does not always support ONNX Runtime dynamic quantization. When that
happens:

- training logs a warning and skips the int8 artifact
- `SERVE_USE_QUANTIZED_MODEL=true` falls back to FP32 with an explicit warning

So quantization is part of the architecture experimentation path, not yet a
guaranteed serving mode.

## Feature Freshness

The system intentionally mixes feature types with different freshness:

| Feature type | Source | Freshness | Notes |
|---|---|---|---|
| Velocity features | Flink state | sub-second | event-driven, stateful |
| Card / merchant aggregates | Feast + Redis | batch-materialized | simpler and cheaper to serve |
| Engineered ratios / calendar features | request path | immediate | computed inside scorer |

This hybrid setup is operationally realistic: the hottest features are streamed,
while slower-moving aggregates are served from Redis.

## Monitoring and Diagnostics

The serving layer now exposes stage-level timing that can be used to isolate
latency causes:

- request parse latency
- feature prep latency
- ONNX session pool wait latency
- ONNX runtime latency
- response build latency
- end-to-end inference latency

If p95 is high, the first question is no longer “is the model slow?” but:

- are we queueing on sessions?
- are we paying JSON / Python overhead?
- is ORT runtime dominating?
- are background system effects, like Ray spill pressure, distorting the run?

For local diagnostics, `make perf-breakdown` snapshots those stage summaries from
the live `FraudScorer` deployment into `reports/perf_breakdown_*.md`.

## Local Benchmark Reality

The repo contains multiple load-test summaries and benchmark matrices in
`reports/`. Those results show:

- the best local configuration so far is still `1 replica / 1 session`
- the direct typed fast path materially improved p95
- the project still does not reliably achieve `<50 ms p95` on a laptop

That means the codebase should present the local performance story as:

- useful for relative optimization work
- not proof of production SLA attainment

## Production Interpretation

The architecture target remains valid as a production goal:

- dedicated serving nodes
- controlled ORT thread placement
- reduced local-system noise
- no laptop disk-pressure issues
- realistic autoscaling and replica routing

But the current local measurements should be treated as a development baseline,
not a final performance claim.
