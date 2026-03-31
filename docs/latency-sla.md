# Latency SLA

## Definition

The architecture target is:

- p95 inference latency below `50 ms`

That target is meaningful for a tuned production deployment. It is not a
guarantee for a full local stack running on one laptop.

## How To Interpret The SLA

There are two separate questions:

1. Can the architecture plausibly support a sub-50ms p95 serving path?
2. Does the current local machine and benchmark setup demonstrate it?

Right now:

- the architecture is designed toward that target
- the local benchmark still misses it

So the SLA is still the design goal, but local benchmark results should be read
as optimization feedback, not proof of attainment.

## Current Local Reality

Recent local runs in `reports/` show a clear trend:

- older local runs were well above `200 ms` p95
- the typed fast path brought short-run verification down materially
- the best recent short verification run is still above `50 ms` p95

That means the local system has improved, but it has not closed the full gap.

## What Actually Moves Latency

The biggest practical latency drivers in this repo are:

### 1. Request-path overhead

- JSON decode and object creation
- timestamp parsing
- Python dict churn
- response serialization

The repo now uses `msgspec` decoding and `orjson` responses on the fast path to
reduce this overhead.

### 2. Ray / serving topology overhead

- router hop
- replica scheduling
- queueing before inference

For this reason, `make serve-perf` binds directly to `FraudScorer` when
shadowing is disabled.

### 3. ONNX runtime behavior

- session acquisition wait
- actual inference runtime
- thread oversubscription

The local perf defaults now use:

- one replica
- one ONNX session
- `ONNX_INTRA_OP_THREADS=1`
- `ONNX_INTER_OP_THREADS=1`

because the benchmark matrix showed extra concurrency hurt on the local machine.

### 4. Background work

- online-update queue writes
- version-manager polling
- shadow deployment work

These are useful operationally, but they are not part of the cleanest latency
benchmark path.

### 5. Environment pressure

- Ray object spilling / temp-dir pressure
- low free disk space
- other services running on the same machine

This has been an active issue in local runs and can distort p95 materially.

## What To Measure

The serving layer now exposes stage-level timings. Use them before drawing
conclusions from one p95 number.

Measure:

- end-to-end inference latency
- request parse latency
- feature prep latency
- session pool wait latency
- ONNX runtime latency
- response build latency

If p95 is high:

- high pool wait means queueing / too much concurrency
- high ONNX time means model/runtime work dominates
- high parse/prep time means Python/request format overhead dominates

## Recommended Local Workflow

Use this when comparing code changes:

```bash
make serve-perf
make load-test-local
```

Use this only for saturation testing:

```bash
make load-test
```

If you need ONNX evidence:

```bash
ONNX_PROFILE_ENABLED=true make serve-perf
make smoke-test
```

This writes ORT profile output under `reports/onnx_profiles/`.

## Quantized Model Note

The repo now attempts to export a quantized ONNX model, but the current
XGBoost-derived ONNX graph does not always support ONNX Runtime dynamic
quantization.

So:

- quantized export is attempted
- if unsupported, training skips it with a warning
- serving falls back to FP32 explicitly when the int8 model is unavailable

Quantization is therefore an experimental optimization path, not yet a stable
part of the SLA story.

## What Success Looks Like

For this repo, success should be evaluated in layers:

### Local development success

- functional end-to-end flow works
- `make smoke-test` passes
- local p95 improves release to release
- the benchmark matrix identifies better configurations reliably

### Production-style success

- p95 under `50 ms`
- low pool wait
- stable tail latency
- no saturation-induced failure rate

Those are related, but they are not the same claim.

## Recommended Language

The most accurate way to describe the current state is:

- sub-50ms p95 is the architecture target
- the local stack is useful for relative benchmarking
- the latest optimizations improved the local fast path materially
- further work is still required to reach the target reliably
