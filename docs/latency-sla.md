# Latency SLA: Sub-50ms p95

## SLA Definition

The system must serve fraud predictions with p95 latency below 50ms under sustained load of 10,000 requests per second. This document explains the design choices that make this achievable and how to verify it.

## The Five Patterns

These five implementation patterns collectively determine whether the SLA is met. Missing any one of them will push p95 above 50ms under load.

---

### Pattern 1: Pre-Warmed ONNX Session Pool

**Problem**: ONNX Runtime sessions take 20-80ms to initialise (loading the model graph, allocating memory, running the first JIT compilation pass). If sessions are created lazily on the request path, cold starts blow the SLA.

**Solution**: `OnnxSessionPool._build_pool()` is called in `__init__`, before Ray Serve declares the actor ready. All sessions are created and pre-warmed with a dummy input before the first real request arrives.

```python
# In OnnxSessionPool.__init__
for _ in range(pool_size):
    sess = ort.InferenceSession(model_path, sess_options=opts, ...)
    self._pool.put(sess)
self._prewarm()  # run one dummy inference per session
```

**Why a pool and not one session per actor?** ONNX Runtime `InferenceSession.run()` acquires an internal lock. With a single session and multiple async request handlers, the second request blocks until the first completes, adding queuing latency equal to inference time. With `pool_size=4` sessions and `max_concurrent_queries=4`, there is no contention.

---

### Pattern 2: Pre-Allocated Numpy Input Buffer

**Problem**: At 10k req/s, `np.array([f1, f2, ..., f21])` is called 10,000 times per second. Each call allocates a new array, which triggers memory allocation and eventually garbage collection. Under GC pressure, p99 spikes occur every 1-5 seconds.

**Solution**: Allocate once at actor startup, fill in place per request:

```python
# In FraudScorer.__init__
self._input_buffer = np.zeros((1, NUM_FEATURES), dtype=np.float32)

# In FraudScorer._prepare_features (called per request)
buf = self._input_buffer   # same object every call
buf[0, 0] = amount
buf[0, 1] = hour_of_day
# ... fill remaining features
return buf
```

This approach is safe because Ray Serve processes requests sequentially within an actor (one at a time, since `max_concurrent_queries=4` matches the pool size — no concurrent execution within a single actor).

---

### Pattern 3: Redis Pipelining for Feature Lookups

**Problem**: Fetching card features + merchant features = 2 separate Redis round-trips ≈ 2-4ms per request. At 10k req/s, Redis becomes a bottleneck.

**Solution**: Use `redis-py`'s pipeline with `transaction=False` to batch both lookups into a single TCP round-trip:

```python
with self._redis.pipeline(transaction=False) as pipe:
    pipe.hgetall(f"feast:card:{card_id}")
    pipe.hgetall(f"feast:merchant:{merchant_id}")
    card_features, merchant_features = pipe.execute()
```

`transaction=False` means no MULTI/EXEC wrapping — just batching the commands into one network packet. Reduces Redis latency from ~3ms to ~1ms for this use case.

---

### Pattern 4: `max_concurrent_queries` = Pool Size

**Problem**: If Ray Serve allows more concurrent requests than the ONNX pool can handle (pool_size=4 but max_concurrent_queries=100), requests queue internally within the actor. The queue adds latency equal to inference_time × queue_depth before the request even touches the session pool.

**Solution**: Set them equal:

```python
@serve.deployment(
    max_concurrent_queries=4,   # equals ONNX pool size
    ...
)
```

When the pool is fully utilised, Ray Serve routes the next request to a different replica (or creates one via auto-scaling) rather than queueing it at the current actor. This distributes load more effectively and prevents queue-induced latency spikes.

---

### Pattern 5: Background GC Thread

**Problem**: Python's cyclic garbage collector (GC) performs stop-the-world pauses. Under high request rates, generation-0 GC triggers frequently (every ~700 allocations). A GC pause during an inference adds 5-20ms to that request's latency, appearing as p99 spikes.

**Solution**: Disable automatic GC on the inference thread and run collection on a background thread every 30 seconds:

```python
# In OnnxSessionPool.__init__
gc.disable()
self._gc_thread = threading.Thread(target=self._gc_worker, daemon=True)
self._gc_thread.start()

@staticmethod
def _gc_worker():
    while True:
        time.sleep(30)
        gc.collect()
```

The 30-second background collection handles long-lived object cleanup. The pre-allocated numpy buffer (Pattern 2) minimises short-lived allocations, so generation-0 collections are infrequent enough that the 30s interval is sufficient.

---

## Measuring the SLA

### Server-Side (Prometheus)

```promql
# p95 inference latency over last 5 minutes
histogram_quantile(0.95, rate(fraud_inference_duration_seconds_bucket[5m]))

# p99 latency
histogram_quantile(0.99, rate(fraud_inference_duration_seconds_bucket[5m]))

# Session pool wait time (should be near zero under normal load)
histogram_quantile(0.99, rate(fraud_session_pool_wait_seconds_bucket[5m]))
```

### Client-Side (Load Test)

```bash
make load-test
# Generates reports/load_test_TIMESTAMP.html
# Contains histogram and CDF plots of p50/p95/p99/p99.9
```

### Pre/Post Optimisation Comparison

To measure the impact of each optimisation, run the load test in the following configurations and compare results:

```bash
# Baseline: no optimisations (single session, no pre-alloc, default GC, pool=1)
ONNX_SESSION_POOL_SIZE=1 RAY_SERVE_MAX_CONCURRENT=100 make load-test

# With session pool only
ONNX_SESSION_POOL_SIZE=4 RAY_SERVE_MAX_CONCURRENT=4 make load-test

# With session pool + pre-allocated buffer (current implementation)
make load-test
```

Expected results:

| Configuration | p50 | p95 | p99 |
|--------------|-----|-----|-----|
| Baseline (no pool, no pre-alloc) | ~15ms | ~85ms | ~180ms |
| Session pool only | ~8ms | ~35ms | ~65ms |
| Full implementation | ~8ms | ~25ms | ~42ms |

---

## When the SLA Is at Risk

### Indicators to Watch

1. `fraud_session_pool_wait_seconds` p99 > 5ms → pool too small or max_concurrent too high
2. Memory growth in Ray actor → memory leak → GC pressure → increase GC frequency
3. Redis `INFO latency` showing `instantaneous_ops_per_sec` near capacity (>100k)
4. XGBoost model too large (>500 trees) → consider pruning to 200-300 trees
5. High CPU steal time in Kubernetes → pod on noisy neighbour node → use dedicated node pool

### Safe Tuning Ranges

| Parameter | Default | Min safe | Max safe | Effect |
|-----------|---------|----------|----------|--------|
| `ONNX_SESSION_POOL_SIZE` | 4 | 1 | 8 | Higher = more throughput, more RAM |
| `max_concurrent_queries` | 4 | = pool_size | = pool_size | Always keep equal to pool_size |
| `num_replicas` | 2 | 1 | 40 (HPA) | More replicas = more throughput |
| `intra_op_num_threads` | 2 | 1 | 4 | Higher may reduce latency on multi-core |
| GC interval (seconds) | 30 | 10 | 120 | Lower = less GC-related spikes |
