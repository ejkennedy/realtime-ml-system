"""
Prometheus latency tracker middleware.

Tracks p50/p95/p99 inference latency with fine-grained buckets around the 50ms SLA.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram

# Histogram buckets tuned for <50ms SLA profiling (ms values converted to seconds)
LATENCY_BUCKETS = [
    0.001, 0.002, 0.005,        # <5ms
    0.010, 0.015, 0.020,        # 10-20ms
    0.025, 0.030, 0.035,        # 25-35ms
    0.040, 0.045, 0.050,        # 40-50ms (SLA boundary)
    0.060, 0.075, 0.100,        # 60-100ms
    0.150, 0.200, 0.500, 1.0,   # degraded
]

inference_latency = Histogram(
    "fraud_inference_duration_seconds",
    "End-to-end inference latency (feature fetch + ONNX run)",
    buckets=LATENCY_BUCKETS,
    labelnames=["model_version", "path"],  # path: "primary" | "shadow"
)

redis_latency = Histogram(
    "fraud_redis_duration_seconds",
    "Redis feature fetch latency",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050],
)

onnx_latency = Histogram(
    "fraud_onnx_duration_seconds",
    "ONNX Runtime inference latency only",
    buckets=[0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050],
)

requests_total = Counter(
    "fraud_requests_total",
    "Total inference requests",
    labelnames=["status", "path"],
)

fraud_predictions = Counter(
    "fraud_predictions_total",
    "Fraud predictions (is_fraud=True)",
    labelnames=["model_version"],
)

fraud_rate_gauge = Gauge(
    "fraud_prediction_fraud_rate",
    "Rolling fraud prediction rate (1-minute EMA)",
)

pool_wait_latency = Histogram(
    "fraud_session_pool_wait_seconds",
    "Time waiting for an ONNX session from the pool",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.010, 0.020, 0.040],
)

shadow_timeout_counter = Counter(
    "fraud_shadow_timeouts_total",
    "Number of shadow inference timeouts (fire-and-forget path)",
)


@contextmanager
def track_inference(model_version: str, path: str = "primary") -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
        requests_total.labels(status="ok", path=path).inc()
    except Exception:
        requests_total.labels(status="error", path=path).inc()
        raise
    finally:
        elapsed = time.perf_counter() - start
        inference_latency.labels(model_version=model_version, path=path).observe(elapsed)
