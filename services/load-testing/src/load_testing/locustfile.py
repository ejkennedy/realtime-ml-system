"""
Locust load test: 10k events/sec fraud inference endpoint.

Run:
    locust -f locustfile.py --host http://localhost:8000 \
           --users 200 --spawn-rate 20 --run-time 5m \
           --headless --html report.html

Profile at different concurrency levels to find the p95 < 50ms operating point.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

from load_testing.transaction_gen import TransactionGenerator

RNG = np.random.default_rng()
_gen = TransactionGenerator()


class FraudScorerUser(HttpUser):
    """
    Simulates a high-throughput client sending transactions for fraud scoring.

    wait_time=between(0, 0) means maximum throughput.
    """
    wait_time = between(0, 0.001)

    def on_start(self) -> None:
        # Pre-generate a batch to avoid generation overhead in the hot loop
        self._batch = _gen.generate_batch(1000)
        self._idx = 0

    @task
    def score_transaction(self) -> None:
        payload = self._batch[self._idx % len(self._batch)]
        self._idx += 1

        with self.client.post(
            "/score",
            json=payload,
            catch_response=True,
            name="/score",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                latency_ms = response.elapsed.total_seconds() * 1000
                if latency_ms > 200:
                    response.failure(f"Latency {latency_ms:.1f}ms exceeds 200ms threshold")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# ── Prometheus metrics export ─────────────────────────────────────────────────

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if not isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        from prometheus_client import start_http_server
        start_http_server(9646)


@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length, exception, **kwargs
):
    from load_testing.latency_report import record_latency
    if exception is None:
        record_latency(response_time)


# ── Custom load shape: ramp to 10k/sec ────────────────────────────────────────

from locust import LoadTestShape


class SteadyStateShape(LoadTestShape):
    """
    Phase 1 (0-60s): ramp from 10 to 200 users
    Phase 2 (60-300s): hold at 200 users (target ~10k req/s with wait_time≈0)
    Phase 3 (300-360s): ramp down for graceful shutdown
    """
    stages = [
        {"duration": 60, "users": 50, "spawn_rate": 5},
        {"duration": 300, "users": 200, "spawn_rate": 20},
        {"duration": 360, "users": 0, "spawn_rate": 50},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
