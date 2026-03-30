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
from pathlib import Path

import numpy as np
from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

from load_testing.transaction_gen import TransactionGenerator
from load_testing.latency_report import plot_distribution, write_markdown_summary

RNG = np.random.default_rng()
_gen = TransactionGenerator()
LATENCY_FAIL_THRESHOLD_MS = float(os.environ.get("LOAD_TEST_LATENCY_FAIL_THRESHOLD_MS", "0"))
LOAD_TEST_STAGE1_DURATION_S = int(os.environ.get("LOAD_TEST_STAGE1_DURATION_S", 60))
LOAD_TEST_STAGE1_USERS = int(os.environ.get("LOAD_TEST_STAGE1_USERS", 50))
LOAD_TEST_STAGE1_SPAWN_RATE = int(os.environ.get("LOAD_TEST_STAGE1_SPAWN_RATE", 5))
LOAD_TEST_STAGE2_DURATION_S = int(os.environ.get("LOAD_TEST_STAGE2_DURATION_S", 300))
LOAD_TEST_STAGE2_USERS = int(os.environ.get("LOAD_TEST_STAGE2_USERS", 200))
LOAD_TEST_STAGE2_SPAWN_RATE = int(os.environ.get("LOAD_TEST_STAGE2_SPAWN_RATE", 20))
LOAD_TEST_STAGE3_DURATION_S = int(os.environ.get("LOAD_TEST_STAGE3_DURATION_S", 360))
LOAD_TEST_STAGE3_USERS = int(os.environ.get("LOAD_TEST_STAGE3_USERS", 0))
LOAD_TEST_STAGE3_SPAWN_RATE = int(os.environ.get("LOAD_TEST_STAGE3_SPAWN_RATE", 50))


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
            "/",
            json=payload,
            catch_response=True,
            name="/",
        ) as response:
            if response.status_code == 200:
                response.json()
                latency_ms = response.elapsed.total_seconds() * 1000
                if LATENCY_FAIL_THRESHOLD_MS > 0 and latency_ms > LATENCY_FAIL_THRESHOLD_MS:
                    response.failure(
                        f"Latency {latency_ms:.1f}ms exceeds {LATENCY_FAIL_THRESHOLD_MS:.0f}ms threshold"
                    )
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


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        return

    reports_dir = Path("./reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    plot_path = reports_dir / f"latency_locust_{timestamp}.png"
    summary_path = reports_dir / f"load_test_summary_{timestamp}.md"

    plot_distribution(label="locust", output_path=str(plot_path))

    total = environment.stats.total
    write_markdown_summary(
        str(summary_path),
        label="locust",
        total_requests=total.num_requests,
        total_failures=total.num_failures,
        avg_rps=total.total_rps,
        sla_ms=50.0,
    )


# ── Custom load shape: ramp to 10k/sec ────────────────────────────────────────

from locust import LoadTestShape


class SteadyStateShape(LoadTestShape):
    """
    Phase 1 (0-60s): ramp from 10 to 200 users
    Phase 2 (60-300s): hold at 200 users (target ~10k req/s with wait_time≈0)
    Phase 3 (300-360s): ramp down for graceful shutdown
    """
    stages = [
        {
            "duration": LOAD_TEST_STAGE1_DURATION_S,
            "users": LOAD_TEST_STAGE1_USERS,
            "spawn_rate": LOAD_TEST_STAGE1_SPAWN_RATE,
        },
        {
            "duration": LOAD_TEST_STAGE2_DURATION_S,
            "users": LOAD_TEST_STAGE2_USERS,
            "spawn_rate": LOAD_TEST_STAGE2_SPAWN_RATE,
        },
        {
            "duration": LOAD_TEST_STAGE3_DURATION_S,
            "users": LOAD_TEST_STAGE3_USERS,
            "spawn_rate": LOAD_TEST_STAGE3_SPAWN_RATE,
        },
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
