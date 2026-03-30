"""
Traffic router with fire-and-forget shadow deployment.

Architecture:
  Incoming → Router → Primary (awaited, returns to caller)
                   ↘ Shadow  (fire-and-forget, never on hot path)

The shadow path has a hard 100ms timeout and never blocks the primary response.
Shadow results are published to Kafka for offline metric comparison.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

import structlog
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from serving.middleware.latency_tracker import shadow_timeout_counter

log = structlog.get_logger()

SHADOW_RATIO = float(os.environ.get("SHADOW_RATIO", 1.0))
SHADOW_TIMEOUT_S = float(os.environ.get("SHADOW_TIMEOUT_S", 0.1))
ROUTER_REPLICAS = int(os.environ.get("SERVE_ROUTER_REPLICAS", 2))
ROUTER_MAX_ONGOING_REQUESTS = int(os.environ.get("SERVE_ROUTER_MAX_ONGOING_REQUESTS", 100))


@serve.deployment(
    num_replicas=ROUTER_REPLICAS,
    max_ongoing_requests=ROUTER_MAX_ONGOING_REQUESTS,
    ray_actor_options={"num_cpus": 0.5},
)
class FraudRouter:
    """
    Routes each request to the primary scorer and optionally shadows to a candidate.
    Shadow path is fully async and never delays the primary response.
    """

    def __init__(self, primary_handle, shadow_handle=None) -> None:
        self._primary = primary_handle
        self._shadow = shadow_handle
        self._shadow_enabled = shadow_handle is not None and os.environ.get("SHADOW_ENABLED", "true") == "true"

        # Lazy Kafka producer — initialised on first shadow write
        self._kafka_producer = None

    async def __call__(self, request: Request) -> JSONResponse:
        payload = await request.json()

        # Primary path — always awaited
        primary_ref = self._primary.score.remote(payload)

        # Shadow path — fire-and-forget
        if self._shadow_enabled and random.random() < SHADOW_RATIO:
            shadow_ref = self._shadow.score.remote(payload)
            asyncio.create_task(self._handle_shadow(shadow_ref, payload))

        result = await primary_ref
        return JSONResponse(result)

    async def _handle_shadow(self, shadow_ref, payload: dict) -> None:
        """
        Collect shadow result and publish to Kafka. Never raises — exceptions are swallowed.
        Hard timeout prevents unbounded background task accumulation.
        """
        try:
            shadow_result = await asyncio.wait_for(shadow_ref, timeout=SHADOW_TIMEOUT_S)
            await self._publish_shadow_result(payload, shadow_result)
        except asyncio.TimeoutError:
            shadow_timeout_counter().inc()
        except Exception as e:
            log.debug("Shadow handler error", error=str(e))

    async def _publish_shadow_result(self, payload: dict, result: dict) -> None:
        """Publish shadow comparison record to Kafka (lazy producer init)."""
        if self._kafka_producer is None:
            try:
                from confluent_kafka import Producer
                self._kafka_producer = Producer(
                    {"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")}
                )
            except Exception:
                return

        record = json.dumps({
            "event_id": payload.get("event_id", ""),
            "shadow_score": result.get("fraud_score"),
            "shadow_is_fraud": result.get("is_fraud"),
            "shadow_model_version": result.get("model_version"),
            "timestamp": time.time(),
        })
        self._kafka_producer.produce(
            "shadow-results",
            value=record.encode(),
        )
        self._kafka_producer.poll(0)  # non-blocking flush

    def update_shadow(self, new_shadow_handle) -> None:
        """Hot-swap the shadow deployment handle."""
        self._shadow = new_shadow_handle
        self._shadow_enabled = new_shadow_handle is not None
        log.info("Shadow deployment updated", enabled=self._shadow_enabled)
