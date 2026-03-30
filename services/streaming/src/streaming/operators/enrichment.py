"""
Feast enrichment operator: fetches online features from Redis via batched pipeline.

Bypasses Feast's high-level get_online_features() to use Redis pipelines directly,
saving ~3-5ms per request at the cost of manual schema validation.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from pyflink.datastream import MapFunction, RuntimeContext

log = structlog.get_logger()


class FeastEnrichmentOperator(MapFunction):
    """
    Flink MapFunction that enriches each transaction with features from Redis.

    Uses a connection pool with pipeline batching to minimise Redis round-trips.
    Gracefully degrades to zero-values if Redis is unavailable (non-blocking inference).
    """

    def open(self, runtime_context: RuntimeContext) -> None:
        self._pool = redis.ConnectionPool(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            db=0,
            max_connections=20,
            socket_keepalive=True,
            socket_connect_timeout=1.0,
            socket_timeout=0.5,
        )
        self._redis = redis.Redis(connection_pool=self._pool, decode_responses=True)
        log.info("FeastEnrichmentOperator opened", host=os.environ.get("REDIS_HOST"))

    def close(self) -> None:
        self._pool.disconnect()

    def map(self, value: dict[str, Any]) -> dict[str, Any]:
        card_id = value["card_id"]
        merchant_id = value["merchant_id"]

        try:
            with self._redis.pipeline(transaction=False) as pipe:
                # Redis key format: feast:{entity_type}:{entity_id}
                pipe.hgetall(f"feast:card:{card_id}")
                pipe.hgetall(f"feast:merchant:{merchant_id}")
                card_features, merchant_features = pipe.execute()
        except redis.RedisError as e:
            log.warning("Redis enrichment failed, using defaults", error=str(e))
            card_features, merchant_features = {}, {}

        value["card_risk_score"] = float(card_features.get("risk_score", 0.0))
        value["card_avg_spend_30d"] = float(card_features.get("avg_spend_30d", 0.0))
        value["card_typical_countries"] = (
            card_features.get("typical_countries", "").split(",")
            if card_features.get("typical_countries")
            else []
        )
        value["merchant_fraud_rate_30d"] = float(
            merchant_features.get("fraud_rate_30d", 0.0)
        )
        value["merchant_avg_amount"] = float(
            merchant_features.get("avg_amount", 0.0)
        )
        return value
