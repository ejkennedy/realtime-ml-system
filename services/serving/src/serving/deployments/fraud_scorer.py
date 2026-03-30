"""
Primary fraud scoring deployment.

Handles feature preparation, ONNX inference, and optional online learning update.
max_concurrent_queries must equal OnnxSessionPool size to prevent queue buildup.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import redis
import structlog
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from serving.middleware.latency_tracker import (
    fraud_rate_gauge,
    fraud_predictions,
    track_inference,
)
from serving.models.onnx_runner import FEATURE_NAMES, NUM_FEATURES, OnnxSessionPool

log = structlog.get_logger()

POOL_SIZE = int(os.environ.get("ONNX_SESSION_POOL_SIZE", 4))
SCORER_REPLICAS = int(os.environ.get("SERVE_SCORER_REPLICAS", 2))
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/models/fraud_detector/latest/model.onnx")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.5))

MERCHANT_CATEGORY_MAP = {
    "retail": 0, "food": 1, "travel": 2, "entertainment": 3,
    "online": 4, "atm": 5, "utility": 6, "healthcare": 7, "other": 8,
}
POS_TYPE_MAP = {"chip": 0, "swipe": 1, "contactless": 2, "online": 3}


@serve.deployment(
    num_replicas=SCORER_REPLICAS,
    max_ongoing_requests=POOL_SIZE,   # matches pool size — no internal queue buildup
    ray_actor_options={"num_cpus": 1.0},
    health_check_period_s=10,
    health_check_timeout_s=5,
)
class FraudScorer:
    def __init__(self) -> None:
        self._pool = OnnxSessionPool(MODEL_PATH, pool_size=POOL_SIZE)
        self._model_version = self._get_model_version()

        # Redis for online learning state
        self._redis = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            socket_timeout=0.01,   # 10ms timeout — never block inference on Redis
        )

        # Pre-allocate input buffer — reused per request, avoids GC pressure
        self._input_buffer = np.zeros((1, NUM_FEATURES), dtype=np.float32)

        # EMA state for fraud rate gauge
        self._ema_alpha = 0.1
        self._ema_rate = 0.0

        log.info("FraudScorer ready", model_version=self._model_version)

    async def __call__(self, request: Request) -> JSONResponse:
        payload = await request.json()
        return await self.score(payload)

    async def score(self, payload: dict) -> dict:
        with track_inference(self._model_version, path="primary"):
            t0 = time.perf_counter()

            features = self._prepare_features(payload)
            fraud_score = self._pool.predict_proba(features)
            is_fraud = fraud_score >= FRAUD_THRESHOLD

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Update EMA fraud rate gauge
            self._ema_rate = self._ema_alpha * float(is_fraud) + (1 - self._ema_alpha) * self._ema_rate
            fraud_rate_gauge().set(self._ema_rate)

            if is_fraud:
                fraud_predictions().labels(model_version=self._model_version).inc()

            result = {
                "event_id": payload.get("event_id", ""),
                "fraud_score": round(fraud_score, 6),
                "is_fraud": is_fraud,
                "model_version": self._model_version,
                "inference_latency_ms": round(elapsed_ms, 2),
            }

            # Non-blocking online learning update (fire and forget via Redis queue)
            self._queue_online_update(payload, is_fraud)

            return result

    def _prepare_features(self, payload: dict) -> np.ndarray:
        """Fill the pre-allocated input buffer with feature values."""
        buf = self._input_buffer  # reuse allocation
        v = payload
        vel = v.get("velocity", {})

        amount = float(v.get("amount", 0))
        ts = v.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)

        card_avg = float(v.get("card_avg_spend_30d", 1)) or 1.0
        merchant_avg = float(v.get("merchant_avg_amount", 1)) or 1.0

        buf[0, 0] = amount
        buf[0, 1] = dt.hour
        buf[0, 2] = dt.weekday()
        buf[0, 3] = MERCHANT_CATEGORY_MAP.get(v.get("merchant_category", "other"), 8)
        buf[0, 4] = POS_TYPE_MAP.get(v.get("pos_type", "chip"), 0)
        buf[0, 5] = float(vel.get("tx_count_1m", 0))
        buf[0, 6] = float(vel.get("tx_count_5m", 0))
        buf[0, 7] = float(vel.get("tx_count_1h", 0))
        buf[0, 8] = float(vel.get("tx_count_24h", 0))
        buf[0, 9] = float(vel.get("amount_sum_1h", 0))
        buf[0, 10] = float(vel.get("amount_avg_1h", 0))
        buf[0, 11] = float(vel.get("amount_max_1h", 0))
        buf[0, 12] = float(vel.get("amount_sum_24h", 0))
        buf[0, 13] = float(vel.get("distinct_merchants_1h", 0))
        buf[0, 14] = float(vel.get("distinct_countries_1h", 0))
        buf[0, 15] = float(v.get("card_risk_score", 0))
        buf[0, 16] = float(v.get("merchant_fraud_rate_30d", 0))
        buf[0, 17] = float(v.get("merchant_avg_amount", 0))
        buf[0, 18] = float(v.get("card_avg_spend_30d", 0))
        buf[0, 19] = amount / card_avg
        buf[0, 20] = amount / merchant_avg

        return buf

    def _queue_online_update(self, payload: dict, label: bool) -> None:
        """Push a lightweight update record to Redis for the online learning worker."""
        try:
            import json
            self._redis.lpush(
                "online_update_queue",
                json.dumps({
                    "features": self._input_buffer[0].tolist(),
                    "label": int(label),
                    "timestamp": time.time(),
                }),
            )
            # Cap queue length — older examples are less relevant
            self._redis.ltrim("online_update_queue", 0, 9999)
        except Exception:
            pass  # never fail inference due to online learning queue issues

    def _get_model_version(self) -> str:
        try:
            import mlflow
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(
                os.environ.get("MLFLOW_MODEL_NAME", "fraud-detector"), "production"
            )
            return mv.version
        except Exception:
            return "local"

    def check_health(self) -> bool:
        return self._pool is not None
