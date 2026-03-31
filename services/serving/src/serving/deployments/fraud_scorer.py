"""
Primary fraud scoring deployment.

Handles feature preparation, ONNX inference, and optional online learning update.
max_concurrent_queries must equal OnnxSessionPool size to prevent queue buildup.
"""

from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone

import msgspec
import numpy as np
import redis
import structlog
from ray import serve
from starlette.requests import Request

from serving.middleware.latency_tracker import (
    feature_prep_latency,
    fraud_rate_gauge,
    fraud_predictions,
    observe_latency,
    online_update_dropped,
    online_update_enqueued,
    request_parse_latency,
    response_build_latency,
    track_inference,
)
from serving.models.onnx_runner import NUM_FEATURES, OnnxSessionPool
from serving.responses import ORJSONResponse
from serving.schemas import (
    ScoreResponse,
    TransactionRequest,
    decode_transaction_request,
    score_response_to_dict,
)

log = structlog.get_logger()

POOL_SIZE = int(os.environ.get("ONNX_SESSION_POOL_SIZE", 4))
SCORER_REPLICAS = int(os.environ.get("SERVE_SCORER_REPLICAS", 2))
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/models/fraud_detector/latest/model.onnx")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.5))
ONLINE_UPDATES_ENABLED = os.environ.get("ONLINE_UPDATES_ENABLED", "true") == "true"
ONLINE_UPDATE_SAMPLE_RATE = float(os.environ.get("ONLINE_UPDATE_SAMPLE_RATE", "1.0"))
ONLINE_UPDATE_QUEUE_MAXSIZE = int(os.environ.get("ONLINE_UPDATE_QUEUE_MAXSIZE", "2048"))
ONLINE_UPDATE_BATCH_SIZE = int(os.environ.get("ONLINE_UPDATE_BATCH_SIZE", "64"))
ONLINE_UPDATE_FLUSH_INTERVAL_S = float(os.environ.get("ONLINE_UPDATE_FLUSH_INTERVAL_S", "0.25"))

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
        self._metrics_lock = threading.Lock()
        self._request_parse_samples_ms: deque[float] = deque(maxlen=5000)
        self._feature_prep_samples_ms: deque[float] = deque(maxlen=5000)
        self._response_build_samples_ms: deque[float] = deque(maxlen=5000)
        self._score_total_samples_ms: deque[float] = deque(maxlen=5000)

        self._online_updates_enabled = ONLINE_UPDATES_ENABLED
        self._online_update_sample_rate = ONLINE_UPDATE_SAMPLE_RATE
        self._online_update_queue: queue.Queue[tuple[np.ndarray, int, float]] | None = None
        self._redis: redis.Redis | None = None
        if self._online_updates_enabled:
            self._online_update_queue = queue.Queue(maxsize=ONLINE_UPDATE_QUEUE_MAXSIZE)
            self._redis = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                socket_timeout=0.05,
            )
            self._online_update_thread = threading.Thread(
                target=self._online_update_worker,
                daemon=True,
                name="fraud-online-update-worker",
            )
            self._online_update_thread.start()

        # Pre-allocate input buffer — reused per request, avoids GC pressure
        self._input_buffer = np.zeros((1, NUM_FEATURES), dtype=np.float32)

        # EMA state for fraud rate gauge
        self._ema_alpha = 0.1
        self._ema_rate = 0.0

        log.info("FraudScorer ready", model_version=self._model_version)

    async def __call__(self, request: Request) -> ORJSONResponse:
        try:
            parse_start = time.perf_counter()
            with observe_latency(request_parse_latency(), "primary"):
                payload = decode_transaction_request(await request.body())
            self._record_stage_sample(self._request_parse_samples_ms, time.perf_counter() - parse_start)
        except msgspec.DecodeError as exc:
            return ORJSONResponse({"error": f"invalid request payload: {exc}"}, status_code=400)

        response = await self.score(payload)
        build_start = time.perf_counter()
        with observe_latency(response_build_latency(), "primary_http"):
            body = score_response_to_dict(response)
            http_response = ORJSONResponse(body)
        self._record_stage_sample(self._response_build_samples_ms, time.perf_counter() - build_start)
        return http_response

    async def score(self, payload: TransactionRequest) -> ScoreResponse:
        with track_inference(self._model_version, path="primary"):
            t0 = time.perf_counter()

            feature_start = time.perf_counter()
            with observe_latency(feature_prep_latency(), "primary"):
                features = self._prepare_features(payload)
            self._record_stage_sample(self._feature_prep_samples_ms, time.perf_counter() - feature_start)
            fraud_score = self._pool.predict_proba(features)
            is_fraud = fraud_score >= FRAUD_THRESHOLD

            self._enqueue_online_update(is_fraud)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record_stage_sample(self._score_total_samples_ms, elapsed_ms / 1000)

            self._ema_rate = self._ema_alpha * float(is_fraud) + (1 - self._ema_alpha) * self._ema_rate
            fraud_rate_gauge().set(self._ema_rate)

            if is_fraud:
                fraud_predictions().labels(model_version=self._model_version).inc()

            return ScoreResponse(
                event_id=payload.event_id,
                fraud_score=round(fraud_score, 6),
                is_fraud=is_fraud,
                model_version=self._model_version,
                inference_latency_ms=round(elapsed_ms, 2),
            )

    def get_perf_snapshot(self, finalize_profiles: bool = False) -> dict[str, object]:
        profile_artifacts = self._pool.finalize_profiles() if finalize_profiles else []
        onnx_summary = self._pool.get_perf_summary()
        return {
            "model_version": self._model_version,
            "online_updates_enabled": self._online_updates_enabled,
            "stage_latency_ms": {
                "request_parse": self._summarize_stage_samples(self._request_parse_samples_ms),
                "feature_prep": self._summarize_stage_samples(self._feature_prep_samples_ms),
                "response_build": self._summarize_stage_samples(self._response_build_samples_ms),
                "score_total": self._summarize_stage_samples(self._score_total_samples_ms),
            },
            "onnx": {
                **onnx_summary,
                "profile_artifacts": profile_artifacts or onnx_summary.get("profile_artifacts", []),
            },
        }

    def _prepare_features(self, payload: TransactionRequest) -> np.ndarray:
        """Fill the pre-allocated input buffer with feature values."""
        buf = self._input_buffer  # reuse allocation
        vel = payload.velocity

        amount = float(payload.amount)
        hour_of_day, day_of_week = self._extract_calendar_features(payload)

        card_avg = float(payload.card_avg_spend_30d) or 1.0
        merchant_avg = float(payload.merchant_avg_amount) or 1.0

        buf[0, 0] = amount
        buf[0, 1] = hour_of_day
        buf[0, 2] = day_of_week
        buf[0, 3] = MERCHANT_CATEGORY_MAP.get(payload.merchant_category, 8)
        buf[0, 4] = POS_TYPE_MAP.get(payload.pos_type, 0)
        buf[0, 5] = self._feature_value(payload, vel, "tx_count_1m")
        buf[0, 6] = self._feature_value(payload, vel, "tx_count_5m")
        buf[0, 7] = self._feature_value(payload, vel, "tx_count_1h")
        buf[0, 8] = self._feature_value(payload, vel, "tx_count_24h")
        buf[0, 9] = self._feature_value(payload, vel, "amount_sum_1h")
        buf[0, 10] = self._feature_value(payload, vel, "amount_avg_1h")
        buf[0, 11] = self._feature_value(payload, vel, "amount_max_1h")
        buf[0, 12] = self._feature_value(payload, vel, "amount_sum_24h")
        buf[0, 13] = self._feature_value(payload, vel, "distinct_merchants_1h")
        buf[0, 14] = self._feature_value(payload, vel, "distinct_countries_1h")
        buf[0, 15] = float(payload.card_risk_score)
        buf[0, 16] = float(payload.merchant_fraud_rate_30d)
        buf[0, 17] = float(payload.merchant_avg_amount)
        buf[0, 18] = float(payload.card_avg_spend_30d)
        buf[0, 19] = amount / card_avg
        buf[0, 20] = amount / merchant_avg

        return buf

    @staticmethod
    def _feature_value(payload: TransactionRequest, velocity: object, key: str) -> float:
        if velocity is not None:
            value = getattr(velocity, key)
            if value:
                return float(value)
        return float(getattr(payload, f"velocity_{key}"))

    @staticmethod
    def _extract_calendar_features(payload: TransactionRequest) -> tuple[int, int]:
        if payload.hour_of_day is not None and payload.day_of_week is not None:
            return int(payload.hour_of_day), int(payload.day_of_week)

        unix_ms = payload.timestamp_unix_ms
        if unix_ms is None:
            unix_ms = payload.timestamp_epoch_ms
        if unix_ms is not None:
            dt = datetime.fromtimestamp(float(unix_ms) / 1000, tz=timezone.utc)
            return dt.hour, dt.weekday()

        ts = payload.timestamp or ""
        try:
            dt = datetime.fromisoformat(ts)
            return dt.hour, dt.weekday()
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)
            return dt.hour, dt.weekday()

    def _enqueue_online_update(self, label: bool) -> None:
        if not self._online_updates_enabled or self._online_update_queue is None:
            return
        if self._online_update_sample_rate < 1.0 and random.random() > self._online_update_sample_rate:
            online_update_dropped().labels(reason="sampled_out").inc()
            return
        try:
            self._online_update_queue.put_nowait(
                (
                    self._input_buffer[0].copy(),
                    int(label),
                    time.time(),
                )
            )
            online_update_enqueued().inc()
        except queue.Full:
            online_update_dropped().labels(reason="queue_full").inc()

    def _online_update_worker(self) -> None:
        assert self._online_update_queue is not None
        assert self._redis is not None
        while True:
            batch: list[tuple[np.ndarray, int, float]] = []
            try:
                item = self._online_update_queue.get(timeout=ONLINE_UPDATE_FLUSH_INTERVAL_S)
                batch.append(item)
            except queue.Empty:
                continue

            while len(batch) < ONLINE_UPDATE_BATCH_SIZE:
                try:
                    batch.append(self._online_update_queue.get_nowait())
                except queue.Empty:
                    break

            try:
                pipe = self._redis.pipeline(transaction=False)
                for features, label, ts in batch:
                    pipe.lpush(
                        "online_update_queue",
                        json.dumps(
                            {
                                "features": features.tolist(),
                                "label": label,
                                "timestamp": ts,
                            }
                        ),
                    )
                pipe.ltrim("online_update_queue", 0, 9999)
                pipe.execute()
            except Exception:
                online_update_dropped().labels(reason="redis_error").inc()

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

    def _record_stage_sample(self, samples: deque[float], elapsed_s: float) -> None:
        with self._metrics_lock:
            samples.append(elapsed_s * 1000)

    def _summarize_stage_samples(self, samples: deque[float]) -> dict[str, float | int | None]:
        with self._metrics_lock:
            values = np.asarray(list(samples), dtype=np.float64)
        if values.size == 0:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
        return {
            "count": int(values.size),
            "mean": round(float(values.mean()), 2),
            "p50": round(float(np.percentile(values, 50)), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
            "p99": round(float(np.percentile(values, 99)), 2),
            "max": round(float(values.max()), 2),
        }

    def check_health(self) -> bool:
        return self._pool is not None
