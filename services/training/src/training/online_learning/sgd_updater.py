"""
SGD online learning adapter.

Maintains a lightweight SGDClassifier that updates via partial_fit() on micro-batches
consumed from a Redis queue. The SGD model is a correction layer on top of the
base XGBoost ONNX model — it learns from recent production traffic.

Runs as a background thread within the Ray Serve deployment or as a standalone process.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import threading
import time
from typing import Optional

import numpy as np
import redis
import structlog
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger()

REDIS_MODEL_KEY = "online_model:sgd:current"
REDIS_SCALER_KEY = "online_model:sgd:scaler"
REDIS_QUEUE_KEY = "online_update_queue"
REDIS_METRICS_KEY = "online_model:sgd:metrics"
BATCH_SIZE = int(os.environ.get("SGD_BATCH_SIZE", 100))
MODEL_TTL_S = 86_400  # 24h — prevents stale model if updates stop


class SGDOnlineUpdater:
    """
    Consumes labelled examples from Redis and updates an SGD classifier.

    The SGD model is serialised to Redis after each micro-batch, making it
    available to all Ray Serve workers. Workers periodically reload it.

    Tradeoff: Different workers may hold stale SGD models for up to poll_interval_s.
    This is acceptable because SGD is a secondary correction, not the primary scorer.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._model: SGDClassifier = self._load_or_init_model()
        self._scaler: Optional[StandardScaler] = self._load_or_init_scaler()
        self._lock = threading.Lock()
        self._update_count = 0

    def _load_or_init_model(self) -> SGDClassifier:
        serialised = self._redis.get(REDIS_MODEL_KEY)
        if serialised:
            try:
                model = pickle.loads(serialised)
                log.info("SGD model loaded from Redis")
                return model
            except Exception as e:
                log.warning("Failed to load SGD model from Redis, initialising fresh", error=str(e))
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1,
            warm_start=True,
            random_state=42,
            class_weight={0: 1, 1: 50},   # mirrors XGBoost scale_pos_weight
        )

    def _load_or_init_scaler(self) -> StandardScaler:
        serialised = self._redis.get(REDIS_SCALER_KEY)
        if serialised:
            try:
                return pickle.loads(serialised)
            except Exception:
                pass
        return StandardScaler()

    def start_background_worker(self, poll_interval_s: float = 0.5) -> threading.Thread:
        """Start consuming from Redis queue and updating the model in background."""
        thread = threading.Thread(
            target=self._consume_loop,
            args=(poll_interval_s,),
            daemon=True,
            name="sgd-updater",
        )
        thread.start()
        log.info("SGD background updater started", batch_size=BATCH_SIZE)
        return thread

    def _consume_loop(self, poll_interval_s: float) -> None:
        buffer_X: list[list[float]] = []
        buffer_y: list[int] = []

        while True:
            # Drain up to BATCH_SIZE items from the Redis queue
            pipeline = self._redis.pipeline(transaction=False)
            for _ in range(BATCH_SIZE):
                pipeline.rpop(REDIS_QUEUE_KEY)
            items = pipeline.execute()

            for item in items:
                if item is None:
                    break
                try:
                    record = json.loads(item)
                    buffer_X.append(record["features"])
                    buffer_y.append(int(record["label"]))
                except (json.JSONDecodeError, KeyError):
                    continue

            if len(buffer_X) >= BATCH_SIZE:
                self._flush(buffer_X, buffer_y)
                buffer_X.clear()
                buffer_y.clear()

            time.sleep(poll_interval_s)

    def _flush(self, X_raw: list[list[float]], y: list[int]) -> None:
        X = np.array(X_raw, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        with self._lock:
            # Fit scaler on first batch, then transform_only for subsequent
            if self._update_count == 0:
                X_scaled = self._scaler.fit_transform(X)
            else:
                X_scaled = self._scaler.transform(X)

            self._model.partial_fit(X_scaled, y_arr, classes=[0, 1])
            self._update_count += 1

            # Persist to Redis — atomic set with TTL
            model_bytes = pickle.dumps(self._model)
            scaler_bytes = pickle.dumps(self._scaler)
            pipeline = self._redis.pipeline()
            pipeline.set(REDIS_MODEL_KEY, model_bytes, ex=MODEL_TTL_S)
            pipeline.set(REDIS_SCALER_KEY, scaler_bytes, ex=MODEL_TTL_S)
            pipeline.hset(REDIS_METRICS_KEY, mapping={
                "update_count": self._update_count,
                "last_batch_size": len(y),
                "last_update_ts": time.time(),
                "fraud_rate_batch": round(sum(y) / len(y), 4),
            })
            pipeline.execute()

        log.debug("SGD updated", batch_size=len(y), update_count=self._update_count)
        gc.collect(0)  # gen-0 only — quick and safe

    def predict_proba(self, features: np.ndarray) -> float:
        """Return P(fraud) from the SGD model. Returns 0.5 if model not trained yet."""
        with self._lock:
            if self._update_count == 0:
                return 0.5
            try:
                X_scaled = self._scaler.transform(features.reshape(1, -1))
                proba = self._model.predict_proba(X_scaled)[0][1]
                return float(proba)
            except Exception:
                return 0.5

    def get_metrics(self) -> dict:
        raw = self._redis.hgetall(REDIS_METRICS_KEY)
        return {k.decode(): v.decode() for k, v in raw.items()} if raw else {}
