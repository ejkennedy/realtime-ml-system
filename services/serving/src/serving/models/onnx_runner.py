"""
ONNX Runtime session pool — the critical path for sub-50ms p95 latency.

Key patterns:
1. Pre-warm all sessions at startup (never lazy-create on request path)
2. Pre-allocate numpy input array (avoid per-request GC pressure)
3. Disable Python GC inside the pool (background thread handles it instead)
4. Pool size == Ray Serve max_concurrent_queries to prevent queueing delays
"""

from __future__ import annotations

import contextlib
import gc
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
import structlog

from serving.middleware.latency_tracker import onnx_latency, pool_wait_latency

log = structlog.get_logger()

# Input feature names — must match training schema exactly
FEATURE_NAMES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category_encoded",
    "pos_type_encoded",
    "tx_count_1m",
    "tx_count_5m",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_1h",
    "amount_avg_1h",
    "amount_max_1h",
    "amount_sum_24h",
    "distinct_merchants_1h",
    "distinct_countries_1h",
    "card_risk_score",
    "merchant_fraud_rate_30d",
    "merchant_avg_amount",
    "card_avg_spend_30d",
    "amount_vs_avg_ratio",        # engineered: amount / card_avg_spend_30d
    "amount_vs_merchant_ratio",   # engineered: amount / merchant_avg_amount
]

NUM_FEATURES = len(FEATURE_NAMES)


class OnnxSessionPool:
    """
    Thread-safe pool of pre-warmed ONNX Runtime inference sessions.

    Each session uses env-configured ORT thread counts and sequential execution mode.
    The pool is populated at startup and returned to idle after each inference.
    """

    def __init__(self, model_path: str, pool_size: int = 4) -> None:
        self._model_path = Path(model_path)
        self._pool_size = pool_size
        self._pool: queue.Queue[ort.InferenceSession] = queue.Queue()
        self._sessions: list[ort.InferenceSession] = []
        self._input_name: str = ""
        self._output_name: str = ""
        self._profiling_enabled = os.environ.get("ONNX_PROFILE_ENABLED", "false") == "true"
        self._profile_dir = Path(os.environ.get("ONNX_PROFILE_DIR", "./reports/onnx_profiles"))
        self._session_counter = 0
        self._metrics_lock = threading.Lock()
        self._pool_wait_samples: deque[float] = deque(maxlen=5000)
        self._onnx_run_samples: deque[float] = deque(maxlen=5000)
        self._profile_artifacts: list[str] = []

        # Disable Python GC on this thread — managed by background thread below
        gc.disable()
        self._gc_thread = threading.Thread(target=self._gc_worker, daemon=True)
        self._gc_thread.start()

        self._build_pool()
        log.info(
            "ONNX session pool ready",
            model=str(self._model_path),
            pool_size=pool_size,
            input_name=self._input_name,
            profiling_enabled=self._profiling_enabled,
        )

    def _build_pool(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", "1"))
        opts.inter_op_num_threads = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_profiling = self._profiling_enabled
        if self._profiling_enabled:
            self._profile_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(self._pool_size):
            if self._profiling_enabled:
                opts.profile_file_prefix = str(
                    self._profile_dir / f"{self._model_path.stem}_session_{self._session_counter}"
                )
                self._session_counter += 1
            sess = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._sessions.append(sess)
            self._pool.put(sess)

        # Cache input/output names from first session
        probe = self._pool.get_nowait()
        self._input_name = probe.get_inputs()[0].name
        self._output_name = probe.get_outputs()[1].name  # probabilities output
        self._pool.put(probe)

        # Pre-warm: run one dummy inference per session to trigger JIT compilation
        self._prewarm()

    def _prewarm(self) -> None:
        dummy = np.zeros((1, NUM_FEATURES), dtype=np.float32)
        sessions = []
        for _ in range(self._pool_size):
            sess = self._pool.get()
            sess.run([self._output_name], {self._input_name: dummy})
            sessions.append(sess)
        for sess in sessions:
            self._pool.put(sess)
        log.info("ONNX sessions pre-warmed")

    @contextlib.contextmanager
    def session(self) -> Iterator[ort.InferenceSession]:
        wait_start = time.perf_counter()
        sess = self._pool.get(timeout=0.04)  # 40ms timeout → raises queue.Empty if pool exhausted
        wait_elapsed = time.perf_counter() - wait_start
        pool_wait_latency().observe(wait_elapsed)
        with self._metrics_lock:
            self._pool_wait_samples.append(wait_elapsed)
        try:
            yield sess
        finally:
            self._pool.put(sess)

    def predict_proba(self, features: np.ndarray) -> float:
        """Run inference and return fraud probability (class 1)."""
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)
        with self.session() as sess:
            run_start = time.perf_counter()
            result = sess.run([self._output_name], {self._input_name: features})
            run_elapsed = time.perf_counter() - run_start
            onnx_latency().observe(run_elapsed)
            with self._metrics_lock:
                self._onnx_run_samples.append(run_elapsed)
        # result[0] is shape (batch, 2) for binary classifier — return P(fraud)
        return float(result[0][0][1])

    def reload(self, new_model_path: str) -> None:
        """Hot-swap model — drains pool, rebuilds with new path, then refills."""
        log.info("Reloading ONNX model", new_path=new_model_path)
        self.finalize_profiles()
        self._model_path = Path(new_model_path)
        # Drain existing sessions
        drained = []
        while not self._pool.empty():
            try:
                drained.append(self._pool.get_nowait())
            except queue.Empty:
                break
        del drained  # release old sessions
        self._sessions = []
        self._build_pool()
        log.info("ONNX model hot-swapped", new_path=new_model_path)

    def get_perf_summary(self) -> dict[str, object]:
        with self._metrics_lock:
            pool_wait_ms = [sample * 1000 for sample in self._pool_wait_samples]
            onnx_run_ms = [sample * 1000 for sample in self._onnx_run_samples]
        return {
            "pool_size": self._pool_size,
            "profiling_enabled": self._profiling_enabled,
            "pool_wait_ms": _summarize_samples(pool_wait_ms),
            "onnx_run_ms": _summarize_samples(onnx_run_ms),
            "profile_artifacts": list(self._profile_artifacts),
        }

    def finalize_profiles(self) -> list[str]:
        if not self._profiling_enabled:
            return list(self._profile_artifacts)

        for sess in self._sessions:
            try:
                profile_path = sess.end_profiling()
            except Exception:
                continue
            if profile_path and profile_path not in self._profile_artifacts:
                self._profile_artifacts.append(profile_path)
        return list(self._profile_artifacts)

    @staticmethod
    def _gc_worker() -> None:
        """Background GC worker — prevents stop-the-world pauses on inference threads."""
        while True:
            time.sleep(30)
            gc.collect()


def _summarize_samples(samples_ms: list[float]) -> dict[str, float | int | None]:
    if not samples_ms:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}

    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": round(float(arr.max()), 2),
    }
