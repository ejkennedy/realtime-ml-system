"""
Model version manager: polls MLflow for new staging models and coordinates rollback.

Flow:
  1. Poll MLflow every 60s for new model in "staging" alias
  2. If found: load into ShadowScorer deployment
  3. After shadow_promotion_window_s: compare metrics, promote if >= baseline
  4. Rollback: revert production alias + hot-swap OnnxSessionPool
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import mlflow
import mlflow.pyfunc
import structlog

log = structlog.get_logger()

MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "fraud-detector")
SHADOW_PROMOTION_WINDOW_S = int(os.environ.get("SHADOW_PROMOTION_WINDOW_S", 1800))  # 30 min
METRIC_TOLERANCE = 0.01  # allow 1% regression before blocking promotion


@dataclass
class ModelVersion:
    version: str
    run_id: str
    model_uri: str
    onnx_path: str
    auc: float
    precision: float
    recall: float
    registered_at: float = field(default_factory=time.time)


class VersionManager:
    def __init__(self, onnx_pool, shadow_scorer=None) -> None:
        self._pool = onnx_pool
        self._shadow = shadow_scorer
        self._current: Optional[ModelVersion] = None
        self._shadow_candidate: Optional[ModelVersion] = None
        self._shadow_start: Optional[float] = None
        self._lock = threading.Lock()
        self._client = mlflow.MlflowClient()

    def start_polling(self) -> None:
        thread = threading.Thread(target=self._poll_loop, daemon=True)
        thread.start()

    def _poll_loop(self) -> None:
        while True:
            try:
                self._check_for_staging_model()
                self._maybe_promote_shadow()
            except Exception as e:
                log.warning("Version manager poll error", error=str(e))
            time.sleep(60)

    def _check_for_staging_model(self) -> None:
        try:
            staging_mv = self._client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, "staging"
            )
        except mlflow.exceptions.MlflowException:
            return  # No staging model

        with self._lock:
            if self._shadow_candidate and self._shadow_candidate.version == staging_mv.version:
                return  # Already tracking this candidate

        onnx_uri = self._run_onnx_uri(staging_mv.run_id)
        local_path = mlflow.artifacts.download_artifacts(onnx_uri)

        metrics = self._client.get_run(staging_mv.run_id).data.metrics
        candidate = ModelVersion(
            version=staging_mv.version,
            run_id=staging_mv.run_id,
            model_uri=onnx_uri,
            onnx_path=local_path,
            auc=metrics.get("val_auc", 0.0),
            precision=metrics.get("val_precision", 0.0),
            recall=metrics.get("val_recall", 0.0),
        )

        log.info("New staging model detected", version=candidate.version, auc=candidate.auc)

        if self._shadow:
            self._shadow.reload(candidate.onnx_path)

        with self._lock:
            self._shadow_candidate = candidate
            self._shadow_start = time.time()

    def _maybe_promote_shadow(self) -> None:
        with self._lock:
            candidate = self._shadow_candidate
            start = self._shadow_start

        if not candidate or not start:
            return

        elapsed = time.time() - start
        if elapsed < SHADOW_PROMOTION_WINDOW_S:
            log.debug(
                "Shadow period ongoing",
                elapsed_min=elapsed / 60,
                remaining_min=(SHADOW_PROMOTION_WINDOW_S - elapsed) / 60,
            )
            return

        # Fetch live shadow comparison metrics from MLflow (written by monitoring service)
        shadow_run_metrics = self._get_shadow_comparison_metrics(candidate.version)
        if not shadow_run_metrics:
            log.warning("No shadow comparison metrics available — holding promotion")
            return

        baseline_auc = self._current.auc if self._current else 0.0
        if shadow_run_metrics["auc"] >= baseline_auc - METRIC_TOLERANCE:
            self._promote(candidate)
        else:
            log.warning(
                "Shadow promotion blocked — metric regression",
                baseline_auc=baseline_auc,
                candidate_auc=shadow_run_metrics["auc"],
            )
            with self._lock:
                self._shadow_candidate = None
                self._shadow_start = None

    def _promote(self, candidate: ModelVersion) -> None:
        log.info("Promoting model to production", version=candidate.version)
        self._pool.reload(candidate.onnx_path)
        self._client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", candidate.version)
        with self._lock:
            self._current = candidate
            self._shadow_candidate = None
            self._shadow_start = None
        log.info("Model promoted", version=candidate.version)

    def rollback(self) -> bool:
        """One-click rollback to the previous production version."""
        try:
            versions = self._client.search_model_versions(
                filter_string=f"name='{MLFLOW_MODEL_NAME}'",
                order_by=["creation_timestamp DESC"],
                max_results=10,
            )
            production_versions = [
                v for v in versions
                if "production" in (v.aliases or []) or v.current_stage == "Production"
            ]
            if len(production_versions) < 2:
                log.warning("No previous version available for rollback")
                return False

            previous = production_versions[1]
            onnx_uri = self._run_onnx_uri(previous.run_id)
            local_path = mlflow.artifacts.download_artifacts(onnx_uri)
            self._pool.reload(local_path)
            self._client.set_registered_model_alias(
                MLFLOW_MODEL_NAME, "production", previous.version
            )
            log.info("Rollback complete", version=previous.version)
            return True
        except Exception as e:
            log.error("Rollback failed", error=str(e))
            return False

    def _get_shadow_comparison_metrics(self, version: str) -> dict | None:
        try:
            runs = self._client.search_runs(
                experiment_ids=["shadow-comparison"],
                filter_string=f"tags.model_version = '{version}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                return None
            return runs[0].data.metrics
        except Exception:
            return None

    @staticmethod
    def _run_onnx_uri(run_id: str) -> str:
        return f"runs:/{run_id}/model/model.onnx"
