"""
XGBoost fraud detection model trainer.

Trains on synthetic or Iceberg-loaded data, evaluates with stratified k-fold,
logs to MLflow, and exports to ONNX via onnxmltools.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

log = structlog.get_logger()

FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category_encoded",
    "pos_type_encoded", "tx_count_1m", "tx_count_5m", "tx_count_1h",
    "tx_count_24h", "amount_sum_1h", "amount_avg_1h", "amount_max_1h",
    "amount_sum_24h", "distinct_merchants_1h", "distinct_countries_1h",
    "card_risk_score", "merchant_fraud_rate_30d", "merchant_avg_amount",
    "card_avg_spend_30d", "amount_vs_avg_ratio", "amount_vs_merchant_ratio",
]
TARGET_COL = "is_fraud"

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr", "logloss"],
    "max_depth": 6,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 49,   # class imbalance: 49:1 legit:fraud
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 20,
}


class XGBoostTrainer:
    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        model_name: str = "fraud-detector",
        output_dir: str = "./models/registry",
    ) -> None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.set_experiment(experiment_name)
        self._model_name = model_name
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, df: pd.DataFrame, register: bool = True) -> tuple[xgb.XGBClassifier, dict]:
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values

        log.info("Training XGBoost", n_samples=len(df), fraud_rate=y.mean())

        with mlflow.start_run() as run:
            mlflow.log_params(XGBOOST_PARAMS)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("fraud_rate", round(float(y.mean()), 4))

            # Stratified 5-fold CV for robust metric estimation
            cv_metrics = self._cross_validate(X, y)
            for k, v in cv_metrics.items():
                mlflow.log_metric(f"cv_{k}", v)
            log.info("Cross-validation complete", **cv_metrics)

            # Final model on full dataset
            model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            X_train, X_val = X[: int(0.9 * len(X))], X[int(0.9 * len(X)) :]
            y_train, y_val = y[: int(0.9 * len(y))], y[int(0.9 * len(y)) :]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )
            model._estimator_type = "classifier"

            val_preds = model.predict_proba(X_val)[:, 1]
            val_metrics = {
                "val_auc": roc_auc_score(y_val, val_preds),
                "val_aucpr": average_precision_score(y_val, val_preds),
                "val_precision": precision_score(y_val, val_preds >= 0.5),
                "val_recall": recall_score(y_val, val_preds >= 0.5),
                "val_f1": f1_score(y_val, val_preds >= 0.5),
            }
            for k, v in val_metrics.items():
                mlflow.log_metric(k, round(v, 4))
            log.info("Validation metrics", **{k: round(v, 4) for k, v in val_metrics.items()})

            # Export to ONNX
            onnx_path, quantized_onnx_path = self._export_onnx(model, run.info.run_id)
            mlflow.log_artifact(str(onnx_path), artifact_path="model")
            if quantized_onnx_path is not None and quantized_onnx_path.exists():
                mlflow.log_artifact(str(quantized_onnx_path), artifact_path="model")

            # Log sklearn-style model for MLflow model registry
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgboost_model",
                registered_model_name=self._model_name if register else None,
            )

            # Tag as staging for shadow deployment workflow
            if register:
                client = mlflow.MlflowClient()
                mv = client.get_latest_versions(self._model_name, stages=["None"])[0]
                client.set_registered_model_alias(self._model_name, "staging", mv.version)
                mlflow.log_param("registered_version", mv.version)
                log.info("Model registered as staging", version=mv.version)

            return model, {**cv_metrics, **val_metrics, "run_id": run.info.run_id}

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> dict:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, aucprs = [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            m = xgb.XGBClassifier(**{**XGBOOST_PARAMS, "n_estimators": 100, "early_stopping_rounds": None})
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))
            aucprs.append(average_precision_score(y[val_idx], preds))
        return {
            "auc_mean": round(np.mean(aucs), 4),
            "auc_std": round(np.std(aucs), 4),
            "aucpr_mean": round(np.mean(aucprs), 4),
        }

    def _export_onnx(self, model: xgb.XGBClassifier, run_id: str) -> tuple[Path, Path | None]:
        from training.onnx_exporter import export_xgboost_to_onnx
        onnx_path = self._output_dir / f"fraud_detector/{run_id}/model.onnx"
        quantized_onnx_path = onnx_path.parent / "model.int8.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        export_xgboost_to_onnx(
            model,
            str(onnx_path),
            len(FEATURE_COLS),
            quantized_output_path=str(quantized_onnx_path),
        )

        # Also update the "latest" symlink
        latest_dir = self._output_dir / "fraud_detector/latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "model.onnx"
        quantized_latest_path = latest_dir / "model.int8.onnx"
        import shutil
        shutil.copy2(onnx_path, latest_path)
        if quantized_onnx_path.exists():
            shutil.copy2(quantized_onnx_path, quantized_latest_path)

        log.info("ONNX model exported", path=str(onnx_path))
        return onnx_path, quantized_onnx_path if quantized_onnx_path.exists() else None
