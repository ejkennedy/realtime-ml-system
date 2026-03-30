"""
Concept drift detector using Evidently AI.

Compares current production window against the reference dataset (training slice)
and triggers retraining if drift thresholds are exceeded.

Run as a standalone process (Kubernetes CronJob):
    python -m monitoring.drift_detector --window-hours 1
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report

log = structlog.get_logger()

DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", 0.30))
TARGET_PSI_THRESHOLD = float(os.environ.get("TARGET_PSI_THRESHOLD", 0.25))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "./reports"))

FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category_encoded",
    "pos_type_encoded", "tx_count_1m", "tx_count_5m", "tx_count_1h",
    "tx_count_24h", "amount_sum_1h", "amount_avg_1h", "amount_max_1h",
    "amount_sum_24h", "distinct_merchants_1h", "distinct_countries_1h",
    "card_risk_score", "merchant_fraud_rate_30d", "merchant_avg_amount",
    "card_avg_spend_30d", "amount_vs_avg_ratio", "amount_vs_merchant_ratio",
]


@dataclass
class DriftReport:
    drift_share: float
    target_psi: float
    drifted_features: list[str]
    data_quality_issues: list[str]
    should_retrain: bool
    retrain_reason: str
    report_path: Optional[str]
    generated_at: datetime


class DriftDetector:
    def __init__(self, reference_path: Optional[str] = None) -> None:
        self._reference_path = reference_path or os.environ.get(
            "REFERENCE_DATA_PATH", "./data/reference/reference_window.parquet"
        )
        self._reference: Optional[pd.DataFrame] = None
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_reference(self) -> pd.DataFrame:
        if self._reference is not None:
            return self._reference
        path = Path(self._reference_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference dataset not found: {path}")
        self._reference = pd.read_parquet(path)
        log.info("Reference dataset loaded", n_rows=len(self._reference))
        return self._reference

    def run(self, current_df: pd.DataFrame) -> DriftReport:
        reference = self._load_reference()

        column_mapping = ColumnMapping(
            target="is_fraud",
            prediction="fraud_score",
            numerical_features=[f for f in FEATURE_COLS if f not in (
                "merchant_category_encoded", "pos_type_encoded", "day_of_week"
            )],
            categorical_features=["merchant_category_encoded", "pos_type_encoded", "day_of_week"],
        )

        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(
            reference_data=reference,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        report_dict = report.as_dict()

        # Extract key metrics
        drift_result = report_dict["metrics"][0]["result"]
        drift_share = float(drift_result.get("drift_share", 0))
        drifted_features = [
            col for col, data in drift_result.get("drift_by_columns", {}).items()
            if data.get("drift_detected", False)
        ]

        target_result = report_dict["metrics"][1]["result"]
        target_psi = float(target_result.get("psi", 0) or 0)

        quality_result = report_dict["metrics"][2]["result"]
        quality_issues = []
        for col, stats in quality_result.get("current", {}).items():
            if isinstance(stats, dict) and stats.get("share_of_missing_values", 0) > 0.01:
                quality_issues.append(f"{col}: {stats['share_of_missing_values']:.1%} missing")

        # Save HTML report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = str(REPORTS_DIR / f"drift_report_{timestamp}.html")
        report.save_html(report_path)

        should_retrain, reason = self._evaluate_thresholds(
            drift_share, target_psi, drifted_features
        )

        result = DriftReport(
            drift_share=drift_share,
            target_psi=target_psi,
            drifted_features=drifted_features,
            data_quality_issues=quality_issues,
            should_retrain=should_retrain,
            retrain_reason=reason,
            report_path=report_path,
            generated_at=datetime.now(timezone.utc),
        )

        self._export_metrics(result)
        log.info(
            "Drift report complete",
            drift_share=round(drift_share, 3),
            target_psi=round(target_psi, 3),
            should_retrain=should_retrain,
            reason=reason,
        )
        return result

    def _evaluate_thresholds(
        self, drift_share: float, target_psi: float, drifted_features: list[str]
    ) -> tuple[bool, str]:
        if drift_share > DRIFT_SHARE_THRESHOLD:
            return True, f"feature_drift:{drift_share:.3f}>{DRIFT_SHARE_THRESHOLD}"
        if target_psi > TARGET_PSI_THRESHOLD:
            return True, f"target_psi:{target_psi:.3f}>{TARGET_PSI_THRESHOLD}"
        # High-impact feature drift even below overall threshold
        critical_features = {"amount", "tx_count_1h", "card_risk_score"}
        critical_drifted = critical_features & set(drifted_features)
        if len(critical_drifted) >= 2:
            return True, f"critical_features_drifted:{','.join(critical_drifted)}"
        return False, "ok"

    def _export_metrics(self, report: DriftReport) -> None:
        """Push metrics to Prometheus push gateway."""
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            registry = CollectorRegistry()
            drift_gauge = Gauge(
                "fraud_feature_drift_share", "Feature drift share", registry=registry
            )
            psi_gauge = Gauge(
                "fraud_target_psi", "Target variable PSI", registry=registry
            )
            drift_gauge.set(report.drift_share)
            psi_gauge.set(report.target_psi)
            push_to_gateway(
                os.environ.get("PROMETHEUS_PUSHGATEWAY", "localhost:9091"),
                job="drift-detector",
                registry=registry,
            )
        except Exception as e:
            log.debug("Prometheus push failed (non-critical)", error=str(e))


def load_current_window(window_hours: int = 1) -> pd.DataFrame:
    """Load the most recent N hours of scored transactions."""
    # In production: query Iceberg table
    # For local dev: load from parquet files written by the mock scorer
    data_path = Path("./data/scored")
    if not data_path.exists():
        raise FileNotFoundError("No scored transaction data found. Run the pipeline first.")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    dfs = []
    for f in sorted(data_path.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= cutoff]
        dfs.append(df)
    if not dfs:
        raise ValueError(f"No data in the last {window_hours}h")
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--window-hours", default=1, show_default=True)
    @click.option("--reference-path", default=None)
    def run_check(window_hours: int, reference_path: Optional[str]) -> None:
        current = load_current_window(window_hours)
        detector = DriftDetector(reference_path)
        report = detector.run(current)

        if report.should_retrain:
            log.warning("RETRAINING TRIGGERED", reason=report.retrain_reason)
            from monitoring.retraining_trigger import RetrainingTrigger
            trigger = RetrainingTrigger()
            trigger.dispatch(report.retrain_reason)
        else:
            log.info("No drift detected — model healthy")

    run_check()
