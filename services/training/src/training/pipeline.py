"""
Full training pipeline entrypoint.

Usage:
    python -m training.pipeline                    # generate synthetic data + train
    python -m training.pipeline --iceberg          # load from Iceberg offline store
    python -m training.pipeline --trigger drift    # triggered by drift detector
"""

from __future__ import annotations

import click
import structlog
from dotenv import load_dotenv

log = structlog.get_logger()

load_dotenv()


@click.command()
@click.option("--n-samples", default=500_000, show_default=True)
@click.option("--fraud-rate", default=0.02, show_default=True)
@click.option("--iceberg", is_flag=True, default=False, help="Load from Iceberg instead of synthetic data")
@click.option("--trigger", default="manual", help="Trigger reason (manual|drift|scheduled)")
@click.option("--auto-promote", is_flag=True, default=False, help="Skip shadow window and auto-promote")
@click.option("--register/--no-register", default=True, show_default=True)
def main(
    n_samples: int,
    fraud_rate: float,
    iceberg: bool,
    trigger: str,
    auto_promote: bool,
    register: bool,
) -> None:
    import mlflow

    log.info("Training pipeline started", trigger=trigger, iceberg=iceberg)

    if iceberg:
        from training.data_loader import load_from_iceberg
        df = load_from_iceberg()
    else:
        from training.data_generator import generate_training_dataset
        df = generate_training_dataset(n_samples=n_samples, fraud_rate=fraud_rate)

    log.info("Data loaded", n_rows=len(df), fraud_rate=round(df["is_fraud"].mean(), 4))

    from training.xgboost_trainer import XGBoostTrainer
    trainer = XGBoostTrainer()
    model, metrics = trainer.train(df, register=register)

    log.info("Training complete", **{k: v for k, v in metrics.items() if k != "run_id"})

    if auto_promote and register:
        client = mlflow.MlflowClient()
        model_name = "fraud-detector"
        staging_mv = client.get_model_version_by_alias(model_name, "staging")
        client.set_registered_model_alias(model_name, "production", staging_mv.version)
        log.info("Model auto-promoted to production", version=staging_mv.version)


if __name__ == "__main__":
    main()
