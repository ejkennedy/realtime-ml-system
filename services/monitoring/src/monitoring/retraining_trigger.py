"""Retraining trigger: dispatches a training job and sends alerts."""

from __future__ import annotations

import json
import os
import subprocess
import time

import httpx
import structlog

log = structlog.get_logger()


class RetrainingTrigger:
    def dispatch(self, reason: str) -> None:
        log.info("Dispatching retraining", reason=reason)

        if os.environ.get("ENV") == "kubernetes":
            self._dispatch_k8s_job(reason)
        else:
            self._dispatch_local(reason)

        self._send_alert(reason)

    def _dispatch_local(self, reason: str) -> None:
        subprocess.Popen([
            "uv", "run", "--package", "training",
            "python", "-m", "training.pipeline",
            "--trigger", reason,
        ])
        log.info("Local training subprocess spawned")

    def _dispatch_k8s_job(self, reason: str) -> None:
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            k8s_config.load_incluster_config()
            batch_v1 = k8s_client.BatchV1Api()

            job = k8s_client.V1Job(
                metadata=k8s_client.V1ObjectMeta(
                    name=f"retrain-{int(time.time())}",
                    namespace="fraud-detection",
                    labels={"trigger": reason[:63]},
                ),
                spec=k8s_client.V1JobSpec(
                    template=k8s_client.V1PodTemplateSpec(
                        spec=k8s_client.V1PodSpec(
                            restart_policy="Never",
                            containers=[k8s_client.V1Container(
                                name="trainer",
                                image=os.environ.get("TRAINER_IMAGE", "fraud-trainer:latest"),
                                command=["python", "-m", "training.pipeline", "--trigger", reason],
                                env=[k8s_client.V1EnvVar(name=k, value=v) for k, v in {
                                    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
                                    "KAFKA_BOOTSTRAP_SERVERS": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
                                }.items()],
                            )],
                        )
                    ),
                    backoff_limit=2,
                    ttl_seconds_after_finished=3600,
                ),
            )
            batch_v1.create_namespaced_job(namespace="fraud-detection", body=job)
            log.info("Kubernetes training job created")
        except Exception as e:
            log.error("Failed to create K8s job, falling back to local", error=str(e))
            self._dispatch_local(reason)

    def _send_alert(self, reason: str) -> None:
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")
        if not webhook_url:
            return
        payload = {
            "text": f":warning: *Fraud model retraining triggered*\nReason: `{reason}`\nEnvironment: `{os.environ.get('ENV', 'unknown')}`",
        }
        try:
            httpx.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            log.warning("Alert webhook failed", error=str(e))
