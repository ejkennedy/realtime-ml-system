"""
Ray Serve application factory.

Wires: FraudRouter → (FraudScorer [primary], FraudScorer [shadow])

Run locally:
    python -m serving.app

Or deploy declaratively:
    serve run serving.app:build_app
"""

from __future__ import annotations

import os

import ray
from ray import serve

import structlog

log = structlog.get_logger()


def build_app():
    from serving.deployments.fraud_scorer import FraudScorer
    from serving.deployments.router import FraudRouter

    primary = FraudScorer.bind()

    shadow_enabled = os.environ.get("SHADOW_ENABLED", "true") == "true"
    shadow = None
    if shadow_enabled:
        shadow_model_path = os.environ.get(
            "SHADOW_MODEL_PATH",
            os.environ.get("ONNX_MODEL_PATH", ""),
        )
        if shadow_model_path:
            # Separate deployment with potentially different model version
            from serving.deployments.fraud_scorer import FraudScorer
            shadow_env = {"ONNX_MODEL_PATH": shadow_model_path}
            shadow = FraudScorer.options(
                name="fraud-scorer-shadow",
                ray_actor_options={"runtime_env": {"env_vars": shadow_env}},
            ).bind()

    router = FraudRouter.bind(primary_handle=primary, shadow_handle=shadow)
    return router


def main() -> None:
    ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)

    app = build_app()
    serve.run(app, host="0.0.0.0", port=8000, name="fraud-detection")

    log.info("Fraud detection serving started", host="0.0.0.0", port=8000)

    # Start version manager polling
    try:
        from serving.models.onnx_runner import OnnxSessionPool
        from serving.models.version_manager import VersionManager
        pool = OnnxSessionPool(
            os.environ.get("ONNX_MODEL_PATH", "/models/fraud_detector/latest/model.onnx"),
            pool_size=int(os.environ.get("ONNX_SESSION_POOL_SIZE", 4)),
        )
        vm = VersionManager(pool)
        vm.start_polling()
        log.info("Version manager polling started")
    except Exception as e:
        log.warning("Version manager not started", error=str(e))

    # Block indefinitely
    import time
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
