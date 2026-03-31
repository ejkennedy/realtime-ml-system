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
from pathlib import Path

import ray
from dotenv import load_dotenv
from ray import serve

import structlog

log = structlog.get_logger()

load_dotenv()


def _resolve_model_path() -> str:
    model_path = os.environ.get("ONNX_MODEL_PATH", "/models/fraud_detector/latest/model.onnx")
    if os.environ.get("SERVE_USE_QUANTIZED_MODEL", "false") != "true":
        return model_path

    candidate = Path(model_path)
    if candidate.suffix == ".onnx":
        quantized_candidate = candidate.with_name(f"{candidate.stem}.int8.onnx")
        if quantized_candidate.exists():
            log.info("Using quantized ONNX model", path=str(quantized_candidate))
            return str(quantized_candidate)
        log.warning("Quantized ONNX model requested but not found; falling back to fp32", path=str(candidate))
    return model_path


def build_app():
    from serving.deployments.fraud_scorer import FraudScorer
    from serving.deployments.router import FraudRouter

    primary_model_path = _resolve_model_path()
    os.environ["ONNX_MODEL_PATH"] = primary_model_path
    primary = FraudScorer.bind()
    use_router = os.environ.get("SERVE_USE_ROUTER", "true") == "true"

    shadow_enabled = os.environ.get("SHADOW_ENABLED", "true") == "true"
    shadow = None
    if shadow_enabled:
        shadow_model_path = os.environ.get(
            "SHADOW_MODEL_PATH",
            primary_model_path,
        )
        if shadow_model_path:
            # Separate deployment with potentially different model version
            from serving.deployments.fraud_scorer import FraudScorer
            shadow_env = {"ONNX_MODEL_PATH": shadow_model_path}
            shadow = FraudScorer.options(
                name="fraud-scorer-shadow",
                ray_actor_options={"runtime_env": {"env_vars": shadow_env}},
            ).bind()

    if not use_router and not shadow_enabled:
        return primary

    router = FraudRouter.bind(primary_handle=primary, shadow_handle=shadow)
    return router


def main() -> None:
    ray_kwargs = {"dashboard_host": "0.0.0.0", "ignore_reinit_error": True}
    temp_dir = os.environ.get("RAY_TMPDIR")
    if temp_dir:
        ray_kwargs["_temp_dir"] = temp_dir
    ray.init(**ray_kwargs)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    app = build_app()
    serve.run(app, name="fraud-detection")

    log.info("Fraud detection serving started", host="0.0.0.0", port=8000)

    # Start version manager polling
    version_manager_enabled = os.environ.get("VERSION_MANAGER_ENABLED", "true") == "true"
    if version_manager_enabled:
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
    else:
        log.info("Version manager disabled")

    # Block indefinitely
    import time
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
