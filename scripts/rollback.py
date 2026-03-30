"""One-click model rollback script."""

import os
import sys

import mlflow
import structlog
from dotenv import load_dotenv

log = structlog.get_logger()

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))

from serving.models.onnx_runner import OnnxSessionPool
from serving.models.version_manager import VersionManager

pool = OnnxSessionPool.__new__(OnnxSessionPool)  # dummy pool for rollback
vm = VersionManager(pool)
success = vm.rollback()
if success:
    print("Rollback successful.")
else:
    print("Rollback failed — check logs.", file=sys.stderr)
    sys.exit(1)
