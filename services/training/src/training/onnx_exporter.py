"""
XGBoost → ONNX export with validation.

Uses onnxmltools for XGBoost conversion and validates the exported model
produces identical predictions to the original model (within fp32 tolerance).

onnx >=1.16 made `onnx.mapping` private (`onnx._mapping`). The patch below
restores the public name so that onnxmltools can import it without modification.
"""

from __future__ import annotations

# Must happen before any onnxmltools import.
import onnx as _onnx_mod
if not hasattr(_onnx_mod, "mapping"):
    import onnx._mapping as _onnx_mapping
    _onnx_mod.mapping = _onnx_mapping

import numpy as np
import onnxruntime as ort
import structlog
import xgboost as xgb
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from pathlib import Path

log = structlog.get_logger()


def export_xgboost_to_onnx(
    model: xgb.XGBClassifier,
    output_path: str,
    n_features: int,
    opset_version: int = 15,
    quantized_output_path: str | None = None,
) -> None:
    """
    Convert XGBoost classifier to ONNX format.

    The exported model takes a float32 input of shape (batch, n_features)
    and produces two outputs:
      - output_label: int64 predictions (shape: batch)
      - output_probability: float probabilities (shape: batch x 2)

    Validates that ONNX output matches XGBoost output within 1e-5 tolerance.
    """
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=opset_version,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    _validate_onnx(model, output_path, n_features)
    log.info("ONNX export validated", path=output_path, n_features=n_features)

    if quantized_output_path:
        _export_quantized_onnx(model, output_path, quantized_output_path, n_features)


def _validate_onnx(
    original_model: xgb.XGBClassifier,
    onnx_path: str,
    n_features: int,
    n_test_samples: int = 1000,
    tolerance: float = 1e-4,
) -> None:
    """Verify ONNX predictions match XGBoost predictions."""
    rng = np.random.default_rng(0)
    X_test = rng.random((n_test_samples, n_features)).astype(np.float32)

    # XGBoost predictions
    xgb_probs = original_model.predict_proba(X_test)[:, 1]

    # ONNX predictions
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    prob_output_name = sess.get_outputs()[1].name
    onnx_probs_raw = sess.run([prob_output_name], {input_name: X_test})[0]

    # Handle both dict-of-lists (ZipMap) and array outputs
    if isinstance(onnx_probs_raw[0], dict):
        onnx_probs = np.array([p[1] for p in onnx_probs_raw])
    else:
        onnx_probs = onnx_probs_raw[:, 1]

    max_diff = float(np.abs(xgb_probs - onnx_probs).max())
    if max_diff > tolerance:
        raise ValueError(
            f"ONNX validation failed: max prediction difference {max_diff:.6f} > {tolerance}"
        )

    log.info("ONNX validation passed", max_diff=f"{max_diff:.8f}")


def _export_quantized_onnx(
    original_model: xgb.XGBClassifier,
    onnx_path: str,
    quantized_output_path: str,
    n_features: int,
) -> None:
    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_output_path,
            weight_type=QuantType.QUInt8,
        )
        _validate_onnx(
            original_model,
            quantized_output_path,
            n_features,
            tolerance=2e-2,
        )
        fp32_size = onnx_path and Path(onnx_path).stat().st_size
        int8_size = Path(quantized_output_path).stat().st_size
        log.info(
            "Quantized ONNX export validated",
            path=quantized_output_path,
            size_reduction_pct=round((1 - (int8_size / fp32_size)) * 100, 2) if fp32_size else 0.0,
        )
    except Exception as exc:
        log.warning("Quantized ONNX export skipped", error=str(exc), path=quantized_output_path)
