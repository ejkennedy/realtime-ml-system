"""
Latency distribution tracker and HTML report generator.

Collects p50/p95/p99 latency data during load tests and produces
matplotlib plots showing pre/post optimization distributions.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

_latency_buffer: deque[float] = deque(maxlen=100_000)


def record_latency(ms: float) -> None:
    _latency_buffer.append(ms)


def get_percentiles() -> dict[str, float]:
    if not _latency_buffer:
        return {"p50": 0, "p95": 0, "p99": 0, "p999": 0, "mean": 0, "count": 0}
    arr = np.array(_latency_buffer)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "p999": round(float(np.percentile(arr, 99.9)), 2),
        "mean": round(float(arr.mean()), 2),
        "count": len(arr),
    }


def plot_distribution(
    label: str = "current",
    output_path: Optional[str] = None,
    sla_ms: float = 50.0,
) -> str:
    """Generate a latency distribution plot. Returns the saved file path."""
    import matplotlib.pyplot as plt

    arr = np.array(_latency_buffer)
    output_path = output_path or f"./reports/latency_{label}_{int(time.time())}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Inference Latency Distribution — {label}", fontsize=14)

    # Histogram
    ax1 = axes[0]
    ax1.hist(arr, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.axvline(x=sla_ms, color="red", linestyle="--", linewidth=2, label=f"SLA ({sla_ms}ms)")
    ax1.axvline(x=float(np.percentile(arr, 95)), color="orange", linestyle="--",
                linewidth=1.5, label=f"p95 ({np.percentile(arr, 95):.1f}ms)")
    ax1.axvline(x=float(np.percentile(arr, 99)), color="purple", linestyle="--",
                linewidth=1.5, label=f"p99 ({np.percentile(arr, 99):.1f}ms)")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram")
    ax1.legend()

    # CDF
    ax2 = axes[1]
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax2.plot(sorted_arr, cdf * 100, color="steelblue", linewidth=1.5)
    ax2.axvline(x=sla_ms, color="red", linestyle="--", linewidth=2, label=f"SLA ({sla_ms}ms)")
    ax2.axhline(y=95, color="orange", linestyle=":", alpha=0.7)
    ax2.axhline(y=99, color="purple", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Percentile")
    ax2.set_title("CDF")
    ax2.set_xlim(0, min(float(np.percentile(arr, 99.9)) * 1.1, 500))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def write_markdown_summary(
    output_path: str,
    *,
    label: str = "locust",
    total_requests: int = 0,
    total_failures: int = 0,
    avg_rps: float = 0.0,
    sla_ms: float = 50.0,
) -> str:
    stats = get_percentiles()
    failure_rate = (total_failures / total_requests * 100) if total_requests else 0.0
    p95_vs_sla = "within" if stats["p95"] <= sla_ms else "above"

    content = f"""# Load Test Summary

Label: `{label}`

## Request Stats

| Metric | Value |
|---|---:|
| Total requests | {total_requests} |
| Total failures | {total_failures} |
| Failure rate | {failure_rate:.2f}% |
| Average RPS | {avg_rps:.2f} |

## Latency

| Percentile | Latency (ms) |
|---|---:|
| p50 | {stats["p50"]:.2f} |
| p95 | {stats["p95"]:.2f} |
| p99 | {stats["p99"]:.2f} |
| p99.9 | {stats["p999"]:.2f} |
| Mean | {stats["mean"]:.2f} |

## SLA Check

- Target p95 SLA: `{sla_ms:.0f} ms`
- Result: p95 is `{p95_vs_sla}` SLA
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)
