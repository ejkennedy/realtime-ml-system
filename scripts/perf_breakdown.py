from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray
from ray import serve


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_perf_snapshot(
        app_name=args.app_name,
        deployment_name=args.deployment,
        finalize_profiles=args.finalize_profiles,
    )
    load_summary_path = resolve_load_summary(reports_dir, args.load_summary)
    load_summary = parse_load_summary(load_summary_path) if load_summary_path else None
    profile_artifacts = snapshot.get("onnx", {}).get("profile_artifacts", [])
    onnx_profile_summary = summarize_onnx_profiles(profile_artifacts)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"perf_breakdown_{timestamp}.md"
    output_path.write_text(
        render_markdown(snapshot, load_summary_path, load_summary, profile_artifacts, onnx_profile_summary),
        encoding="utf-8",
    )
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Markdown perf breakdown from the running Serve deployment.")
    parser.add_argument("--app-name", default="fraud-detection")
    parser.add_argument("--deployment", default="FraudScorer")
    parser.add_argument("--load-summary", default="", help="Optional explicit load-test summary markdown path.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--finalize-profiles", action="store_true", default=False)
    return parser.parse_args()


def fetch_perf_snapshot(app_name: str, deployment_name: str, finalize_profiles: bool) -> dict[str, Any]:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
    response = handle.options(method_name="get_perf_snapshot").remote(finalize_profiles=finalize_profiles)
    if hasattr(response, "result"):
        return response.result()
    return ray.get(response)


def resolve_load_summary(reports_dir: Path, explicit_path: str) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    candidates = [
        path for path in reports_dir.glob("load_test_summary*.md")
        if "benchmark_" not in path.name and "verify" not in path.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_load_summary(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    stats = {}
    for metric in ["Total requests", "Total failures", "Failure rate", "Average RPS"]:
        match = re.search(rf"\| {re.escape(metric)} \| ([^|]+) \|", text)
        if match:
            stats[metric] = match.group(1).strip()
    for metric in ["p50", "p95", "p99", "p99.9", "Mean"]:
        match = re.search(rf"\| {re.escape(metric)} \| ([^|]+) \|", text)
        if match:
            stats[metric] = match.group(1).strip()
    label_match = re.search(r"Label: `([^`]+)`", text)
    if label_match:
        stats["Label"] = label_match.group(1)
    return stats


def render_markdown(
    snapshot: dict[str, Any],
    load_summary_path: Path | None,
    load_summary: dict[str, str] | None,
    profile_artifacts: list[str],
    onnx_profile_summary: dict[str, Any],
) -> str:
    lines = [
        "# Perf Breakdown",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Model version: `{snapshot.get('model_version', 'unknown')}`",
        "",
        "## Request Path Stages",
        "",
        "| Stage | Count | Mean (ms) | p50 | p95 | p99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, stats in snapshot.get("stage_latency_ms", {}).items():
        lines.append(render_row(name, stats))

    lines.extend(
        [
            "",
            "## ONNX And Pool",
            "",
            f"- Pool size: `{snapshot.get('onnx', {}).get('pool_size', 'n/a')}`",
            f"- Profiling enabled: `{snapshot.get('onnx', {}).get('profiling_enabled', False)}`",
            "",
            "| Stage | Count | Mean (ms) | p50 | p95 | p99 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|",
            render_row("pool_wait", snapshot.get("onnx", {}).get("pool_wait_ms", {})),
            render_row("onnx_run", snapshot.get("onnx", {}).get("onnx_run_ms", {})),
        ]
    )

    if load_summary_path and load_summary:
        lines.extend(
            [
                "",
                "## Latest Load Summary",
                "",
                f"Source: `{load_summary_path}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for key in ["Label", "Total requests", "Total failures", "Failure rate", "Average RPS", "p50", "p95", "p99", "p99.9", "Mean"]:
            if key in load_summary:
                lines.append(f"| {key} | {load_summary[key]} |")

    lines.extend(
        [
            "",
            "## ONNX Profile Files",
            "",
        ]
    )
    if profile_artifacts:
        for artifact in profile_artifacts:
            artifact_path = Path(artifact)
            size = artifact_path.stat().st_size if artifact_path.exists() else 0
            lines.append(f"- `{artifact}` ({size} bytes)")
    else:
        lines.append("- None")

    if onnx_profile_summary:
        lines.extend(
            [
                "",
                "## ONNX Profile Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| model_run_ms | {fmt(onnx_profile_summary.get('model_run_ms'))} |",
                f"| executor_ms | {fmt(onnx_profile_summary.get('executor_ms'))} |",
                f"| kernel_ms | {fmt(onnx_profile_summary.get('kernel_ms'))} |",
                f"| node_count | {onnx_profile_summary.get('node_count', 0)} |",
            ]
        )
        if onnx_profile_summary.get("top_nodes"):
            lines.extend(
                [
                    "",
                    "| Node | Kernel Time (ms) |",
                    "|---|---:|",
                ]
            )
            for node in onnx_profile_summary["top_nodes"]:
                lines.append(f"| {node['name']} | {fmt(node['kernel_ms'])} |")

    lines.extend(
        [
            "",
            "## Raw Snapshot",
            "",
            "```json",
            json.dumps(snapshot, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| {name} | {stats.get('count', 0)} | {fmt(stats.get('mean'))} | "
        f"{fmt(stats.get('p50'))} | {fmt(stats.get('p95'))} | "
        f"{fmt(stats.get('p99'))} | {fmt(stats.get('max'))} |"
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summarize_onnx_profiles(profile_artifacts: list[str]) -> dict[str, Any]:
    if not profile_artifacts:
        return {}

    latest_path = max((Path(path) for path in profile_artifacts if Path(path).exists()), key=lambda p: p.stat().st_mtime, default=None)
    if latest_path is None:
        return {}

    try:
        events = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    model_run_ms = None
    executor_ms = None
    node_rows: list[dict[str, Any]] = []
    for event in events:
        name = event.get("name", "")
        dur_ms = round(float(event.get("dur", 0)) / 1000, 3)
        if name == "model_run":
            model_run_ms = dur_ms
        elif name == "SequentialExecutor::Execute":
            executor_ms = dur_ms
        elif name.endswith("_kernel_time"):
            node_rows.append({"name": name, "kernel_ms": dur_ms})

    return {
        "profile_path": str(latest_path),
        "model_run_ms": model_run_ms,
        "executor_ms": executor_ms,
        "kernel_ms": round(sum(node["kernel_ms"] for node in node_rows), 3) if node_rows else None,
        "node_count": len(node_rows),
        "top_nodes": sorted(node_rows, key=lambda row: row["kernel_ms"], reverse=True)[:5],
    }


if __name__ == "__main__":
    main()
