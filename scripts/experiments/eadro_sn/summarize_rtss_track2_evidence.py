"""Build RTSS Track 2 evidence tables from existing Eadro-SN replay traces.

This script is intentionally lightweight: it does not load models, datasets, or
large arrays. It only reads replay summary JSON files and the clean run JSONL
event trace, then writes a compact evidence summary for documentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MAIN_SUMMARY = Path(
    "results/experiments/eadro_sn/realtime_moe_top3_guarded_cleancheck/"
    "service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_163640_summary.json"
)
DEFAULT_MAIN_EVENTS = Path(
    "results/experiments/eadro_sn/realtime_moe_top3_guarded_cleancheck/"
    "service_aware_eadro_sn_s42_warmv6_topk2_lr3e4_blr0p1_prior0p6_wo_logs_test_20260427_163640_events.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/experiments/eadro_sn/rtss_track2_evidence/rtss_track2_evidence_summary.json"
)

DEFAULT_DEADLINES_MS = [25, 50, 75, 100, 150, 200]
DEFAULT_OVERLOAD_INTERVALS_MS = [200, 100, 50, 25, 10]
DEFAULT_OVERLOAD_DEADLINE_MS = 100

DEFAULT_RESOURCE_ROWS = [
    (
        "Ours full",
        DEFAULT_MAIN_SUMMARY,
    ),
    (
        "w/o service prior",
        Path(
            "results/experiments/eadro_sn/final_ablation_20260428/prior_sensitivity/"
            "arch_no_service_prior_wo_logs_test_20260428_090737_summary.json"
        ),
    ),
    (
        "w/o MoE / single shared",
        Path(
            "results/experiments/eadro_sn/final_ablation_20260427/architecture/"
            "v6_3layer_anomaly_label_wo_logs_top3_guarded_posthoc_summary.json"
        ),
    ),
    (
        "rank2 small",
        Path(
            "results/experiments/eadro_sn/final_ablation_20260427/efficiency_cleancheck/"
            "efficiency_small_rank2_prior0p6_wo_logs_test_20260427_222153_summary.json"
        ),
    ),
    (
        "rank8 large",
        Path(
            "results/experiments/eadro_sn/final_ablation_20260427/efficiency_cleancheck/"
            "efficiency_large_rank8_prior0p6_wo_logs_test_20260427_222431_summary.json"
        ),
    ),
    (
        "XGBoost ensemble-64",
        Path(
            "results/baselines/xgboost_ensemble64_eadro_strict_s42_metrics_logs_traces_e500_d3/"
            "full_replay_summary.json"
        ),
    ),
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": sum(values) / len(values),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def binary_metrics(truth: list[bool], pred: list[bool]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(truth, pred) if y and p)
    tn = sum(1 for y, p in zip(truth, pred) if not y and not p)
    fp = sum(1 for y, p in zip(truth, pred) if not y and p)
    fn = sum(1 for y, p in zip(truth, pred) if y and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(truth) if truth else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def detection_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("replay_detection_metrics")
    if metrics is None:
        metrics = get_nested(summary, "test_result", "metrics", default={})
    return metrics or {}


def response_latency(summary: dict[str, Any]) -> dict[str, Any]:
    return get_nested(summary, "response_latency", "response_time_ms", default={}) or {}


def build_multi_deadline(events: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    response_times = [float(row["response_time_ms"]) for row in events]
    truth = [bool(row["window_anomaly_label"]) for row in events]
    raw_predictions = [bool(row["window_anomaly_prediction"]) for row in events]
    summary_response = response_latency(summary)
    p99_ms = float(summary_response.get("p99_ms", percentile(response_times, 99)))
    max_ms = float(summary_response.get("max_ms", max(response_times) if response_times else 0.0))
    out = []
    total = len(response_times)
    for deadline_ms in DEFAULT_DEADLINES_MS:
        miss_count = sum(1 for value in response_times if value > deadline_ms)
        timely_predictions = [
            pred if response_ms <= deadline_ms else False
            for pred, response_ms in zip(raw_predictions, response_times)
        ]
        out.append(
            {
                "deadline_ms": deadline_ms,
                "miss_count": miss_count,
                "miss_rate_pct": (miss_count / total * 100.0) if total else 0.0,
                "p99_ms": p99_ms,
                "max_ms": max_ms,
                "deadline_effective_metrics": binary_metrics(truth, timely_predictions),
            }
        )
    return out


def simulate_overload(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    service_times = [float(row["processing_ms"]) for row in events]
    out = []
    for interval_ms in DEFAULT_OVERLOAD_INTERVALS_MS:
        queue_delays: list[float] = []
        responses: list[float] = []
        server_available_ms = 0.0
        for idx, service_ms in enumerate(service_times):
            arrival_ms = idx * interval_ms
            start_ms = max(float(arrival_ms), server_available_ms)
            finish_ms = start_ms + service_ms
            queue_delays.append(start_ms - arrival_ms)
            responses.append(finish_ms - arrival_ms)
            server_available_ms = finish_ms

        elapsed_s = (server_available_ms / 1000.0) if server_available_ms > 0 else 0.0
        miss_count = sum(1 for value in responses if value > DEFAULT_OVERLOAD_DEADLINE_MS)
        total = len(responses)
        out.append(
            {
                "interval_ms": interval_ms,
                "load_x": 100.0 / interval_ms,
                "deadline_ms": DEFAULT_OVERLOAD_DEADLINE_MS,
                "miss_count": miss_count,
                "miss_rate_pct": (miss_count / total * 100.0) if total else 0.0,
                "throughput_windows_per_s": (total / elapsed_s) if elapsed_s else 0.0,
                "queue_delay": latency_stats(queue_delays),
                "response_latency": latency_stats(responses),
                "processing_latency": latency_stats(service_times),
            }
        )
    return out


def build_resource_row(method: str, summary_path: Path, repo_root: Path) -> dict[str, Any]:
    full_path = resolve_path(summary_path, repo_root)
    summary = read_json(full_path)
    metrics = detection_metrics(summary)
    response = response_latency(summary)
    return {
        "method": method,
        "f1": metrics.get("f1"),
        "miss_rate_pct": summary.get("deadline_miss_rate_pct"),
        "p99_ms": response.get("p99_ms"),
        "max_ms": response.get("max_ms"),
        "gpu_peak_allocated_mb": summary.get("max_memory_allocated_mb"),
        "num_steps": summary.get("num_steps"),
        "source_summary": str(full_path),
    }


def build_summary(repo_root: Path, main_summary_path: Path, main_events_path: Path) -> dict[str, Any]:
    main_summary_abs = resolve_path(main_summary_path, repo_root)
    main_events_abs = resolve_path(main_events_path, repo_root)
    main_summary = read_json(main_summary_abs)
    main_events = read_jsonl(main_events_abs)
    main_metrics = detection_metrics(main_summary)
    main_response = response_latency(main_summary)

    return {
        "source_summary": str(main_summary_abs),
        "source_events": str(main_events_abs),
        "main_result": {
            "f1": main_metrics.get("f1"),
            "precision": main_metrics.get("precision"),
            "recall": main_metrics.get("recall"),
            "accuracy": main_metrics.get("accuracy"),
            "miss_rate_pct": main_summary.get("deadline_miss_rate_pct"),
            "p99_ms": main_response.get("p99_ms"),
            "max_ms": main_response.get("max_ms"),
            "gpu_peak_allocated_mb": main_summary.get("max_memory_allocated_mb"),
            "num_steps": main_summary.get("num_steps"),
        },
        "multi_deadline_from_clean_trace": build_multi_deadline(main_events, main_summary),
        "trace_driven_overload_simulation": simulate_overload(main_events),
        "resource_efficiency_rows": [
            build_resource_row(method, path, repo_root) for method, path in DEFAULT_RESOURCE_ROWS
        ],
        "notes": [
            "Multi-deadline is recomputed from the clean 568-step paced replay response_time_ms trace.",
            "Deadline-effective metrics treat a late decision as no timely alert for that window.",
            "Overload is a deterministic single-server replay simulation using measured per-window processing_ms from the clean run; it estimates queueing under shorter arrival intervals without rerunning the model.",
            "CPU RSS was not recorded by the original replay runner; GPU peak allocated memory is available for CUDA runs.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN_SUMMARY)
    parser.add_argument("--main-events", type=Path, default=DEFAULT_MAIN_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script()
    output_path = resolve_path(args.output, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(repo_root, args.main_summary, args.main_events)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
