from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATTERN = re.compile(
    r"Test:\s*F1=(?P<f1>\d+\.\d+),\s*P=(?P<p>\d+\.\d+),\s*R=(?P<r>\d+\.\d+),\s*Acc=(?P<acc>\d+\.\d+)"
)

DATASET_CONFIG = {
    "msds": {
        "log_template": "results/experiments/moe_stage2/service_aware_msds_s{seed}_bs8ga2_live.log",
        "checkpoint_template": (
            "checkpoints/msds/experiments/moe_stage2/service_aware_msds_s{seed}_bs8ga2/best_model.pth"
        ),
        "replay_glob": "results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_*_summary.json",
        "routing_glob": (
            "results/experiments/moe_stage2/diagnostics/"
            "service_aware_moe_routing_msds_test_*_summary.json"
        ),
        "default_interval_ms": 1000.0,
        "default_deadline_ms": 1000.0,
    },
    "re2tt": {
        "log_template": "results/experiments/moe_stage2/service_aware_re2tt_s{seed}_bs4ga8_live.log",
        "checkpoint_template": (
            "checkpoints/rcaeval/experiments/moe_stage2/service_aware_re2tt_s{seed}_bs4ga8/best_model.pth"
        ),
        "replay_glob": "results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_*_summary.json",
        "routing_glob": (
            "results/experiments/moe_stage2/diagnostics/"
            "service_aware_moe_routing_test_*_summary.json"
        ),
        "default_interval_ms": 100.0,
        "default_deadline_ms": 100.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stage-2 service-aware MoE results across seeds")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIG.keys()), default=["msds", "re2tt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/moe_stage2/summary",
    )
    parser.add_argument(
        "--markdown-path",
        type=str,
        default="docs/ServiceAwareMoE多seed汇总.md",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_test_metrics(log_path: Path) -> dict[str, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(TEST_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "f1": float(match.group("f1")),
        "precision": float(match.group("p")),
        "recall": float(match.group("r")),
        "accuracy": float(match.group("acc")),
    }


def _match_checkpoint(candidate: dict[str, Any], checkpoint_path: Path) -> bool:
    run_info = candidate.get("run", {})
    recorded = run_info.get("checkpoint") or candidate.get("checkpoint_path")
    if recorded is None:
        return False
    try:
        return Path(recorded).resolve() == checkpoint_path.resolve()
    except OSError:
        return str(recorded) == str(checkpoint_path)


def _find_latest_matching_json(
    glob_pattern: str,
    checkpoint_path: Path,
    predicate: callable | None = None,
) -> Path | None:
    matches: list[Path] = []
    for path in PROJECT_ROOT.glob(glob_pattern):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if not _match_checkpoint(payload, checkpoint_path):
            continue
        if predicate is not None and not predicate(payload):
            continue
        matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def _format_cell(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _build_dataset_summary(dataset: str, seeds: list[int]) -> dict[str, Any]:
    cfg = DATASET_CONFIG[dataset]
    seed_rows: list[dict[str, Any]] = []
    collected = {
        "offline_f1": [],
        "offline_precision": [],
        "offline_recall": [],
        "offline_accuracy": [],
        "replay_miss_rate_pct": [],
        "replay_mean_ms": [],
        "replay_p95_ms": [],
        "replay_p99_ms": [],
        "replay_max_ms": [],
        "effective_experts": [],
        "route_switch_rate": [],
        "dominant_top1_share": [],
    }

    for seed in seeds:
        log_path = (PROJECT_ROOT / cfg["log_template"].format(seed=seed)).resolve()
        checkpoint_path = (PROJECT_ROOT / cfg["checkpoint_template"].format(seed=seed)).resolve()
        row: dict[str, Any] = {
            "seed": seed,
            "log_path": str(log_path),
            "checkpoint_path": str(checkpoint_path),
        }

        offline = _parse_test_metrics(log_path)
        if offline is not None:
            row.update(
                {
                    "offline_f1": offline["f1"],
                    "offline_precision": offline["precision"],
                    "offline_recall": offline["recall"],
                    "offline_accuracy": offline["accuracy"],
                }
            )
            collected["offline_f1"].append(offline["f1"])
            collected["offline_precision"].append(offline["precision"])
            collected["offline_recall"].append(offline["recall"])
            collected["offline_accuracy"].append(offline["accuracy"])

        replay_path = _find_latest_matching_json(
            cfg["replay_glob"],
            checkpoint_path,
            predicate=lambda payload: (
                float(payload.get("interval_ms", -1.0)) == cfg["default_interval_ms"]
                and float(payload.get("deadline_ms", -1.0)) == cfg["default_deadline_ms"]
                and bool(payload.get("prefetch_enabled", False))
                and bool(payload.get("pin_memory_enabled", False))
            ),
        )
        if replay_path is not None:
            replay = _read_json(replay_path)
            replay_latency = replay.get("response_latency", {}).get("response_time_ms", {})
            row.update(
                {
                    "replay_summary_path": str(replay_path),
                    "replay_interval_ms": float(replay.get("interval_ms", 0.0)),
                    "replay_deadline_ms": float(replay.get("deadline_ms", 0.0)),
                    "replay_miss_rate_pct": float(replay.get("deadline_miss_rate_pct", 0.0)),
                    "replay_mean_ms": float(replay_latency.get("mean_ms", 0.0)),
                    "replay_p95_ms": float(replay_latency.get("p95_ms", 0.0)),
                    "replay_p99_ms": float(replay_latency.get("p99_ms", 0.0)),
                    "replay_max_ms": float(replay_latency.get("max_ms", 0.0)),
                }
            )
            collected["replay_miss_rate_pct"].append(row["replay_miss_rate_pct"])
            collected["replay_mean_ms"].append(row["replay_mean_ms"])
            collected["replay_p95_ms"].append(row["replay_p95_ms"])
            collected["replay_p99_ms"].append(row["replay_p99_ms"])
            collected["replay_max_ms"].append(row["replay_max_ms"])

        routing_path = _find_latest_matching_json(cfg["routing_glob"], checkpoint_path)
        if routing_path is not None:
            routing = _read_json(routing_path)
            layer = next(iter(routing.get("layers", {}).values()), {})
            row.update(
                {
                    "routing_summary_path": str(routing_path),
                    "effective_experts": float(layer.get("effective_experts", 0.0)),
                    "route_switch_rate": float(layer.get("route_switch_rate", 0.0)),
                    "dominant_top1_share": float(layer.get("dominant_top1_share", 0.0)),
                }
            )
            collected["effective_experts"].append(row["effective_experts"])
            collected["route_switch_rate"].append(row["route_switch_rate"])
            collected["dominant_top1_share"].append(row["dominant_top1_share"])

        row["complete"] = offline is not None and replay_path is not None and routing_path is not None
        seed_rows.append(row)

    aggregate = {metric: _safe_stats(values) for metric, values in collected.items()}
    complete_count = sum(1 for row in seed_rows if row["complete"])
    return {
        "dataset": dataset,
        "default_interval_ms": cfg["default_interval_ms"],
        "default_deadline_ms": cfg["default_deadline_ms"],
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "complete_count": complete_count,
        "requested_seeds": seeds,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Service-aware MoE 多 Seed 汇总")
    lines.append("")
    lines.append("说明：offline 指训练日志里的最终 test 指标；replay 指默认标准口径下的 paced replay。")
    lines.append("")
    for dataset, payload in summary["datasets"].items():
        lines.append(f"## {dataset.upper()}")
        lines.append("")
        lines.append(
            f"标准 replay 口径：`interval={payload['default_interval_ms']:.0f}ms`, "
            f"`deadline={payload['default_deadline_ms']:.0f}ms`"
        )
        lines.append("")
        lines.append(
            "| seed | F1 | P | R | Acc | miss@deadline(%) | mean(ms) | p95(ms) | p99(ms) | max(ms) | eff. experts | switch rate | top1 dom. |"
        )
        lines.append(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in payload["seed_rows"]:
            lines.append(
                "| "
                f"{row['seed']} | "
                f"{_format_cell(row.get('offline_f1'))} | "
                f"{_format_cell(row.get('offline_precision'))} | "
                f"{_format_cell(row.get('offline_recall'))} | "
                f"{_format_cell(row.get('offline_accuracy'))} | "
                f"{_format_cell(row.get('replay_miss_rate_pct'))} | "
                f"{_format_cell(row.get('replay_mean_ms'))} | "
                f"{_format_cell(row.get('replay_p95_ms'))} | "
                f"{_format_cell(row.get('replay_p99_ms'))} | "
                f"{_format_cell(row.get('replay_max_ms'))} | "
                f"{_format_cell(row.get('effective_experts'))} | "
                f"{_format_cell(row.get('route_switch_rate'))} | "
                f"{_format_cell(row.get('dominant_top1_share'))} |"
            )
        agg = payload["aggregate"]
        lines.append(
            "| mean | "
            f"{_format_cell(agg['offline_f1']['mean'])} | "
            f"{_format_cell(agg['offline_precision']['mean'])} | "
            f"{_format_cell(agg['offline_recall']['mean'])} | "
            f"{_format_cell(agg['offline_accuracy']['mean'])} | "
            f"{_format_cell(agg['replay_miss_rate_pct']['mean'])} | "
            f"{_format_cell(agg['replay_mean_ms']['mean'])} | "
            f"{_format_cell(agg['replay_p95_ms']['mean'])} | "
            f"{_format_cell(agg['replay_p99_ms']['mean'])} | "
            f"{_format_cell(agg['replay_max_ms']['mean'])} | "
            f"{_format_cell(agg['effective_experts']['mean'])} | "
            f"{_format_cell(agg['route_switch_rate']['mean'])} | "
            f"{_format_cell(agg['dominant_top1_share']['mean'])} |"
        )
        lines.append(
            "| std | "
            f"{_format_cell(agg['offline_f1']['std'])} | "
            f"{_format_cell(agg['offline_precision']['std'])} | "
            f"{_format_cell(agg['offline_recall']['std'])} | "
            f"{_format_cell(agg['offline_accuracy']['std'])} | "
            f"{_format_cell(agg['replay_miss_rate_pct']['std'])} | "
            f"{_format_cell(agg['replay_mean_ms']['std'])} | "
            f"{_format_cell(agg['replay_p95_ms']['std'])} | "
            f"{_format_cell(agg['replay_p99_ms']['std'])} | "
            f"{_format_cell(agg['replay_max_ms']['std'])} | "
            f"{_format_cell(agg['effective_experts']['std'])} | "
            f"{_format_cell(agg['route_switch_rate']['std'])} | "
            f"{_format_cell(agg['dominant_top1_share']['std'])} |"
        )
        lines.append("")
        lines.append(
            f"完整闭环 seeds: {payload['complete_count']} / {len(payload['requested_seeds'])}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        dataset: _build_dataset_summary(dataset, args.seeds)
        for dataset in args.datasets
    }
    summary = {
        "experiment": "service-aware-moe-multi-seed-summary",
        "datasets": datasets,
        "seeds": args.seeds,
    }

    summary_path = output_dir / "service_aware_moe_multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = (PROJECT_ROOT / args.markdown_path).resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"JSON summary saved to: {summary_path}")
    print(f"Markdown table saved to: {markdown_path}")
    for dataset, payload in datasets.items():
        agg = payload["aggregate"]
        print(
            f"[{dataset}] "
            f"offline F1 mean/std={_format_cell(agg['offline_f1']['mean'])}/{_format_cell(agg['offline_f1']['std'])}, "
            f"miss mean/std={_format_cell(agg['replay_miss_rate_pct']['mean'])}/{_format_cell(agg['replay_miss_rate_pct']['std'])}, "
            f"p99 mean/std={_format_cell(agg['replay_p99_ms']['mean'])}/{_format_cell(agg['replay_p99_ms']['std'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
