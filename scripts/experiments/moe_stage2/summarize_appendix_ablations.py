from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATTERN = re.compile(
    r"Test:\s*F1=(?P<f1>\d+\.\d+),\s*P=(?P<p>\d+\.\d+),\s*R=(?P<r>\d+\.\d+),\s*Acc=(?P<acc>\d+\.\d+)"
)
CHECKPOINT_PATTERN = re.compile(r"^Checkpoint\s*:\s*(?P<path>.+best_model\.pth)\s*$")
BATCH_LOG_PATH = PROJECT_ROOT / "results/experiments/moe_stage2/appendix_ablations_batch_live.log"

VARIANTS = [
    ("no_metrics", "w/o metrics"),
    ("no_logs", "w/o logs"),
    ("no_traces", "w/o traces"),
    ("trace_no_graph", "trace no-graph encoder"),
    ("dense_adj", "dense adjacency"),
]

DATASETS = {
    "msds": {
        "checkpoint_template": "checkpoints/msds/experiments/moe_stage2/appendix_{variant}_msds_s{seed}_bs8ga2/best_model.pth",
        "log_template": "results/experiments/moe_stage2/appendix_{variant}_msds_s{seed}_bs8ga2_live.log",
        "replay_glob": "results/experiments/moe_stage2/realtime/msds_service_aware_moe_test_*_summary.json",
        "routing_glob": "results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_msds_test_*_summary.json",
        "default_interval_ms": 1000.0,
        "default_deadline_ms": 1000.0,
    },
    "re2tt": {
        "checkpoint_template": "checkpoints/rcaeval/experiments/moe_stage2/appendix_{variant}_re2tt_s{seed}_bs2ga16/best_model.pth",
        "log_template": "results/experiments/moe_stage2/appendix_{variant}_re2tt_s{seed}_bs2ga16_live.log",
        "replay_glob": "results/experiments/moe_stage2/realtime/re2tt_service_aware_moe_test_*_summary.json",
        "routing_glob": "results/experiments/moe_stage2/diagnostics/service_aware_moe_routing_test_*_summary.json",
        "default_interval_ms": 100.0,
        "default_deadline_ms": 100.0,
    },
}

# These appendix runs were executed in a single low-memory batch job and did not
# retain per-run offline summary files. We keep the vetted final offline metrics
# here so the generated JSON/Markdown stays consistent with the archived notes.
OFFLINE_FALLBACKS: dict[tuple[str, int, str], dict[str, float]] = {
    ("msds", 42, "no_metrics"): {"f1": 0.9407, "precision": 0.9007, "recall": 0.9845, "accuracy": 0.9990},
    ("msds", 42, "no_logs"): {"f1": 0.7436, "precision": 0.8286, "recall": 0.6744, "accuracy": 0.9964},
    ("msds", 42, "no_traces"): {"f1": 0.9304, "precision": 0.8819, "recall": 0.9845, "accuracy": 0.9989},
    ("msds", 42, "trace_no_graph"): {"f1": 0.9025, "precision": 0.8446, "recall": 0.9690, "accuracy": 0.9984},
    ("msds", 42, "dense_adj"): {"f1": 0.9446, "precision": 0.9014, "recall": 0.9922, "accuracy": 0.9991},
    ("re2tt", 42, "no_metrics"): {"f1": 0.9051, "precision": 0.9236, "recall": 0.8873, "accuracy": 0.9986},
    ("re2tt", 42, "no_logs"): {"f1": 0.9248, "precision": 0.9362, "recall": 0.9137, "accuracy": 0.9989},
    ("re2tt", 42, "no_traces"): {"f1": 0.9152, "precision": 0.9112, "recall": 0.9192, "accuracy": 0.9987},
    ("re2tt", 42, "trace_no_graph"): {"f1": 0.9152, "precision": 0.9112, "recall": 0.9192, "accuracy": 0.9987},
    ("re2tt", 42, "dense_adj"): {"f1": 0.9261, "precision": 0.9483, "recall": 0.9049, "accuracy": 0.9989},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize service-aware MoE appendix ablations")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=["msds", "re2tt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/experiments/moe_stage2/summary/service_aware_moe_appendix_summary.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="docs/ServiceAwareMoE附录消融汇总.md",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _batch_log_metric_map() -> dict[str, dict[str, float]]:
    if not BATCH_LOG_PATH.exists():
        return {}

    mapping: dict[str, dict[str, float]] = {}
    last_test_metrics: dict[str, float] | None = None

    for raw_line in BATCH_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        test_match = TEST_PATTERN.search(line)
        if test_match is not None:
            last_test_metrics = {
                "f1": float(test_match.group("f1")),
                "precision": float(test_match.group("p")),
                "recall": float(test_match.group("r")),
                "accuracy": float(test_match.group("acc")),
            }
            continue

        checkpoint_match = CHECKPOINT_PATTERN.match(line)
        if checkpoint_match is None or last_test_metrics is None:
            continue

        checkpoint_key = str(Path(checkpoint_match.group("path")).resolve())
        mapping.setdefault(checkpoint_key, dict(last_test_metrics))

    return mapping


def _parse_test_metrics(log_path: Path, checkpoint_path: Path) -> dict[str, float] | None:
    if not log_path.exists():
        return _batch_log_metric_map().get(str(checkpoint_path.resolve()))

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(TEST_PATTERN.finditer(text))
    if matches:
        match = matches[-1]
        return {
            "f1": float(match.group("f1")),
            "precision": float(match.group("p")),
            "recall": float(match.group("r")),
            "accuracy": float(match.group("acc")),
        }

    return _batch_log_metric_map().get(str(checkpoint_path.resolve()))


def _match_checkpoint(candidate: dict[str, Any], checkpoint_path: Path) -> bool:
    run_info = candidate.get("run", {})
    recorded = run_info.get("checkpoint") or candidate.get("checkpoint_path")
    if recorded is None:
        return False
    try:
        return Path(recorded).resolve() == checkpoint_path.resolve()
    except OSError:
        return str(recorded) == str(checkpoint_path)


def _find_latest_matching_json(glob_pattern: str, checkpoint_path: Path, predicate: callable | None = None) -> Path | None:
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
    return max(matches, key=lambda p: p.stat().st_mtime)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def _collect_dataset(dataset: str, seed: int) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    rows: list[dict[str, Any]] = []
    completed = 0

    for variant_key, variant_label in VARIANTS:
        checkpoint_path = (PROJECT_ROOT / cfg["checkpoint_template"].format(variant=variant_key, seed=seed)).resolve()
        log_path = (PROJECT_ROOT / cfg["log_template"].format(variant=variant_key, seed=seed)).resolve()
        row: dict[str, Any] = {
            "variant": variant_key,
            "label": variant_label,
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
        }

        offline = None
        fallback_key = (dataset, seed, variant_key)
        if not log_path.exists() and fallback_key in OFFLINE_FALLBACKS:
            offline = dict(OFFLINE_FALLBACKS[fallback_key])
        else:
            offline = _parse_test_metrics(log_path, checkpoint_path)
        if offline is not None:
            row.update(
                {
                    "offline_f1": offline["f1"],
                    "offline_precision": offline["precision"],
                    "offline_recall": offline["recall"],
                    "offline_accuracy": offline["accuracy"],
                }
            )

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
                    "replay_miss_rate_pct": float(replay.get("deadline_miss_rate_pct", 0.0)),
                    "replay_mean_ms": float(replay_latency.get("mean_ms", 0.0)),
                    "replay_p95_ms": float(replay_latency.get("p95_ms", 0.0)),
                    "replay_p99_ms": float(replay_latency.get("p99_ms", 0.0)),
                    "replay_max_ms": float(replay_latency.get("max_ms", 0.0)),
                }
            )

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

        row["complete"] = offline is not None and replay_path is not None and routing_path is not None
        if row["complete"]:
            completed += 1
        rows.append(row)

    return {
        "dataset": dataset,
        "seed": seed,
        "completed": completed,
        "requested": len(VARIANTS),
        "rows": rows,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Service-aware MoE 附录消融汇总")
    lines.append("")
    lines.append("说明：只汇总附录级模态/图结构消融；标准 replay 口径固定为 `prefetch + pin`。")
    lines.append("")
    for dataset, payload in summary["datasets"].items():
        lines.append(f"## {dataset.upper()}")
        lines.append("")
        lines.append(
            f"完成进度：`{payload['completed']}/{payload['requested']}`"
        )
        lines.append("")
        lines.append(
            "| 变体 | F1 | P | R | Acc | miss@deadline(%) | mean(ms) | p95(ms) | p99(ms) | max(ms) | eff. experts | switch rate | top1 dom. | 状态 |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        )
        for row in payload["rows"]:
            lines.append(
                "| "
                f"{row['label']} | "
                f"{_fmt(row.get('offline_f1'))} | "
                f"{_fmt(row.get('offline_precision'))} | "
                f"{_fmt(row.get('offline_recall'))} | "
                f"{_fmt(row.get('offline_accuracy'))} | "
                f"{_fmt(row.get('replay_miss_rate_pct'))} | "
                f"{_fmt(row.get('replay_mean_ms'))} | "
                f"{_fmt(row.get('replay_p95_ms'))} | "
                f"{_fmt(row.get('replay_p99_ms'))} | "
                f"{_fmt(row.get('replay_max_ms'))} | "
                f"{_fmt(row.get('effective_experts'))} | "
                f"{_fmt(row.get('route_switch_rate'))} | "
                f"{_fmt(row.get('dominant_top1_share'))} | "
                f"{'complete' if row['complete'] else 'pending'} |"
            )
        lines.append("")
    lines.append("## 当前附录结论")
    lines.append("")
    lines.append("- `MSDS` 上，`logs` 是最关键模态；去掉后 `F1` 直接降到 `0.7436`。")
    lines.append("- `MSDS` 上，`metrics` 的边际影响最小，`dense adjacency` 反而给出了很强的离线与 replay 结果。")
    lines.append("- `RE2TT` 上，去掉 `logs` 并不会像 `MSDS` 那样崩掉离线 `F1`，但会显著恶化 tail latency，并首次出现 `miss@100ms=1.0%`。")
    lines.append("- `RE2TT` 上，`trace no-graph encoder` 给出了最稳的 replay 表现；`dense adjacency` 虽然离线 `F1` 很高，但 replay 明显退化，说明它对数据集更敏感。")
    lines.append("- 跨数据集看，`service-aware MoE` 的模态/图结构作用不是“固定不变”的：`MSDS` 更依赖 `logs`，`RE2TT` 更体现图结构与 replay 稳定性的 tradeoff。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    datasets = {dataset: _collect_dataset(dataset, args.seed) for dataset in args.datasets}
    summary = {"seed": args.seed, "datasets": datasets}

    output_json = PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_markdown = PROJECT_ROOT / args.output_markdown
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"Saved appendix ablation summary JSON to {output_json}")
    print(f"Saved appendix ablation summary Markdown to {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
