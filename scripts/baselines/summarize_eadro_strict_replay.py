from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.eadro_strict_replay_common import ensure_dir, load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the latest Eadro-SN strict replay summaries into one JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/baselines/eadro_strict_replay",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/baselines/eadro_strict_replay/summary/eadro_strict_replay_summary.json",
    )
    return parser.parse_args()


def _pick_latest(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_path = (PROJECT_ROOT / args.output_path).resolve()
    ensure_dir(output_path.parent)

    candidates = [
        path for path in input_dir.glob("eadro_strict_*_summary.json")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No strict replay summaries found in: {input_dir}")

    grouped: dict[str, list[Path]] = {}
    for path in candidates:
        payload = load_json(path)
        baseline = str(payload.get("run", {}).get("baseline", "unknown"))
        grouped.setdefault(baseline, []).append(path)

    rows = []
    for baseline, paths in sorted(grouped.items()):
        latest_path = _pick_latest(paths)
        payload = load_json(latest_path)
        response = payload["response_latency"]["response_time_ms"]
        replay_metrics = payload["replay_detection_metrics"]
        rows.append(
            {
                "baseline": baseline,
                "summary_path": str(latest_path),
                "device": payload["run"]["device"],
                "mode": payload["mode"],
                "interval_ms": float(payload["interval_ms"]),
                "deadline_ms": float(payload["deadline_ms"]),
                "num_steps": int(payload["num_steps"]),
                "deadline_miss_rate_pct": float(payload["deadline_miss_rate_pct"]),
                "response_mean_ms": float(response["mean_ms"]),
                "response_p95_ms": float(response["p95_ms"]),
                "response_p99_ms": float(response["p99_ms"]),
                "response_max_ms": float(response["max_ms"]),
                "replay_f1": float(replay_metrics["f1"]),
                "replay_precision": float(replay_metrics["precision"]),
                "replay_recall": float(replay_metrics["recall"]),
                "replay_accuracy": float(replay_metrics["accuracy"]),
                "offline_test_f1": float(payload["offline_test_metrics"]["f1"]),
                "offline_test_precision": float(payload["offline_test_metrics"]["precision"]),
                "offline_test_recall": float(payload["offline_test_metrics"]["recall"]),
                "offline_test_accuracy": float(payload["offline_test_metrics"]["accuracy"]),
                "representation": payload["baseline_metadata"]["representation"],
            }
        )

    aggregate = {
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "replay_alignment_note": "Resident-model paced replay for strict baselines currently covers TraceAnomaly / TranAD / Anomaly Transformer / GDN-style / GDN-official when available. Routing diagnostics are N/A for non-MoE baselines.",
        "rows": rows,
    }
    save_json(output_path, aggregate)
    print(f"Saved replay aggregate summary to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
