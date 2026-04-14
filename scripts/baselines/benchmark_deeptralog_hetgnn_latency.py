from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPTRALOG_CODE_ROOT = PROJECT_ROOT.parents[1] / "external" / "DeepTraLog" / "HetGNN" / "code"
if str(DEEPTRALOG_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPTRALOG_CODE_ROOT))

from HetGNN import model_class  # type: ignore  # noqa: E402
from args import read_args  # type: ignore  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(statistics.mean(values)),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepTraLog HetGNN latency smoke benchmark")
    parser.add_argument("--data-path", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\ProcessedData"))
    parser.add_argument("--model-path", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke\HetGNN_0.pt"))
    parser.add_argument("--center", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke\HetGNN_SVDD_Center.pt"))
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    sys.argv = [
        "benchmark_deeptralog_hetgnn_latency.py",
        "--data_path", str(cli.data_path),
        "--model_path", str(cli.model_path),
        "--preprocess", "1",
        "--train_iter_n", "1",
        "--save_model_freq", "1",
        "--batch_s", str(cli.batch_size),
        "--mini_batch_s", str(cli.batch_size),
        "--embed_d", "7",
        "--cuda", "0",
    ]
    args = read_args()

    t0 = time.perf_counter()
    model_object = model_class(args)
    setup_ms = (time.perf_counter() - t0) * 1000.0

    model_object.model.load_state_dict(torch.load(cli.checkpoint, map_location="cpu"))
    model_object.model.svdd_center = torch.load(cli.center, map_location="cpu")
    model_object.model.eval()

    _, _, test_gid_list = model_object.train_eval_test_split()
    test_gid_list = np.asarray(test_gid_list)
    if cli.num_samples > 0:
        test_gid_list = test_gid_list[: cli.num_samples]

    batch_size = max(1, int(cli.batch_size))
    records: list[float] = []

    with torch.no_grad():
        for start in range(0, len(test_gid_list), batch_size):
            batch = test_gid_list[start : start + batch_size].tolist()
            if len(batch) == 0:
                continue
            if len(records) < cli.warmup:
                _ = model_object.model.predict_score(batch)
                records.append(-1.0)
                continue
            tic = time.perf_counter()
            _ = model_object.model.predict_score(batch)
            toc = time.perf_counter()
            records.append((toc - tic) * 1000.0)

    latency_values = [v for v in records if v >= 0]
    summary = {
        "setup_ms": setup_ms,
        "samples": len(latency_values),
        "batch_size": batch_size,
        "checkpoint": str(cli.checkpoint),
        "center": str(cli.center),
        "latency_summary": summarize(latency_values),
    }

    out_dir = PROJECT_ROOT / "results" / "baselines" / "deeptralog_hetgnn_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
