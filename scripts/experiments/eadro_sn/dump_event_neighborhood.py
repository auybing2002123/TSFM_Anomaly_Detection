from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print compact event neighborhoods for temporal error analysis.")
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--steps", required=True, help="Comma-separated center steps.")
    parser.add_argument("--radius", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.events_jsonl).read_text(encoding="utf-8").splitlines() if line]
    by_step = {int(row["step"]): row for row in rows}
    centers = [int(item) for item in args.steps.split(",") if item.strip()]
    for center in centers:
        print(f"\n# around step {center}")
        print("step label pred raw win top1 top2 top3 root n_pred source")
        for step in range(center - args.radius, center + args.radius + 1):
            row = by_step.get(step)
            if row is None:
                continue
            top = [float(item["score"]) for item in row.get("top_services", [])[:3]]
            while len(top) < 3:
                top.append(0.0)
            print(
                f"{step:03d} "
                f"{int(bool(row['window_anomaly_label']))} "
                f"{int(bool(row['window_anomaly_prediction']))} "
                f"{int(bool(row.get('raw_window_anomaly_prediction', row['window_anomaly_prediction'])))} "
                f"{float(row['window_score']):.4f} "
                f"{top[0]:.4f} {top[1]:.4f} {top[2]:.4f} "
                f"{float(row.get('root_cause_score', 0.0)):.4f} "
                f"{int(row.get('num_predicted_anomalies', 0))} "
                f"{row['source_id'].rsplit('_w', 1)[-1]}"
            )


if __name__ == "__main__":
    main()
