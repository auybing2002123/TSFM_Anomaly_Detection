from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe guarded top3_mean high-trigger rules.")
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--base", type=float, default=None)
    parser.add_argument("--high", type=float, default=None)
    parser.add_argument("--guard-top3", type=float, default=None)
    return parser.parse_args()


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def read_events(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    top = []
    for row in rows:
        values = [float(item["score"]) for item in row.get("top_services", [])[:3]]
        while len(values) < 3:
            values.append(0.0)
        top.append(values)
    top_arr = np.asarray(top, dtype=np.float64)
    return {
        "score": top_arr.mean(axis=1),
        "top3": top_arr[:, 2],
        "labels": np.asarray([int(bool(row["window_anomaly_label"])) for row in rows], dtype=np.int64),
        "cases": [case_id(str(row["source_id"])) for row in rows],
    }


def metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def apply_rule(split: dict, spec: dict) -> np.ndarray:
    scores = split["score"]
    top3 = split["top3"]
    cases = split["cases"]
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    active_count = 0
    case_pos = 0
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            active_count = 0
            case_pos = 0
            prev_case = cases[idx]

        base_hit = int(score >= spec["base"])
        prev_base_hits = sum(history)
        history.append(base_hit)
        if len(history) > 3:
            history = history[-3:]

        high = score >= spec["high"]
        if high and prev_base_hits == 0 and case_pos > 0:
            high = top3[idx] >= spec["guard_top3"]
        pred = bool(high or sum(history) >= 2)

        if pred:
            active_count += 1
            if active_count > spec["max_active"]:
                pred = False
        elif score < spec["base"]:
            active_count = 0

        preds[idx] = int(pred)
        case_pos += 1
    return preds


def iter_specs() -> list[dict]:
    specs = []
    for base in np.linspace(0.295, 0.315, 21):
        for high in np.linspace(0.4625, 0.49, 23):
            if high <= base:
                continue
            for guard_top3 in np.linspace(0.48, 0.54, 25):
                specs.append(
                    {
                        "base": float(base),
                        "high": float(high),
                        "guard_top3": float(guard_top3),
                        "max_active": 24,
                    }
                )
    return specs


def main() -> None:
    args = parse_args()
    split = read_events(Path(args.events_jsonl))
    if args.base is not None and args.high is not None and args.guard_top3 is not None:
        spec = {
            "base": float(args.base),
            "high": float(args.high),
            "guard_top3": float(args.guard_top3),
            "max_active": 24,
        }
        print(json.dumps({**spec, **metrics(apply_rule(split, spec), split["labels"])}, indent=2))
        return

    rows = []
    for spec in iter_specs():
        result = metrics(apply_rule(split, spec), split["labels"])
        rows.append((spec, result))
    rows.sort(
        key=lambda row: (
            row[1]["f1"],
            row[1]["precision"],
            row[1]["recall"],
            -row[1]["fp"],
            -row[1]["fn"],
        ),
        reverse=True,
    )
    payload = []
    for spec, result in rows[: args.top_k]:
        payload.append(
            {
                "base": round(spec["base"], 6),
                "high": round(spec["high"], 6),
                "guard_top3": round(spec["guard_top3"], 6),
                "max_active": spec["max_active"],
                **result,
            }
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
