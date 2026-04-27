from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused top3_mean temporal search near the 0.979 candidate.")
    parser.add_argument("--events-jsonl", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def read_events(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    scores = []
    for row in rows:
        top = [float(item["score"]) for item in row.get("top_services", [])[:3]]
        while len(top) < 3:
            top.append(0.0)
        scores.append(sum(top) / 3.0)
    labels = np.asarray([int(bool(row["window_anomaly_label"])) for row in rows], dtype=np.int64)
    cases = [case_id(str(row["source_id"])) for row in rows]
    return np.asarray(scores, dtype=np.float64), labels, cases


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


def apply_rule(scores: np.ndarray, cases: list[str], spec: dict) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    active_count = 0
    inactive_count = 10**9
    case_pos = 0
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            active_count = 0
            inactive_count = 10**9
            case_pos = 0
            prev_case = cases[idx]

        base_hit = int(score >= spec["base"])
        prev_base_hits = sum(history)
        history.append(base_hit)
        if len(history) > 3:
            history = history[-3:]

        confirmed = sum(history) >= 2
        high = score >= spec["high"]
        if spec["kind"] == "high_after_base":
            high = high and (case_pos == 0 or prev_base_hits > 0)
        start = False
        if spec["kind"] == "start_gap":
            start = score >= spec["start"] and inactive_count >= spec["min_inactive"]
        pred = bool(high or confirmed or start)

        if pred:
            active_count += 1
            max_active = spec["max_active"]
            if max_active > 0 and active_count > max_active:
                extend = False
                if spec["kind"] == "extend_high":
                    extend = score >= spec["extend"]
                if not extend:
                    pred = False
        elif score < spec["base"]:
            active_count = 0

        preds[idx] = int(pred)
        inactive_count = 0 if pred else inactive_count + 1
        case_pos += 1
    return preds


def iter_specs() -> list[dict]:
    specs: list[dict] = []
    for base in np.linspace(0.285, 0.325, 21):
        for high in np.linspace(0.44, 0.52, 33):
            if high <= base:
                continue
            for max_active in (23, 24):
                specs.append(
                    {
                        "kind": "base",
                        "base": float(base),
                        "high": float(high),
                        "max_active": max_active,
                    }
                )
                specs.append(
                    {
                        "kind": "high_after_base",
                        "base": float(base),
                        "high": float(high),
                        "max_active": max_active,
                    }
                )
                for start in np.linspace(0.34, min(float(high), 0.46), 7):
                    if start <= base:
                        continue
                    for min_inactive in (4, 6, 8, 10, 12):
                        specs.append(
                            {
                                "kind": "start_gap",
                                "base": float(base),
                                "high": float(high),
                                "start": float(start),
                                "min_inactive": min_inactive,
                                "max_active": max_active,
                            }
                        )
                for extend in np.linspace(0.48, 0.58, 11):
                    specs.append(
                        {
                            "kind": "extend_high",
                            "base": float(base),
                            "high": float(high),
                            "extend": float(extend),
                            "max_active": max_active,
                        }
                    )
    return specs


def compact(spec: dict, result: dict) -> dict:
    out = {
        "kind": spec["kind"],
        "base": round(spec["base"], 6),
        "high": round(spec["high"], 6),
        "max_active": spec["max_active"],
        **result,
    }
    if "start" in spec:
        out["start"] = round(spec["start"], 6)
        out["min_inactive"] = spec["min_inactive"]
    if "extend" in spec:
        out["extend"] = round(spec["extend"], 6)
    return out


def main() -> None:
    args = parse_args()
    scores, labels, cases = read_events(Path(args.events_jsonl))
    rows = []
    for spec in iter_specs():
        rows.append((spec, metrics(apply_rule(scores, cases, spec), labels)))
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
    print(json.dumps([compact(spec, result) for spec, result in rows[: args.top_k]], indent=2))


if __name__ == "__main__":
    main()
