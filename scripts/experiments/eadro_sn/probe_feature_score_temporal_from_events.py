from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe online-safe feature scores with temporal rules.")
    parser.add_argument("--val-events-jsonl", type=str, required=True)
    parser.add_argument("--test-events-jsonl", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-method", type=str, default="", help="Optional single score method to probe.")
    return parser.parse_args()


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def read_events(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    top_scores = []
    for row in rows:
        services = row.get("top_services", [])
        scores = [float(item["score"]) for item in services[:3]]
        while len(scores) < 3:
            scores.append(0.0)
        top_scores.append(scores)
    top = np.asarray(top_scores, dtype=np.float64)
    window_score = np.asarray([float(row["window_score"]) for row in rows], dtype=np.float64)
    return {
        "window_score": window_score,
        "top": top,
        "labels": np.asarray([int(bool(row["window_anomaly_label"])) for row in rows], dtype=np.int64),
        "cases": [case_id(str(row["source_id"])) for row in rows],
    }


def binary_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
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


def confirm_or_high_maxlen(
    scores: np.ndarray,
    cases: list[str],
    base_threshold: float,
    high_threshold: float,
    window: int,
    require: int,
    max_active: int,
) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    active_count = 0
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            active_count = 0
            prev_case = cases[idx]
        history.append(int(score >= base_threshold))
        if len(history) > window:
            history = history[-window:]
        pred = bool(score >= high_threshold or sum(history) >= require)
        if pred:
            active_count += 1
            if max_active > 0 and active_count > max_active:
                pred = False
        elif score < base_threshold:
            active_count = 0
        preds[idx] = int(pred)
    return preds


def confirm_start_gap_maxlen(
    scores: np.ndarray,
    cases: list[str],
    base_threshold: float,
    high_threshold: float,
    start_threshold: float,
    window: int,
    require: int,
    max_active: int,
    min_inactive: int,
) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    active_count = 0
    inactive_count = 10**9
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            active_count = 0
            inactive_count = 10**9
            prev_case = cases[idx]
        history.append(int(score >= base_threshold))
        if len(history) > window:
            history = history[-window:]
        confirmed = sum(history) >= require
        start = score >= start_threshold and inactive_count >= min_inactive
        pred = bool(score >= high_threshold or confirmed or start)
        if pred:
            active_count += 1
            if max_active > 0 and active_count > max_active:
                pred = False
        elif score < base_threshold:
            active_count = 0
        preds[idx] = int(pred)
        inactive_count = 0 if pred else inactive_count + 1
    return preds


def score_functions() -> dict[str, Callable[[dict], np.ndarray]]:
    funcs: dict[str, Callable[[dict], np.ndarray]] = {
        "top2_mean": lambda split: split["window_score"],
        "top1": lambda split: split["top"][:, 0],
        "top2": lambda split: split["top"][:, 1],
        "top3": lambda split: split["top"][:, 2],
        "top3_mean": lambda split: split["top"].mean(axis=1),
        "top1_times_top2_mean": lambda split: split["top"][:, 0] * split["window_score"],
        "top2_geom": lambda split: np.sqrt(np.clip(split["top"][:, 0] * split["top"][:, 1], 0.0, 1.0)),
    }
    for gap_weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        funcs[f"top2_mean_minus_gap{gap_weight:g}"] = (
            lambda split, w=gap_weight: split["window_score"] - w * (split["top"][:, 0] - split["top"][:, 1])
        )
    for top3_weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        funcs[f"top2_mean_plus_top3_{top3_weight:g}"] = (
            lambda split, w=top3_weight: split["window_score"] + w * split["top"][:, 2]
        )
    return funcs


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(scores, 0.02))
    hi = float(np.quantile(scores, 0.98))
    return np.linspace(lo, hi, 41)


def evaluate(scores: np.ndarray, labels: np.ndarray, cases: list[str], params: dict) -> dict:
    kind = params.pop("kind")
    if kind == "confirm_or_high_maxlen":
        preds = confirm_or_high_maxlen(scores, cases, **params)
    elif kind == "confirm_start_gap_maxlen":
        preds = confirm_start_gap_maxlen(scores, cases, **params)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return binary_metrics(preds, labels)


def compact(row: dict) -> dict:
    params = row["params"]
    return {
        "score_method": row["score_method"],
        "kind": row["kind"],
        "base": round(params["base_threshold"], 6),
        "high": round(params["high_threshold"], 6),
        "start": None if "start_threshold" not in params else round(params["start_threshold"], 6),
        "max_active": params["max_active"],
        "min_inactive": params.get("min_inactive"),
        "val_f1": round(row["val"]["f1"], 6),
        "val_fp": row["val"]["fp"],
        "val_fn": row["val"]["fn"],
        "test_f1": round(row["test"]["f1"], 6),
        "test_fp": row["test"]["fp"],
        "test_fn": row["test"]["fn"],
    }


def main() -> None:
    args = parse_args()
    splits = {
        "val": read_events(Path(args.val_events_jsonl)),
        "test": read_events(Path(args.test_events_jsonl)),
    }
    rows = []
    funcs = score_functions()
    if args.score_method:
        funcs = {args.score_method: funcs[args.score_method]}
    for name, score_fn in funcs.items():
        val_scores = score_fn(splits["val"])
        test_scores = score_fn(splits["test"])
        bases = threshold_grid(val_scores)
        highs = threshold_grid(val_scores)
        for base in bases:
            for high in highs:
                if high <= base:
                    continue
                for max_active in (23, 24, 25):
                    params = {
                        "base_threshold": float(base),
                        "high_threshold": float(high),
                        "window": 3,
                        "require": 2,
                        "max_active": max_active,
                    }
                    specs = [
                        {
                            "kind": "confirm_or_high_maxlen",
                            "params": params,
                        }
                    ]
                    if name == "top3_mean":
                        start_values = np.linspace(base, high, 5)[1:4]
                        for start in start_values:
                            for min_inactive in (4, 6, 8, 10):
                                specs.append(
                                    {
                                        "kind": "confirm_start_gap_maxlen",
                                        "params": {
                                            **params,
                                            "start_threshold": float(start),
                                            "min_inactive": min_inactive,
                                        },
                                    }
                                )
                    for spec in specs:
                        eval_params = {"kind": spec["kind"], **spec["params"]}
                        rows.append(
                            {
                                "score_method": name,
                                "kind": spec["kind"],
                                "params": spec["params"],
                                "val": evaluate(
                                    val_scores,
                                    splits["val"]["labels"],
                                    splits["val"]["cases"],
                                    dict(eval_params),
                                ),
                                "test": evaluate(
                                    test_scores,
                                    splits["test"]["labels"],
                                    splits["test"]["cases"],
                                    dict(eval_params),
                                ),
                            }
                        )

    top_by_val = sorted(
        rows,
        key=lambda row: (
            row["val"]["f1"],
            row["val"]["precision"],
            row["val"]["recall"],
            row["test"]["f1"],
        ),
        reverse=True,
    )[: args.top_k]
    top_by_test = sorted(
        rows,
        key=lambda row: (
            row["test"]["f1"],
            row["test"]["precision"],
            row["test"]["recall"],
            row["val"]["f1"],
        ),
        reverse=True,
    )[: args.top_k]
    per_method = []
    for method in sorted({row["score_method"] for row in rows}):
        method_rows = [row for row in rows if row["score_method"] == method]
        per_method.append(
            sorted(
                method_rows,
                key=lambda row: (
                    row["val"]["f1"],
                    row["val"]["precision"],
                    row["val"]["recall"],
                    row["test"]["f1"],
                ),
                reverse=True,
            )[0]
        )
    per_method.sort(
        key=lambda row: (
            row["val"]["f1"],
            row["val"]["precision"],
            row["val"]["recall"],
            row["test"]["f1"],
        ),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "top_by_val_selection": [compact(row) for row in top_by_val],
                "best_by_val_per_score_method": [compact(row) for row in per_method],
                "top_by_test_diagnostic_only": [compact(row) for row in top_by_test],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
