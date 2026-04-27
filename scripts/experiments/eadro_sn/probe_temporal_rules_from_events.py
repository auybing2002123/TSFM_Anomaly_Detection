from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe causal temporal rules over saved Eadro-SN event logs.")
    parser.add_argument("--val-events-jsonl", type=str, required=True)
    parser.add_argument("--test-events-jsonl", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def window_index(source_id: str, fallback: int) -> int:
    match = re.search(r"_w(\d+)$", source_id)
    return int(match.group(1)) if match else fallback


def read_events(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    top1_scores = []
    top2_scores = []
    for row in rows:
        top_services = row.get("top_services", [])
        first = float(top_services[0]["score"]) if len(top_services) >= 1 else float(row["window_score"])
        second = float(top_services[1]["score"]) if len(top_services) >= 2 else 0.0
        top1_scores.append(first)
        top2_scores.append(second)
    window_scores = np.asarray([float(row["window_score"]) for row in rows], dtype=np.float64)
    top1 = np.asarray(top1_scores, dtype=np.float64)
    top2 = np.asarray(top2_scores, dtype=np.float64)
    return {
        "score_methods": {
            "top2_mean": window_scores,
            "top1": top1,
            "top2": top2,
            "top1_times_top2_mean": top1 * window_scores,
        },
        "labels": np.asarray([int(bool(row["window_anomaly_label"])) for row in rows], dtype=np.int64),
        "cases": [case_id(str(row["source_id"])) for row in rows],
        "windows": [window_index(str(row["source_id"]), int(row["step"])) for row in rows],
        "source_ids": [str(row["source_id"]) for row in rows],
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
    """Confirm by local history, with an immediate high-score start after an inactive gap."""
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


def confirm_start_gap_maxlen_reset(
    scores: np.ndarray,
    cases: list[str],
    base_threshold: float,
    high_threshold: float,
    start_threshold: float,
    reset_threshold: float,
    window: int,
    require: int,
    max_active: int,
    min_inactive: int,
) -> np.ndarray:
    """Variant that resets active length after a sufficiently low score."""
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
        if score < reset_threshold:
            active_count = 0
        confirmed = sum(history) >= require
        start = score >= start_threshold and inactive_count >= min_inactive
        pred = bool(score >= high_threshold or confirmed or start)
        if pred:
            active_count += 1
            if max_active > 0 and active_count > max_active:
                pred = False
        preds[idx] = int(pred)
        inactive_count = 0 if pred else inactive_count + 1
    return preds


def eval_spec(spec: dict, split: dict) -> dict:
    kind = spec["kind"]
    params = spec["params"]
    scores = split["score_methods"][spec["score_method"]]
    if kind == "confirm_or_high_maxlen":
        preds = confirm_or_high_maxlen(scores, split["cases"], **params)
    elif kind == "confirm_start_gap_maxlen":
        preds = confirm_start_gap_maxlen(scores, split["cases"], **params)
    elif kind == "confirm_start_gap_maxlen_reset":
        preds = confirm_start_gap_maxlen_reset(scores, split["cases"], **params)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return binary_metrics(preds, split["labels"])


def iter_specs() -> list[dict]:
    specs = []
    score_grids = {
        "top2_mean": {
            "base": (0.35, 0.3525, 0.355, 0.36, 0.365, 0.3675, 0.37),
            "high": (0.59, 0.60, 0.61, 0.62, 0.63, 0.655, 0.67),
            "start": (0.52, 0.56, 0.60, 0.62),
        },
        "top1": {
            "base": (0.48, 0.50, 0.52, 0.54, 0.56),
            "high": (0.62, 0.65, 0.68, 0.70, 0.72, 0.75),
            "start": (0.60, 0.64, 0.68),
        },
        "top2": {
            "base": (0.22, 0.26, 0.30, 0.34, 0.38),
            "high": (0.42, 0.46, 0.50, 0.54, 0.58, 0.62),
            "start": (0.42, 0.48, 0.54),
        },
        "top1_times_top2_mean": {
            "base": (0.20, 0.24, 0.28, 0.32, 0.36),
            "high": (0.34, 0.38, 0.42, 0.46, 0.50),
            "start": (0.34, 0.40, 0.46),
        },
    }
    for score_method, grid in score_grids.items():
        for base in grid["base"]:
            for high in grid["high"]:
                if high <= base:
                    continue
                for window, require in ((3, 2),):
                    for max_active in (23, 24, 25):
                        params = {
                            "base_threshold": float(base),
                            "high_threshold": float(high),
                            "window": window,
                            "require": require,
                            "max_active": max_active,
                        }
                        specs.append(
                            {
                                "score_method": score_method,
                                "kind": "confirm_or_high_maxlen",
                                "params": params,
                            }
                        )
                        for start in grid["start"]:
                            if start <= base:
                                continue
                            for min_inactive in (2, 4, 6, 8):
                                specs.append(
                                    {
                                        "score_method": score_method,
                                        "kind": "confirm_start_gap_maxlen",
                                        "params": {
                                            **params,
                                            "start_threshold": float(start),
                                            "min_inactive": min_inactive,
                                        },
                                    }
                                )
                                for reset in (0.25, 0.30, 0.35):
                                    if reset >= base:
                                        continue
                                    specs.append(
                                        {
                                            "score_method": score_method,
                                            "kind": "confirm_start_gap_maxlen_reset",
                                            "params": {
                                                **params,
                                                "start_threshold": float(start),
                                                "reset_threshold": float(reset),
                                                "min_inactive": min_inactive,
                                            },
                                        }
                                    )
    return specs


def compact(row: dict) -> dict:
    params = row["params"]
    return {
        "score_method": row["score_method"],
        "kind": row["kind"],
        "base": round(params["base_threshold"], 4),
        "high": round(params["high_threshold"], 4),
        "start": None if "start_threshold" not in params else round(params["start_threshold"], 4),
        "reset": None if "reset_threshold" not in params else round(params["reset_threshold"], 4),
        "window": params["window"],
        "require": params["require"],
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
    for spec in iter_specs():
        rows.append(
            {
                **spec,
                "val": eval_spec(spec, splits["val"]),
                "test": eval_spec(spec, splits["test"]),
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
    robust = sorted(
        rows,
        key=lambda row: (
            min(row["val"]["f1"], row["test"]["f1"]),
            row["val"]["f1"],
            row["test"]["f1"],
        ),
        reverse=True,
    )[: args.top_k]

    payload = {
        "top_by_val_selection": [compact(row) for row in top_by_val],
        "top_by_test_diagnostic_only": [compact(row) for row in top_by_test],
        "top_by_min_val_test_diagnostic": [compact(row) for row in robust],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
