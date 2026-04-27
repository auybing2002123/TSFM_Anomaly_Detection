from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast local search over saved realtime event window scores.")
    parser.add_argument("--events-jsonl", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


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


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def confirm_or_high(
    scores: np.ndarray,
    cases: list[str],
    base_threshold: float,
    high_threshold: float,
    window: int,
    require: int,
) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            prev_case = cases[idx]
        history.append(int(score >= base_threshold))
        if len(history) > window:
            history = history[-window:]
        preds[idx] = int(score >= high_threshold or sum(history) >= require)
    return preds


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


def confirm_with_start(
    scores: np.ndarray,
    cases: list[str],
    base_threshold: float,
    start_threshold: float,
    window: int,
    require: int,
    cooldown: int,
) -> np.ndarray:
    preds = np.zeros(scores.shape[0], dtype=np.int64)
    history: list[int] = []
    since_positive = 10**9
    prev_case: str | None = None
    for idx, score in enumerate(scores):
        if cases[idx] != prev_case:
            history = []
            since_positive = 10**9
            prev_case = cases[idx]
        history.append(int(score >= base_threshold))
        if len(history) > window:
            history = history[-window:]
        confirmed = sum(history) >= require
        start = score >= start_threshold and since_positive >= cooldown
        pred = confirmed or start
        preds[idx] = int(pred)
        since_positive = 0 if pred else since_positive + 1
    return preds


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.events_jsonl).read_text(encoding="utf-8").splitlines() if line]
    scores = np.asarray([float(row["window_score"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(bool(row["window_anomaly_label"])) for row in rows], dtype=np.int64)
    cases = [case_id(str(row["source_id"])) for row in rows]

    results = []
    # Focus around the known 0.9513 candidate. Wider searches should use the
    # split-level calibration script, not this quick event-log probe.
    for base in np.linspace(0.335, 0.395, 25):
        for high in np.linspace(0.52, 0.72, 81):
            if high <= base:
                continue
            for window, require in ((2, 2), (3, 2)):
                preds = confirm_or_high(scores, cases, float(base), float(high), window, require)
                metrics = binary_metrics(preds, labels)
                results.append(
                    {
                        "kind": "confirm_or_high",
                        "base_threshold": float(base),
                        "high_threshold": float(high),
                        "window": window,
                        "require": require,
                        **metrics,
                    }
                )
                for max_active in range(20, 31):
                    preds = confirm_or_high_maxlen(
                        scores,
                        cases,
                        float(base),
                        float(high),
                        window,
                        require,
                        max_active,
                    )
                    metrics = binary_metrics(preds, labels)
                    results.append(
                        {
                            "kind": "confirm_or_high_maxlen",
                            "base_threshold": float(base),
                            "high_threshold": float(high),
                            "window": window,
                            "require": require,
                            "max_active": max_active,
                            **metrics,
                        }
                    )
            for cooldown in (4, 6, 8, 12):
                preds = confirm_with_start(scores, cases, float(base), float(high), 2, 2, cooldown)
                metrics = binary_metrics(preds, labels)
                results.append(
                    {
                        "kind": "confirm_with_start_cooldown",
                        "base_threshold": float(base),
                        "start_threshold": float(high),
                        "window": 2,
                        "require": 2,
                        "cooldown": cooldown,
                        **metrics,
                    }
                )

    results.sort(
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            -item["fp"],
            -item["fn"],
        ),
        reverse=True,
    )
    print(json.dumps(results[: args.top_k], indent=2))


if __name__ == "__main__":
    main()
