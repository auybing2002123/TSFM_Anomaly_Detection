from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a validation-only score ensemble for Eadro-SN strict baselines. "
            "Weights and threshold are selected on val only; test is evaluated once."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Baseline summary path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Grid step for simplex weights.",
    )
    parser.add_argument(
        "--normalization",
        choices=["rank", "zscore", "minmax"],
        default="rank",
        help="Normalize each oriented score stream using validation statistics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "eadro_strict_score_ensemble" / "summary.json",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_scores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["id"])
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return ids, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def orient_scores(scores: np.ndarray, direction: str) -> np.ndarray:
    if direction == "high_is_abnormal":
        return scores.astype(np.float64)
    if direction == "low_is_abnormal":
        return -scores.astype(np.float64)
    raise ValueError(f"Unsupported direction: {direction}")


def normalize_scores(val_scores: np.ndarray, test_scores: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "rank":
        ref = np.sort(val_scores)
        denom = max(len(ref) - 1, 1)
        val_norm = np.searchsorted(ref, val_scores, side="right") / denom
        test_norm = np.searchsorted(ref, test_scores, side="right") / denom
        return val_norm.astype(np.float64), test_norm.astype(np.float64)

    if mode == "zscore":
        center = float(val_scores.mean())
        scale = float(val_scores.std() + 1e-12)
        return (val_scores - center) / scale, (test_scores - center) / scale

    if mode == "minmax":
        min_v = float(val_scores.min())
        max_v = float(val_scores.max())
        scale = max(max_v - min_v, 1e-12)
        return (val_scores - min_v) / scale, (test_scores - min_v) / scale

    raise ValueError(f"Unsupported normalization: {mode}")


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(scores):
        metrics = compute_metrics(scores, labels, float(threshold))
        candidate = {"threshold": float(threshold), "metrics": metrics}
        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = candidate
    if best is None:
        raise ValueError("No threshold candidates found")
    return best


def simplex_weights(num_items: int, step: float) -> list[np.ndarray]:
    if num_items < 2:
        return [np.ones(1, dtype=np.float64)]
    units = int(round(1.0 / step))
    weights: list[np.ndarray] = []
    for raw in product(range(units + 1), repeat=num_items - 1):
        last = units - sum(raw)
        if last < 0:
            continue
        weights.append(np.asarray([*raw, last], dtype=np.float64) / units)
    return weights


def load_stream(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    artifacts = summary["artifacts"]
    val_ids, val_labels, val_scores = read_scores(resolve_path(artifacts["val_score_csv"]))
    test_ids, test_labels, test_scores = read_scores(resolve_path(artifacts["test_score_csv"]))
    direction = summary["validation_selection"]["direction"]
    return {
        "name": f"{summary.get('baseline', summary_path.parent.name)}:{summary.get('representation', '')}",
        "summary_path": str(summary_path),
        "direction": direction,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "val_scores": orient_scores(val_scores, direction),
        "test_scores": orient_scores(test_scores, direction),
    }


def ensure_aligned(streams: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    val_ids = streams[0]["val_ids"]
    test_ids = streams[0]["test_ids"]
    val_labels = streams[0]["val_labels"]
    test_labels = streams[0]["test_labels"]
    for stream in streams:
        if set(stream["val_ids"]) != set(val_ids) or set(stream["test_ids"]) != set(test_ids):
            raise ValueError("Score CSV id sets differ across streams")

        val_payload = {
            sample_id: (int(label), float(score))
            for sample_id, label, score in zip(stream["val_ids"], stream["val_labels"], stream["val_scores"])
        }
        test_payload = {
            sample_id: (int(label), float(score))
            for sample_id, label, score in zip(stream["test_ids"], stream["test_labels"], stream["test_scores"])
        }
        stream["val_scores_aligned"] = np.asarray([val_payload[sample_id][1] for sample_id in val_ids], dtype=np.float64)
        stream["test_scores_aligned"] = np.asarray([test_payload[sample_id][1] for sample_id in test_ids], dtype=np.float64)

        stream_val_labels = np.asarray([val_payload[sample_id][0] for sample_id in val_ids], dtype=np.int64)
        stream_test_labels = np.asarray([test_payload[sample_id][0] for sample_id in test_ids], dtype=np.int64)
        if not np.array_equal(stream_val_labels, val_labels) or not np.array_equal(stream_test_labels, test_labels):
            raise ValueError("Score CSV labels differ across streams")
    return val_labels, test_labels


def main() -> None:
    args = parse_args()
    streams = [load_stream(resolve_path(path)) for path in args.summary]
    val_labels, test_labels = ensure_aligned(streams)

    val_matrix = []
    test_matrix = []
    for stream in streams:
        val_norm, test_norm = normalize_scores(
            stream["val_scores_aligned"],
            stream["test_scores_aligned"],
            args.normalization,
        )
        val_matrix.append(val_norm)
        test_matrix.append(test_norm)
    val_matrix_np = np.stack(val_matrix, axis=1)
    test_matrix_np = np.stack(test_matrix, axis=1)

    best: dict[str, Any] | None = None
    for weights in simplex_weights(len(streams), args.weight_step):
        val_ensemble = val_matrix_np @ weights
        val_selection = select_threshold(val_ensemble, val_labels)
        candidate = {
            "weights": weights,
            "val_selection": val_selection,
            "val_scores": val_ensemble,
        }
        if best is None or val_selection["metrics"]["f1"] > best["val_selection"]["metrics"]["f1"]:
            best = candidate
    if best is None:
        raise RuntimeError("No ensemble candidate found")

    test_scores = test_matrix_np @ best["weights"]
    test_result = {
        "threshold": best["val_selection"]["threshold"],
        "metrics": compute_metrics(test_scores, test_labels, best["val_selection"]["threshold"]),
    }

    summary = {
        "baseline": "Eadro strict score ensemble",
        "protocol": "strict case-level split + val-only weight/threshold selection + test-final",
        "normalization": args.normalization,
        "weight_step": args.weight_step,
        "streams": [
            {
                "name": stream["name"],
                "summary_path": stream["summary_path"],
                "direction": stream["direction"],
            }
            for stream in streams
        ],
        "selected_weights": {
            streams[index]["name"]: float(weight)
            for index, weight in enumerate(best["weights"].tolist())
        },
        "validation_selection": best["val_selection"],
        "test_result": test_result,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    val_metrics = best["val_selection"]["metrics"]
    test_metrics = test_result["metrics"]
    print("Eadro strict score ensemble finished")
    print(f"Normalization: {args.normalization}, streams={len(streams)}")
    print(f"Selected weights: {summary['selected_weights']}")
    print(
        f"Val F1={val_metrics['f1']:.4f}, P={val_metrics['precision']:.4f}, "
        f"R={val_metrics['recall']:.4f}, Acc={val_metrics['accuracy']:.4f}"
    )
    print(
        f"Test F1={test_metrics['f1']:.4f}, P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, Acc={test_metrics['accuracy']:.4f}"
    )
    print(f"Summary: {args.output_json}")


if __name__ == "__main__":
    main()
