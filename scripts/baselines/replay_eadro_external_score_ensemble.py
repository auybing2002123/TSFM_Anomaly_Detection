from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an external Eadro-SN strict score ensemble with resident models."
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "baselines"
        / "eadro_strict_score_ensemble"
        / "xgb_mltraces_svm64_zscore_summary.json",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "baselines"
        / "eadro_strict_score_ensemble"
        / "xgb_mltraces_svm64_zscore_replay_summary.json",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    return parser.parse_args()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_scores(path: Path, direction: str) -> np.ndarray:
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            scores.append(score if direction == "high_is_abnormal" else -score)
    return np.asarray(scores, dtype=np.float64)


def build_features(split: dict[str, np.ndarray], representation: str) -> np.ndarray:
    modalities = [item.strip() for item in representation.split("+") if item.strip()]
    parts = [
        split[modality].reshape(split[modality].shape[0], -1).astype(np.float32)
        for modality in modalities
    ]
    return np.concatenate(parts, axis=1)


def normalize(score: float, stats: dict[str, Any]) -> float:
    mode = stats["mode"]
    if mode == "zscore":
        return (score - stats["mean"]) / stats["std"]
    if mode == "minmax":
        return (score - stats["min"]) / stats["scale"]
    if mode == "rank":
        ref = stats["sorted"]
        return float(np.searchsorted(ref, score, side="right") / stats["denom"])
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_norm_stats(val_scores: np.ndarray, mode: str) -> dict[str, Any]:
    if mode == "zscore":
        return {"mode": mode, "mean": float(val_scores.mean()), "std": float(val_scores.std() + 1e-12)}
    if mode == "minmax":
        min_v = float(val_scores.min())
        max_v = float(val_scores.max())
        return {"mode": mode, "min": min_v, "scale": max(max_v - min_v, 1e-12)}
    if mode == "rank":
        ref = np.sort(val_scores)
        return {"mode": mode, "sorted": ref, "denom": max(len(ref) - 1, 1)}
    raise ValueError(f"Unsupported normalization mode: {mode}")


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
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


def latency_stats(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "std_ms": float(arr.std()) if arr.size else 0.0,
        "min_ms": float(arr.min()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "max_ms": float(arr.max()) if arr.size else 0.0,
    }


def load_stream(stream: dict[str, Any], normalization: str, test_split: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = load_json(resolve(stream["summary_path"]))
    val_scores = read_scores(resolve(summary["artifacts"]["val_score_csv"]), stream["direction"])
    norm_stats = build_norm_stats(val_scores, normalization)
    representation = summary["representation"]
    features = build_features(test_split, representation)

    if summary["baseline"] == "XGBoost":
        models = []
        for item in summary["artifacts"]["models"]:
            with resolve(item["checkpoint_pickle"]).open("rb") as handle:
                models.append(pickle.load(handle))

        def predict_score(row: np.ndarray) -> float:
            return float(np.mean([model.predict_proba(row)[:, 1] for model in models], axis=0)[0])

    elif summary["baseline"] == "RBF-SVM ensemble":
        with resolve(summary["artifacts"]["checkpoint_pickle"]).open("rb") as handle:
            models = pickle.load(handle)

        def predict_score(row: np.ndarray) -> float:
            return float(np.mean([model.decision_function(row) for model in models], axis=0)[0])

    else:
        raise ValueError(f"Unsupported resident replay baseline: {summary['baseline']}")

    return {
        "name": stream["name"],
        "direction": stream["direction"],
        "weight": float(stream["weight"]),
        "norm_stats": norm_stats,
        "features": features,
        "predict_score": predict_score,
        "summary_path": stream["summary_path"],
    }


def main() -> None:
    args = parse_args()
    ensemble = load_json(args.ensemble_summary)
    test_split = dict(np.load(args.export_dir / "test.npz", allow_pickle=True))
    labels = test_split["window_labels"].astype(np.int64)

    weight_lookup = ensemble["selected_weights"]
    streams = []
    for stream in ensemble["streams"]:
        enriched = dict(stream)
        enriched["weight"] = weight_lookup[stream["name"]]
        streams.append(load_stream(enriched, ensemble["normalization"], test_split))

    threshold = float(ensemble["validation_selection"]["threshold"])
    steps = min(args.num_steps, labels.shape[0])
    scores: list[float] = []
    total_ms: list[float] = []
    response_ms: list[float] = []

    for stream in streams:
        _ = stream["predict_score"](stream["features"][:1])

    start = time.perf_counter()
    for step in range(steps):
        scheduled = start + (step * args.interval_ms / 1000.0)
        now = time.perf_counter()
        if now < scheduled:
            time.sleep(scheduled - now)

        begin = time.perf_counter()
        ensemble_score = 0.0
        for stream in streams:
            raw_score = stream["predict_score"](stream["features"][step : step + 1])
            if stream["direction"] == "low_is_abnormal":
                raw_score = -raw_score
            ensemble_score += stream["weight"] * normalize(raw_score, stream["norm_stats"])
        end = time.perf_counter()

        scores.append(float(ensemble_score))
        total_ms.append((end - begin) * 1000.0)
        response_ms.append((end - scheduled) * 1000.0)

    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = labels[:steps]
    response_arr = np.asarray(response_ms, dtype=np.float64)
    miss_count = int((response_arr > args.deadline_ms).sum())
    summary = {
        "baseline": "External score ensemble resident replay",
        "ensemble_summary": str(args.ensemble_summary),
        "mode": "paced_replay",
        "num_steps": steps,
        "interval_ms": args.interval_ms,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": miss_count,
        "deadline_miss_rate_pct": float(miss_count * 100.0 / max(steps, 1)),
        "processing_latency": {"total_ms": latency_stats(total_ms)},
        "response_latency": {"response_time_ms": latency_stats(response_ms)},
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_metrics": metrics_from_scores(score_arr, label_arr, threshold),
        "streams": [
            {
                "name": stream["name"],
                "summary_path": stream["summary_path"],
                "weight": stream["weight"],
            }
            for stream in streams
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latency = summary["response_latency"]["response_time_ms"]
    print("External score ensemble replay finished")
    print(
        "Replay F1={:.4f}, miss@{}ms={:.1f}%, response p99={:.2f}ms".format(
            summary["replay_detection_metrics"]["f1"],
            args.deadline_ms,
            summary["deadline_miss_rate_pct"],
            latency["p99_ms"],
        )
    )
    print(f"Summary: {args.output_json}")


if __name__ == "__main__":
    main()
