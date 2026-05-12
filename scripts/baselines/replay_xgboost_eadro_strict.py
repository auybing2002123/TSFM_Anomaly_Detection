from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a trained XGBoost Eadro-SN strict baseline without retraining."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-events-jsonl",
        type=Path,
        default=None,
        help="Optional per-step replay event log used for paper latency CDF figures.",
    )
    parser.add_argument("--num-steps", type=int, default=568)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument(
        "--predict-mode",
        choices=["sklearn_proba", "booster_inplace"],
        default="sklearn_proba",
        help=(
            "sklearn_proba uses the original XGBClassifier.predict_proba path; "
            "booster_inplace uses the lower-overhead Booster.inplace_predict path."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_features(split: dict[str, np.ndarray], representation: str) -> np.ndarray:
    modalities = [item.strip() for item in representation.split("+") if item.strip()]
    parts = [
        split[modality].reshape(split[modality].shape[0], -1).astype(np.float32)
        for modality in modalities
    ]
    return np.concatenate(parts, axis=1)


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


def predict_ensemble_score(models: list[Any], row: np.ndarray, mode: str) -> float:
    if mode == "booster_inplace":
        scores = []
        for model in models:
            booster = model.get_booster()
            pred = booster.inplace_predict(row, validate_features=False)
            scores.append(float(np.asarray(pred, dtype=np.float64).reshape(-1)[0]))
        return float(np.mean(scores))
    return float(np.mean([model.predict_proba(row)[:, 1] for model in models], axis=0)[0])


def main() -> None:
    args = parse_args()
    summary_path = resolve(args.summary_json)
    summary = load_json(summary_path)
    test = dict(np.load(args.export_dir / "test.npz", allow_pickle=True))
    features = np.ascontiguousarray(build_features(test, summary["representation"]), dtype=np.float32)
    labels = test["window_labels"].astype(np.int64)
    threshold = float(summary["validation_selection"]["threshold"])

    models = []
    for item in summary["artifacts"]["models"]:
        with resolve(Path(item["checkpoint_pickle"])).open("rb") as handle:
            models.append(pickle.load(handle))

    for _ in range(3):
        _ = predict_ensemble_score(models, features[:1], args.predict_mode)

    steps = min(args.num_steps, labels.shape[0])
    scores: list[float] = []
    processing_ms: list[float] = []
    response_ms: list[float] = []
    events: list[dict[str, Any]] = []

    start = time.perf_counter()
    for step in range(steps):
        scheduled = start + step * args.interval_ms / 1000.0
        now = time.perf_counter()
        if now < scheduled:
            time.sleep(scheduled - now)

        begin = time.perf_counter()
        row = features[step : step + 1]
        score = predict_ensemble_score(models, row, args.predict_mode)
        end = time.perf_counter()

        scores.append(score)
        current_processing_ms = (end - begin) * 1000.0
        current_response_ms = (end - scheduled) * 1000.0
        current_prediction = int(score >= threshold)
        current_label = int(labels[step])
        processing_ms.append(current_processing_ms)
        response_ms.append(current_response_ms)
        events.append(
            {
                "step": step,
                "score": score,
                "label": current_label,
                "prediction": current_prediction,
                "processing_ms": current_processing_ms,
                "response_time_ms": current_response_ms,
                "deadline_ms": args.deadline_ms,
                "deadline_met": current_response_ms <= args.deadline_ms,
            }
        )

        if step in {0, steps - 1}:
            print(
                f"[step={step:03d}] score={score:.4f} "
                f"proc={processing_ms[-1]:.2f}ms resp={response_ms[-1]:.2f}ms"
            )

    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = labels[:steps]
    response_arr = np.asarray(response_ms, dtype=np.float64)
    miss_count = int((response_arr > args.deadline_ms).sum())

    replay_summary = {
        "baseline": "XGBoost resident replay",
        "source_summary": str(summary_path),
        "representation": summary["representation"],
        "params": summary.get("params", {}),
        "serving_optimization": {
            "predict_mode": args.predict_mode,
            "resident_models": True,
            "resident_features": True,
            "notes": (
                "booster_inplace avoids per-step sklearn probability wrapper overhead "
                "while preserving the trained XGBoost trees and online batch size one."
            ),
        },
        "mode": "paced_replay",
        "num_steps": steps,
        "interval_ms": args.interval_ms,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_count": miss_count,
        "deadline_miss_rate_pct": float(miss_count * 100.0 / max(steps, 1)),
        "processing_latency": {"total_ms": latency_stats(processing_ms)},
        "response_latency": {"response_time_ms": latency_stats(response_ms)},
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_metrics": metrics_from_scores(score_arr, label_arr, threshold),
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_events_jsonl is not None:
        events_path = resolve(args.output_events_jsonl)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    latency = replay_summary["response_latency"]["response_time_ms"]
    metrics = replay_summary["replay_detection_metrics"]
    print(
        "XGBoost replay finished: "
        f"F1={metrics['f1']:.4f}, miss@{args.deadline_ms:.0f}ms={replay_summary['deadline_miss_rate_pct']:.1f}%, "
        f"p99={latency['p99_ms']:.2f}ms, max={latency['max_ms']:.2f}ms"
    )
    print(f"Summary: {output_path}")
    if args.output_events_jsonl is not None:
        print(f"Events : {resolve(args.output_events_jsonl)}")


if __name__ == "__main__":
    main()
