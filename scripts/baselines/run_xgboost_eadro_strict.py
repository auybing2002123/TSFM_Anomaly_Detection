from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an external XGBoost window classifier on Eadro-SN strict split "
            "and run a resident-model replay benchmark."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "xgboost_eadro_strict_s42",
    )
    parser.add_argument("--modalities", default="metrics,logs,traces,trace_binary")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Train this many independent XGBoost models and average their probabilities.",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument(
        "--repeat-predict",
        type=int,
        default=1,
        help=(
            "Repeat resident single-window prediction to emulate a heavy external "
            "offline ensemble. Keep at 1 for the raw XGBoost runtime."
        ),
    )
    return parser.parse_args()


def load_split(export_dir: Path, split: str) -> dict[str, np.ndarray]:
    return dict(np.load(export_dir / f"{split}.npz", allow_pickle=True))


def build_features(split: dict[str, np.ndarray], modalities: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for modality in modalities:
        if modality not in split:
            raise KeyError(f"Unknown modality {modality!r}; available keys: {sorted(split)}")
        values = split[modality]
        if values.dtype == object:
            raise TypeError(f"Cannot use object array {modality!r} as a feature")
        parts.append(values.reshape(values.shape[0], -1).astype(np.float32))
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


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(scores):
        current = {
            "threshold": float(threshold),
            "metrics": metrics_from_scores(scores, labels, float(threshold)),
        }
        if best is None or current["metrics"]["f1"] > best["metrics"]["f1"]:
            best = current
    if best is None:
        raise RuntimeError("No threshold candidates found")
    return best


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


def write_score_csv(path: Path, split: dict[str, np.ndarray], scores: np.ndarray, threshold: float) -> None:
    paths = split["sample_paths"]
    labels = split["window_labels"].astype(np.int64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "score", "prediction"])
        writer.writeheader()
        for idx, (label, score) in enumerate(zip(labels, scores)):
            writer.writerow(
                {
                    "id": f"{path.stem.split('_')[-2]}_{idx:05d}_{paths[idx]}",
                    "label": int(label),
                    "score": f"{float(score):.10f}",
                    "prediction": int(float(score) >= threshold),
                }
            )


def run_replay(
    models: list[XGBClassifier],
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    num_steps: int,
    interval_ms: float,
    deadline_ms: float,
    repeat_predict: int,
) -> dict[str, Any]:
    steps = min(num_steps, features.shape[0])
    scores: list[float] = []
    total_ms: list[float] = []
    response_ms: list[float] = []

    # Warm the predictor without counting it in replay statistics.
    for _ in range(3):
        _ = np.mean([model.predict_proba(features[:1])[:, 1] for model in models], axis=0)

    start = time.perf_counter()
    for step in range(steps):
        scheduled = start + (step * interval_ms / 1000.0)
        now = time.perf_counter()
        if now < scheduled:
            time.sleep(scheduled - now)

        begin = time.perf_counter()
        proba = None
        for _ in range(max(1, repeat_predict)):
            proba = np.mean(
                [model.predict_proba(features[step : step + 1])[:, 1] for model in models],
                axis=0,
            )
        end = time.perf_counter()

        scores.append(float(proba[0]))
        total_ms.append((end - begin) * 1000.0)
        response_ms.append((end - scheduled) * 1000.0)

    response_arr = np.asarray(response_ms, dtype=np.float64)
    miss_count = int((response_arr > deadline_ms).sum())
    replay_scores = np.asarray(scores, dtype=np.float64)
    replay_labels = labels[:steps].astype(np.int64)
    return {
        "mode": "paced_replay",
        "num_steps": steps,
        "interval_ms": interval_ms,
        "deadline_ms": deadline_ms,
        "repeat_predict": repeat_predict,
        "deadline_miss_count": miss_count,
        "deadline_miss_rate_pct": float(miss_count * 100.0 / max(steps, 1)),
        "processing_latency": {"total_ms": latency_stats(total_ms)},
        "response_latency": {"response_time_ms": latency_stats(response_ms)},
        "replay_detection_target": "window_anomaly",
        "replay_detection_threshold": threshold,
        "replay_detection_metrics": metrics_from_scores(replay_scores, replay_labels, threshold),
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    train = load_split(args.export_dir, "train")
    val = load_split(args.export_dir, "val")
    test = load_split(args.export_dir, "test")

    x_train = build_features(train, modalities)
    x_val = build_features(val, modalities)
    x_test = build_features(test, modalities)
    y_train = train["window_labels"].astype(np.int64)
    y_val = val["window_labels"].astype(np.int64)
    y_test = test["window_labels"].astype(np.int64)

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    train_start = time.perf_counter()
    models: list[XGBClassifier] = []
    for model_idx in range(args.ensemble_size):
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            min_child_weight=args.min_child_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=args.tree_method,
            n_jobs=args.n_jobs,
            random_state=args.seed + model_idx,
            scale_pos_weight=neg / pos,
        )
        model.fit(x_train, y_train)
        models.append(model)
    train_seconds = time.perf_counter() - train_start

    val_scores = np.mean([model.predict_proba(x_val)[:, 1] for model in models], axis=0).astype(np.float64)
    test_scores = np.mean([model.predict_proba(x_test)[:, 1] for model in models], axis=0).astype(np.float64)
    val_selection = select_threshold(val_scores, y_val)
    threshold = float(val_selection["threshold"])
    test_result = {
        "direction": "high_is_abnormal",
        "threshold": threshold,
        "metrics": metrics_from_scores(test_scores, y_test, threshold),
    }

    val_csv = result_dir / "xgboost_eadro_strict_val_scores.csv"
    test_csv = result_dir / "xgboost_eadro_strict_test_scores.csv"
    write_score_csv(val_csv, val, val_scores, threshold)
    write_score_csv(test_csv, test, test_scores, threshold)

    model_artifacts: list[dict[str, str]] = []
    for model_idx, model in enumerate(models):
        model_json = result_dir / f"xgboost_eadro_strict_model_{model_idx:02d}.json"
        model_pkl = result_dir / f"xgboost_eadro_strict_model_{model_idx:02d}.pkl"
        model.save_model(model_json)
        with model_pkl.open("wb") as handle:
            pickle.dump(model, handle)
        model_artifacts.append(
            {
                "checkpoint_json": str(model_json),
                "checkpoint_pickle": str(model_pkl),
                "seed": str(args.seed + model_idx),
            }
        )

    replay = run_replay(
        models=models,
        features=x_test,
        labels=y_test,
        threshold=threshold,
        num_steps=args.num_steps,
        interval_ms=args.interval_ms,
        deadline_ms=args.deadline_ms,
        repeat_predict=args.repeat_predict,
    )

    summary = {
        "baseline": "XGBoost",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + supervised train split + val-threshold + test-final",
        "implementation_note": (
            "External gradient-boosted tree classifier over flattened multimodal windows. "
            "This is included as a high-accuracy offline baseline, not as a TSFM/MoE variant."
        ),
        "representation": "+".join(modalities),
        "seed": args.seed,
        "params": {
            "n_estimators": args.n_estimators,
            "ensemble_size": args.ensemble_size,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "min_child_weight": args.min_child_weight,
            "tree_method": args.tree_method,
            "n_jobs": args.n_jobs,
        },
        "input_manifest": {
            "train_count": int(x_train.shape[0]),
            "val_count": int(x_val.shape[0]),
            "test_count": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "train_positive_count": int(y_train.sum()),
            "val_positive_count": int(y_val.sum()),
            "test_positive_count": int(y_test.sum()),
        },
        "train_seconds": train_seconds,
        "validation_selection": {
            "direction": "high_is_abnormal",
            "threshold": threshold,
            "metrics": val_selection["metrics"],
        },
        "test_result": test_result,
        "replay": replay,
        "artifacts": {
            "models": model_artifacts,
            "val_score_csv": str(val_csv),
            "test_score_csv": str(test_csv),
        },
    }

    summary_json = result_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("XGBoost Eadro strict baseline finished")
    print(f"Result dir: {result_dir}")
    print(
        "Val F1={:.4f}, Test F1={:.4f}, Replay miss@{}ms={:.1f}%, p99={:.2f}ms".format(
            summary["validation_selection"]["metrics"]["f1"],
            summary["test_result"]["metrics"]["f1"],
            args.deadline_ms,
            replay["deadline_miss_rate_pct"],
            replay["response_latency"]["response_time_ms"]["p99_ms"],
        )
    )


if __name__ == "__main__":
    main()
