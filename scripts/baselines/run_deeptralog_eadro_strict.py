from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEEPTRALOG_CODE_ROOT = WORKSPACE_ROOT / "external" / "DeepTraLog" / "HetGNN" / "code"
DEFAULT_DATA_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "deeptralog_eadro_strict_s42" / "ProcessedData"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "baselines" / "deeptralog_eadro_strict_s42_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an isolated DeepTraLog HetGNN baseline on Eadro-SN strict export. "
            "This wrapper forces the current strict train/val/test split instead of the official random split."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--embed-d", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-s", type=int, default=256)
    parser.add_argument("--mini-batch-s", type=int, default=128)
    parser.add_argument("--train-iter-n", type=int, default=1)
    parser.add_argument("--save-model-freq", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
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


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(scores)
    best: dict[str, Any] | None = None
    for direction in ("high_is_abnormal", "low_is_abnormal"):
        for threshold in thresholds:
            preds = (scores >= threshold).astype(np.int64) if direction == "high_is_abnormal" else (scores <= threshold).astype(np.int64)
            metrics = compute_metrics(preds, labels)
            candidate = {"direction": direction, "threshold": float(threshold), "metrics": metrics}
            if best is None:
                best = candidate
                continue
            best_metrics = best["metrics"]
            if metrics["f1"] > best_metrics["f1"] + 1e-12:
                best = candidate
            elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and metrics["recall"] > best_metrics["recall"]:
                best = candidate
    assert best is not None
    return best


def apply_threshold(scores: np.ndarray, labels: np.ndarray, direction: str, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64) if direction == "high_is_abnormal" else (scores <= threshold).astype(np.int64)
    return {
        "direction": direction,
        "threshold": float(threshold),
        "metrics": compute_metrics(preds, labels),
        "predictions": preds,
    }


def load_preset_split_lists(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    trace_info = pd.read_csv(data_path / "trace_info.csv")
    train_ids = trace_info[(trace_info["preset_split"] == "train") & (trace_info["trace_bool"] == True)]["trace_id"].to_numpy(dtype=np.int64)  # noqa: E712
    val_ids = trace_info[trace_info["preset_split"] == "val"]["trace_id"].to_numpy(dtype=np.int64)
    test_ids = trace_info[trace_info["preset_split"] == "test"]["trace_id"].to_numpy(dtype=np.int64)
    if train_ids.size == 0 or val_ids.size == 0 or test_ids.size == 0:
        raise ValueError("Preset strict split is incomplete in trace_info.csv")
    return train_ids, val_ids, test_ids, trace_info


def import_deeptralog_modules(cli: argparse.Namespace):
    if str(DEEPTRALOG_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(DEEPTRALOG_CODE_ROOT))

    sys.argv = [
        "run_deeptralog_eadro_strict.py",
        "--data_path",
        str(cli.data_path),
        "--model_path",
        str(cli.result_dir) + "\\",
        "--preprocess",
        "1",
        "--train_iter_n",
        str(cli.train_iter_n),
        "--save_model_freq",
        str(cli.save_model_freq),
        "--batch_s",
        str(cli.batch_s),
        "--mini_batch_s",
        str(cli.mini_batch_s),
        "--embed_d",
        str(cli.embed_d),
        "--random_seed",
        str(cli.random_seed),
        "--cuda",
        str(cli.cuda),
    ]

    args_mod = importlib.import_module("args")
    hetgnn_mod = importlib.import_module("HetGNN")
    return args_mod, hetgnn_mod


def score_graph_ids(model_object: Any, graph_ids: np.ndarray, trace_info: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with __import__("torch").no_grad():
        scores = model_object.model.predict_score(graph_ids.tolist()).detach().cpu().numpy().astype(np.float64)
    label_map = trace_info.set_index("trace_id")["trace_bool"].to_dict()
    labels = np.asarray([0 if bool(label_map[int(gid)]) else 1 for gid in graph_ids], dtype=np.int64)
    return scores, labels


def main() -> int:
    cli = parse_args()
    data_path = cli.data_path.resolve()
    result_dir = cli.result_dir.resolve()
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not cli.overwrite:
        print(f"Summary already exists: {summary_path}")
        return 0
    if result_dir.exists() and cli.overwrite:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids, trace_info = load_preset_split_lists(data_path)
    args_mod, hetgnn_mod = import_deeptralog_modules(cli)
    args = args_mod.read_args()

    t0 = time.perf_counter()
    model_object = hetgnn_mod.model_class(args)
    setup_seconds = time.perf_counter() - t0

    def fixed_split(self):
        return train_ids, val_ids, test_ids

    model_object.train_eval_test_split = types.MethodType(fixed_split, model_object)

    print(
        "DeepTraLog Eadro strict baseline\n"
        f"  train_benign={train_ids.size}, val={val_ids.size}, test={test_ids.size}\n"
        f"  data_path={data_path}\n"
        f"  result_dir={result_dir}",
        flush=True,
    )

    train_start = time.perf_counter()
    model_object.model_train()
    train_seconds = time.perf_counter() - train_start

    eval_auc, eval_ap = model_object.eval_model(val_ids)
    test_auc, test_ap = model_object.eval_model(test_ids)

    val_scores, val_labels = score_graph_ids(model_object, val_ids, trace_info)
    test_scores, test_labels = score_graph_ids(model_object, test_ids, trace_info)

    best_val = find_best_threshold(val_scores, val_labels)
    val_eval = apply_threshold(val_scores, val_labels, str(best_val["direction"]), float(best_val["threshold"]))
    test_eval = apply_threshold(test_scores, test_labels, str(best_val["direction"]), float(best_val["threshold"]))

    summary = {
        "baseline": "DeepTraLog-HetGNN",
        "dataset": "Eadro-SN",
        "protocol": "strict preset split + train-normal-only + val-threshold + test-final",
        "implementation_note": (
            "Official DeepTraLog HetGNN code wrapped by an isolated runner. "
            "The wrapper reuses the current Eadro strict split instead of DeepTraLog's internal random split."
        ),
        "data_path": str(data_path),
        "result_dir": str(result_dir),
        "embed_d": cli.embed_d,
        "lr_note": "DeepTraLog external args.py declares --lr as int, so this wrapper uses the repo default parser value.",
        "batch_s": cli.batch_s,
        "mini_batch_s": cli.mini_batch_s,
        "train_iter_n": cli.train_iter_n,
        "save_model_freq": cli.save_model_freq,
        "cuda": cli.cuda,
        "random_seed": cli.random_seed,
        "input_manifest": {
            "train_benign_graphs": int(train_ids.size),
            "val_graphs": int(val_ids.size),
            "test_graphs": int(test_ids.size),
            "val_anomaly_graphs": int(val_labels.sum()),
            "test_anomaly_graphs": int(test_labels.sum()),
        },
        "timing": {
            "setup_seconds": float(setup_seconds),
            "train_seconds": float(train_seconds),
        },
        "validation_selection": {
            "direction": val_eval["direction"],
            "threshold": val_eval["threshold"],
            "metrics": val_eval["metrics"],
            "roc_auc": float(eval_auc),
            "avg_precision": float(eval_ap),
        },
        "test_result": {
            "direction": test_eval["direction"],
            "threshold": test_eval["threshold"],
            "metrics": test_eval["metrics"],
            "roc_auc": float(test_auc),
            "avg_precision": float(test_ap),
        },
        "artifacts": {
            "checkpoint": str(result_dir / "HetGNN_0.pt"),
            "center": str(result_dir / "HetGNN_SVDD_Center.pt"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DeepTraLog Eadro strict run finished")
    print(
        f"Validation best: dir={val_eval['direction']} thr={val_eval['threshold']:.6f} "
        f"F1={val_eval['metrics']['f1']:.4f}"
    )
    print(
        f"Test result: F1={test_eval['metrics']['f1']:.4f}, "
        f"P={test_eval['metrics']['precision']:.4f}, "
        f"R={test_eval['metrics']['recall']:.4f}, "
        f"Acc={test_eval['metrics']['accuracy']:.4f}, "
        f"AUC={test_auc:.4f}, AP={test_ap:.4f}"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
