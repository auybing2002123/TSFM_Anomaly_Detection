from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    external_root = PROJECT_ROOT.parents[1] / "external" / "TraceAnomaly"
    parser = argparse.ArgumentParser(
        description="Run official TraceAnomaly on Eadro-SN strict-protocol exports."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42",
        help="Directory produced by prepare_eadro_sn_baseline_exports.py",
    )
    parser.add_argument(
        "--traceanomaly-root",
        type=Path,
        default=external_root,
        help="Path to the cloned external TraceAnomaly repository.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baselines" / "traceanomaly_eadro_strict_s42",
        help="Working directory for converted inputs and copied outputs.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines" / "traceanomaly_eadro_strict_s42",
        help="Directory for final summary and copied score CSVs.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="silence1990/docker_for_traceanomaly:latest",
        help="Docker image that can run the official TraceAnomaly code.",
    )
    parser.add_argument(
        "--flow-type",
        type=str,
        default="rnvp",
        choices=["rnvp", "planar_nf", "none"],
        help="Official TraceAnomaly flow type. 'none' maps to VAE without flow.",
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument(
        "--representation",
        type=str,
        default="aggregate",
        choices=["aggregate", "flatten_time"],
        help="How to turn a window trace tensor into the single vector expected by TraceAnomaly.",
    )
    parser.add_argument(
        "--seed-tag",
        type=str,
        default="s42",
        help="Tag only used in filenames for this strict-protocol run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class SplitBundle:
    ids: list[str]
    labels: np.ndarray
    vectors: np.ndarray


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {name: payload[name] for name in payload.files}


def build_vector_from_trace(trace_tensor: np.ndarray, representation: str) -> np.ndarray:
    if representation == "flatten_time":
        return trace_tensor.reshape(-1).astype(np.float32)

    counts = trace_tensor[..., 0].astype(np.float32)
    mean_durations = trace_tensor[..., 1].astype(np.float32)
    agg_counts = counts.sum(axis=0)
    active_mask = counts > 0
    active_steps = active_mask.sum(axis=0).astype(np.float32)
    duration_sum = np.where(active_mask, mean_durations, 0.0).sum(axis=0)
    agg_mean_duration = np.divide(
        duration_sum,
        np.maximum(active_steps, 1.0),
        out=np.zeros_like(duration_sum, dtype=np.float32),
        where=active_steps > 0,
    )
    return np.concatenate([agg_counts.reshape(-1), agg_mean_duration.reshape(-1)], axis=0).astype(
        np.float32
    )


def vectorize_split(payload: dict[str, np.ndarray], representation: str, split_name: str) -> SplitBundle:
    trace_tensor = payload["traces"]
    labels = payload["window_labels"].astype(np.int64)
    case_names = payload["case_names"]
    vectors = []
    ids: list[str] = []
    for idx in range(trace_tensor.shape[0]):
        vectors.append(build_vector_from_trace(trace_tensor[idx], representation))
        case_name = str(case_names[idx])
        ids.append(f"{split_name}_{idx:05d}_{case_name.replace('/', '_')}")
    return SplitBundle(ids=ids, labels=labels, vectors=np.stack(vectors, axis=0))


def write_vector_file(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample_id, vector in zip(ids, vectors):
            values = ",".join(f"{float(v):.6f}" for v in vector.tolist())
            f.write(f"{sample_id}:{values}\n")


def build_official_inputs(
    export_dir: Path, work_dir: Path, representation: str, overwrite: bool
) -> dict[str, Path]:
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "input_manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    train_payload = load_npz(export_dir / "train.npz")
    val_payload = load_npz(export_dir / "val.npz")
    test_payload = load_npz(export_dir / "test.npz")

    train_split = vectorize_split(train_payload, representation, "train")
    val_split = vectorize_split(val_payload, representation, "val")
    test_split = vectorize_split(test_payload, representation, "test")

    train_normal_mask = train_split.labels == 0
    val_normal_mask = val_split.labels == 0
    val_abnormal_mask = val_split.labels == 1
    test_normal_mask = test_split.labels == 0
    test_abnormal_mask = test_split.labels == 1

    paths = {
        "train": input_dir / "train.txt",
        "val_normal": input_dir / "val_normal.txt",
        "val_abnormal": input_dir / "val_abnormal.txt",
        "test_normal": input_dir / "test_normal.txt",
        "test_abnormal": input_dir / "test_abnormal.txt",
    }

    write_vector_file(paths["train"], list(np.array(train_split.ids)[train_normal_mask]), train_split.vectors[train_normal_mask])
    write_vector_file(
        paths["val_normal"], list(np.array(val_split.ids)[val_normal_mask]), val_split.vectors[val_normal_mask]
    )
    write_vector_file(
        paths["val_abnormal"],
        list(np.array(val_split.ids)[val_abnormal_mask]),
        val_split.vectors[val_abnormal_mask],
    )
    write_vector_file(
        paths["test_normal"],
        list(np.array(test_split.ids)[test_normal_mask]),
        test_split.vectors[test_normal_mask],
    )
    write_vector_file(
        paths["test_abnormal"],
        list(np.array(test_split.ids)[test_abnormal_mask]),
        test_split.vectors[test_abnormal_mask],
    )

    manifest = {
        "representation": representation,
        "train": str(paths["train"]),
        "val_normal": str(paths["val_normal"]),
        "val_abnormal": str(paths["val_abnormal"]),
        "test_normal": str(paths["test_normal"]),
        "test_abnormal": str(paths["test_abnormal"]),
        "train_normal_count": int(train_normal_mask.sum()),
        "val_normal_count": int(val_normal_mask.sum()),
        "val_abnormal_count": int(val_abnormal_mask.sum()),
        "test_normal_count": int(test_normal_mask.sum()),
        "test_abnormal_count": int(test_abnormal_mask.sum()),
        "vector_dim": int(train_split.vectors.shape[1]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def ensure_path_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def to_container_project_path(project_root: Path, host_path: str | Path) -> str:
    rel = Path(host_path).resolve().relative_to(project_root.resolve())
    return str(PurePosixPath("/workspace/project", *rel.parts))


def run_traceanomaly_docker(
    traceanomaly_root: Path,
    project_root: Path,
    input_paths: dict[str, str],
    docker_image: str,
    flow_type: str,
    max_epoch: int,
    batch_size: int,
    test_batch_size: int,
    output_name: str,
    model_name: str,
    skip_train: bool,
) -> tuple[Path, Path]:
    ensure_path_exists(traceanomaly_root, "TraceAnomaly repo")
    webank_dir = traceanomaly_root / "webankdata"
    webank_dir.mkdir(parents=True, exist_ok=True)

    flow_value = flow_type if flow_type != "none" else None
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "-v",
        f"{traceanomaly_root}:/workspace/TraceAnomaly",
        "-v",
        f"{project_root}:/workspace/project",
        "-w",
        "/workspace/TraceAnomaly",
        docker_image,
        "-m",
        "traceanomaly.main",
        "--trainpath",
        to_container_project_path(project_root, input_paths["train"]),
        "--normalpath",
        to_container_project_path(project_root, input_paths["normal"]),
        "--abnormalpath",
        to_container_project_path(project_root, input_paths["abnormal"]),
        "--outputpath",
        output_name,
        "--modelname",
        model_name,
    ]
    if skip_train:
        cmd.append("--skip-train")
    if flow_value is not None:
        cmd.extend(["-c", f"flow_type={flow_value}"])
    cmd.extend(["-c", f"max_epoch={max_epoch}"])
    cmd.extend(["-c", f"batch_size={batch_size}"])
    cmd.extend(["-c", f"test_batch_size={test_batch_size}"])

    subprocess.run(cmd, check=True)

    flow_prefix = flow_value or "vae"
    score_csv = webank_dir / f"{flow_prefix}_{output_name}.csv"
    valid_csv = webank_dir / f"v{flow_prefix}_{output_name}.csv"
    ensure_path_exists(score_csv, "TraceAnomaly score CSV")
    ensure_path_exists(valid_csv, "TraceAnomaly validation CSV")
    return score_csv, valid_csv


def read_score_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id", "label", "score"}.issubset(df.columns):
        raise ValueError(f"Unexpected score CSV columns in {path}: {list(df.columns)}")
    return df


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


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    thresholds = np.unique(scores)
    best: dict[str, object] | None = None
    for direction in ("low_is_abnormal", "high_is_abnormal"):
        for threshold in thresholds:
            if direction == "low_is_abnormal":
                preds = (scores <= threshold).astype(np.int64)
            else:
                preds = (scores >= threshold).astype(np.int64)
            metrics = compute_metrics(preds, labels)
            candidate = {
                "direction": direction,
                "threshold": float(threshold),
                "metrics": metrics,
            }
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


def apply_threshold(scores: np.ndarray, labels: np.ndarray, direction: str, threshold: float) -> dict[str, object]:
    if direction == "low_is_abnormal":
        preds = (scores <= threshold).astype(np.int64)
    else:
        preds = (scores >= threshold).astype(np.int64)
    return {
        "direction": direction,
        "threshold": float(threshold),
        "metrics": compute_metrics(preds, labels),
    }


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    traceanomaly_root = args.traceanomaly_root.resolve()
    work_dir = args.work_dir.resolve()
    result_dir = args.result_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    ensure_path_exists(export_dir, "Eadro strict export dir")
    ensure_path_exists(traceanomaly_root / "traceanomaly" / "main.py", "Patched TraceAnomaly main.py")

    manifest = build_official_inputs(export_dir, work_dir, args.representation, args.overwrite)
    model_name = f"eadro_traceanomaly_{args.seed_tag}_{args.representation}"
    output_prefix = f"eadro_traceanomaly_{args.seed_tag}_{args.representation}"

    val_score_csv, val_internal_csv = run_traceanomaly_docker(
        traceanomaly_root=traceanomaly_root,
        project_root=PROJECT_ROOT,
        input_paths={
            "train": manifest["train"],
            "normal": manifest["val_normal"],
            "abnormal": manifest["val_abnormal"],
        },
        docker_image=args.docker_image,
        flow_type=args.flow_type,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        output_name=f"{output_prefix}_val",
        model_name=model_name,
        skip_train=False,
    )

    test_score_csv, test_internal_csv = run_traceanomaly_docker(
        traceanomaly_root=traceanomaly_root,
        project_root=PROJECT_ROOT,
        input_paths={
            "train": manifest["train"],
            "normal": manifest["test_normal"],
            "abnormal": manifest["test_abnormal"],
        },
        docker_image=args.docker_image,
        flow_type=args.flow_type,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        output_name=f"{output_prefix}_test",
        model_name=model_name,
        skip_train=True,
    )

    copied_val_csv = result_dir / val_score_csv.name
    copied_test_csv = result_dir / test_score_csv.name
    copied_val_internal = result_dir / val_internal_csv.name
    copied_test_internal = result_dir / test_internal_csv.name
    copy_if_needed(val_score_csv, copied_val_csv)
    copy_if_needed(test_score_csv, copied_test_csv)
    copy_if_needed(val_internal_csv, copied_val_internal)
    copy_if_needed(test_internal_csv, copied_test_internal)

    val_df = read_score_csv(copied_val_csv)
    test_df = read_score_csv(copied_test_csv)
    best_val = find_best_threshold(val_df["score"].to_numpy(), val_df["label"].to_numpy())
    test_eval = apply_threshold(
        test_df["score"].to_numpy(),
        test_df["label"].to_numpy(),
        direction=str(best_val["direction"]),
        threshold=float(best_val["threshold"]),
    )

    summary = {
        "baseline": "TraceAnomaly",
        "dataset": "Eadro-SN",
        "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
        "representation": args.representation,
        "flow_type": args.flow_type,
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "traceanomaly_root": str(traceanomaly_root),
        "export_dir": str(export_dir),
        "work_dir": str(work_dir),
        "result_dir": str(result_dir),
        "input_manifest": manifest,
        "validation_selection": best_val,
        "test_result": test_eval,
        "artifacts": {
            "val_score_csv": str(copied_val_csv),
            "test_score_csv": str(copied_test_csv),
            "val_internal_csv": str(copied_val_internal),
            "test_internal_csv": str(copied_test_internal),
        },
    }
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("TraceAnomaly Eadro strict run finished")
    print(f"Validation best: dir={best_val['direction']} thr={best_val['threshold']:.6f} "
          f"F1={best_val['metrics']['f1']:.4f}")
    print(f"Test result: F1={test_eval['metrics']['f1']:.4f}, "
          f"P={test_eval['metrics']['precision']:.4f}, "
          f"R={test_eval['metrics']['recall']:.4f}, "
          f"Acc={test_eval['metrics']['accuracy']:.4f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
