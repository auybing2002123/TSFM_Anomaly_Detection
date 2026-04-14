from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from functools import reduce
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parents[1]
    default_tranad_root = workspace_root / "external" / "TranAD"
    default_source_metrics = repo_root / "datasets" / "MSDS" / "metrics"

    parser = argparse.ArgumentParser(
        description="Prepare MSDS train/test CSVs for the official TranAD repository."
    )
    parser.add_argument(
        "--tranad-root",
        type=Path,
        default=default_tranad_root,
        help=f"Path to the cloned TranAD repository (default: {default_tranad_root})",
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=default_source_metrics,
        help=f"Path to local MSDS metric CSVs (default: {default_source_metrics})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metric CSVs and generated train/test files.",
    )
    return parser.parse_args()


def copy_metric_files(source_metrics: Path, target_metrics: Path, overwrite: bool) -> list[Path]:
    if not source_metrics.exists():
        raise FileNotFoundError(f"Source metrics directory not found: {source_metrics}")

    target_metrics.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for csv_file in sorted(source_metrics.glob("wally*_metrics_concurrent.csv")):
        target_file = target_metrics / csv_file.name
        if target_file.exists() and not overwrite:
            copied.append(target_file)
            continue
        shutil.copy2(csv_file, target_file)
        copied.append(target_file)
    if not copied:
        raise FileNotFoundError(
            f"No MSTGAD/TranAD-compatible metric CSVs found in {source_metrics}"
        )
    return copied


def load_metric_frames(metric_files: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    drop_columns = ["load.cpucore", "load.min1", "load.min5", "load.min15"]
    for csv_file in metric_files:
        df = pd.read_csv(csv_file)
        host_name = csv_file.name.replace("_metrics_concurrent.csv", "")
        existing_drop = [col for col in drop_columns if col in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop)
        rename_map = {
            col: f"{host_name}.{col}"
            for col in df.columns
            if col != "now"
        }
        df = df.rename(columns=rename_map)
        frames.append(df)
    return frames


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    start = max(df["now"].min() for df in frames)
    end = min(df["now"].max() for df in frames)
    processed: list[pd.DataFrame] = []
    for df in frames:
        clipped = df[(df["now"] >= start) & (df["now"] <= end)]
        melted = clipped.melt(id_vars=["now"]).dropna()
        pivoted = melted.pivot_table(index="now", columns="variable", values="value")
        processed.append(pivoted)

    merged = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        processed,
    )

    merged.index = [
        datetime.strptime(str(timestamp)[:-5], "%Y-%m-%d %H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for timestamp in merged.index
    ]
    return merged


def write_splits(
    merged: pd.DataFrame, output_dir: Path, labels_csv: Path, overwrite: bool
) -> tuple[Path, Path]:
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"

    if (train_csv.exists() or test_csv.exists()) and not overwrite:
        return train_csv, test_csv

    start = round(merged.shape[0] * 0.1)
    trimmed = merged[start:]
    split = round(trimmed.shape[0] / 2)
    train_df = trimmed[:split]
    test_df = trimmed[split:]

    label_rows = len(pd.read_csv(labels_csv))
    desired_test_rows = label_rows * 5
    if desired_test_rows <= 0:
        raise ValueError(f"Invalid label row count in {labels_csv}: {label_rows}")
    if test_df.shape[0] < desired_test_rows:
        raise ValueError(
            f"Generated test.csv has only {test_df.shape[0]} rows, "
            f"but labels.csv implies {desired_test_rows} raw rows are required."
        )
    if test_df.shape[0] != desired_test_rows:
        test_df = test_df.iloc[:desired_test_rows]

    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)
    return train_csv, test_csv


def main() -> int:
    args = parse_args()
    tranad_root = args.tranad_root.resolve()
    source_metrics = args.source_metrics.resolve()
    msds_dir = tranad_root / "data" / "MSDS"
    metrics_dir = msds_dir / "metrics"
    labels_csv = msds_dir / "labels.csv"

    if not tranad_root.exists():
        raise FileNotFoundError(f"TranAD repo not found: {tranad_root}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"TranAD labels.csv not found: {labels_csv}")

    copied_files = copy_metric_files(source_metrics, metrics_dir, args.overwrite)
    frames = load_metric_frames(copied_files)
    merged = merge_frames(frames)
    train_csv, test_csv = write_splits(merged, msds_dir, labels_csv, args.overwrite)

    print("Prepared TranAD MSDS dataset")
    print(f"TranAD root   : {tranad_root}")
    print(f"Source metrics: {source_metrics}")
    print(f"Metric files  : {len(copied_files)}")
    print(f"Rows merged   : {merged.shape[0]}")
    print(f"Features      : {merged.shape[1]}")
    print(f"train.csv     : {train_csv}")
    print(f"test.csv      : {test_csv}")
    print(f"labels.csv    : {labels_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
