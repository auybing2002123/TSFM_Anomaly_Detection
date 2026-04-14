from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Dict, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_rcaeval.dataset_loader import load_rcaeval_lazy_stratified  # noqa: E402
from scripts.baselines.run_rcaeval_baselines import (  # noqa: E402
    CaseRecord,
    TRACE_DATA_METHODS,
    ensure_causallearn_compatibility,
    ensure_rcaeval_importable,
    evaluate_records,
    infer_sli,
    load_case_dataframe,
    load_evaluator_types,
    load_method,
    print_case_progress,
    print_method_summary,
    resolve_dataset_root,
    resolve_method_dataset,
    split_service_fault,
    write_summary_files,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "rca_direction1"
SUPPORTED_METHODS = {"baro", "tracerca"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BARO/TraceRCA on the aligned RE2-TT split with a short post-injection horizon."
    )
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["re2-tt"], default="re2-tt")
    parser.add_argument("--methods", nargs="+", default=["baro", "tracerca"])
    parser.add_argument("--normal-context-seconds", type=int, default=600)
    parser.add_argument("--post-injection-seconds", type=int, default=20)
    parser.add_argument("--rcaeval-root", type=Path, default=PROJECT_ROOT.parent / "RCAEval")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--include-case-records", action="store_true")
    return parser.parse_args()


def get_split_case_ids(data_dir: str, split: str, seed: int) -> List[str]:
    splits = load_rcaeval_lazy_stratified(data_dir, seed=seed)
    if split == "all":
        entries = []
        for key in ("train", "val", "test"):
            entries.extend(splits[key].entries)
    else:
        entries = splits[split].entries
    return sorted({entry["case_name"].rsplit("_w", 1)[0] for entry in entries})


def resolve_case_path(dataset_root: Path, case_id: str) -> Path:
    fault_dir, case_num = case_id.split("/", 1)
    candidate = dataset_root / fault_dir / case_num / "simple_metrics.csv"
    if candidate.exists():
        return candidate
    fallback = dataset_root / fault_dir / case_num / "data.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No simple_metrics.csv/data.csv found for case {case_id} under {dataset_root}")


def trim_case_with_horizon(
    data: pd.DataFrame,
    inject_time: int,
    normal_context_seconds: int,
    post_injection_seconds: int,
) -> pd.DataFrame:
    normal_df = data[data["time"] < inject_time].tail(normal_context_seconds)
    anomal_df = data[
        (data["time"] >= inject_time) & (data["time"] < inject_time + post_injection_seconds)
    ].copy()
    return pd.concat([normal_df, anomal_df], ignore_index=True)


def trim_trace_with_horizon(
    trace_df: pd.DataFrame,
    inject_time_us: int,
    normal_context_seconds: int,
    post_injection_seconds: int,
) -> pd.DataFrame:
    normal_context_us = normal_context_seconds * 1_000_000
    post_injection_us = post_injection_seconds * 1_000_000
    lower_bound = inject_time_us - normal_context_us
    upper_bound = inject_time_us + post_injection_us
    trace_end = trace_df["startTime"] + trace_df["duration"]
    window_mask = (trace_end >= lower_bound) & (trace_df["startTime"] < upper_bound)
    return trace_df.loc[window_mask].copy()


def load_method_input_with_horizon(
    method_name: str,
    data_path: Path,
    normal_context_seconds: int,
    post_injection_seconds: int,
):
    case_dir = data_path.parent
    service, fault = split_service_fault(case_dir.parent.name)
    inject_time = int((case_dir / "inject_time.txt").read_text(encoding="utf-8").strip())

    if method_name in TRACE_DATA_METHODS:
        inject_time_us = inject_time * 1_000_000
        trace_df = pd.read_csv(case_dir / "traces.csv")
        trace_df["methodName"] = trace_df["methodName"].fillna(trace_df["operationName"])
        trace_df["operation"] = trace_df["serviceName"] + "_" + trace_df["methodName"]
        normal_mask = trace_df["startTime"] + trace_df["duration"] < inject_time_us
        normal_ops = set(trace_df.loc[normal_mask, "operation"].dropna().tolist())
        if normal_ops:
            trace_df = trace_df[trace_df["operation"].isin(normal_ops)].copy()
        trace_df = trim_trace_with_horizon(
            trace_df,
            inject_time_us=inject_time_us,
            normal_context_seconds=normal_context_seconds,
            post_injection_seconds=post_injection_seconds,
        )
        return trace_df, inject_time_us, service, fault

    data, inject_time, service, fault = load_case_dataframe(data_path)
    data = trim_case_with_horizon(
        data,
        inject_time=inject_time,
        normal_context_seconds=normal_context_seconds,
        post_injection_seconds=post_injection_seconds,
    )
    return data, inject_time, service, fault


def run_single_method_with_horizon(
    method_name: str,
    method_fn,
    case_paths: Sequence[Path],
    dataset: str,
    normal_context_seconds: int,
    post_injection_seconds: int,
    rcaeval_root: Path,
    include_case_records: bool,
    fail_fast: bool,
    show_progress: bool,
) -> Dict[str, object]:
    case_records: List[CaseRecord] = []
    errors: List[Dict[str, str]] = []
    start = time.perf_counter()
    ensure_causallearn_compatibility(method_name)

    total_cases = len(case_paths)
    for index, data_path in enumerate(case_paths, start=1):
        case_start = time.perf_counter()
        try:
            data, inject_time, service, fault = load_method_input_with_horizon(
                method_name,
                data_path,
                normal_context_seconds=normal_context_seconds,
                post_injection_seconds=post_injection_seconds,
            )
            sli_source = data
            sli = infer_sli(data_path, service, sli_source)
            run_args = SimpleNamespace(root_path=str(rcaeval_root), data_path=str(data_path))
            num_node = max(len(sli_source.columns) - 1, 1)

            result = method_fn(
                data,
                inject_time,
                dataset=resolve_method_dataset(method_name, dataset),
                anomalies=None,
                dk_select_useful=False,
                sli=sli,
                verbose=False,
                n_iter=num_node,
                args=run_args,
            )
            elapsed = time.perf_counter() - case_start
            ranks = list(result.get("ranks", []))
            case_records.append(
                CaseRecord(
                    service=service,
                    fault=fault,
                    case_id=data_path.parent.name,
                    data_path=str(data_path),
                    elapsed_sec=elapsed,
                    top5=ranks[:5],
                )
            )
        except Exception as exc:
            errors.append({"data_path": str(data_path), "error": repr(exc)})
            if fail_fast:
                raise
        finally:
            if show_progress:
                print_case_progress(
                    method_name=method_name,
                    index=index,
                    total=total_cases,
                    num_successful=len(case_records),
                    num_failed=len(errors),
                    started_at=start,
                    last_case_sec=time.perf_counter() - case_start,
                    case_path=data_path,
                )

    total_elapsed = time.perf_counter() - start
    Evaluator, Node = load_evaluator_types()
    metrics = evaluate_records(case_records, Evaluator, Node)

    payload = {
        "method": method_name,
        "num_cases": len(case_paths),
        "num_successful": len(case_records),
        "num_failed": len(errors),
        "avg_speed_sec": (total_elapsed / len(case_paths)) if case_paths else None,
        "total_elapsed_sec": total_elapsed,
        "metrics": metrics,
        "errors": errors,
    }
    if include_case_records:
        payload["case_records"] = [record.__dict__ for record in case_records]
    return payload


def main() -> int:
    args = parse_args()
    unsupported = sorted(set(args.methods) - SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported methods for horizon probe: {unsupported}")

    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    case_ids = get_split_case_ids(args.data_dir, args.split, args.seed)
    case_paths = [resolve_case_path(dataset_root, case_id) for case_id in case_ids]

    ensure_rcaeval_importable(args.rcaeval_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        args.output_dir
        / f"rcaeval_{args.dataset}_{args.split}_h{args.post_injection_seconds}s_aligned_{timestamp}"
    )
    payload: Dict[str, object] = {
        "metadata": {
            "dataset": args.dataset,
            "dataset_root": str(dataset_root),
            "rcaeval_root": str(args.rcaeval_root),
            "processed_data_dir": args.data_dir,
            "split": args.split,
            "seed": args.seed,
            "methods": args.methods,
            "num_cases": len(case_paths),
            "normal_context_seconds": args.normal_context_seconds,
            "post_injection_seconds": args.post_injection_seconds,
            "timestamp": timestamp,
            "note": (
                "Short-horizon baseline probe on the same split case set. "
                "This is an approximation of stricter early RCA, not an exact first_3 window metric."
            ),
            "case_ids": case_ids,
        },
        "methods": {},
    }

    print("=" * 72)
    print("RCAEval Baseline Runner (Aligned RE2-TT Split, Short Horizon)")
    print("=" * 72)
    print(f"Dataset               : {args.dataset}")
    print(f"Split                 : {args.split}")
    print(f"Seed                  : {args.seed}")
    print(f"Normal context (sec)  : {args.normal_context_seconds}")
    print(f"Post-injection (sec)  : {args.post_injection_seconds}")
    print(f"Cases                 : {len(case_paths)}")
    print(f"Methods               : {', '.join(args.methods)}")

    for method_name in args.methods:
        try:
            method_fn = load_method(method_name, args.rcaeval_root)
            result = run_single_method_with_horizon(
                method_name=method_name,
                method_fn=method_fn,
                case_paths=case_paths,
                dataset=args.dataset,
                normal_context_seconds=args.normal_context_seconds,
                post_injection_seconds=args.post_injection_seconds,
                rcaeval_root=args.rcaeval_root,
                include_case_records=args.include_case_records,
                fail_fast=args.fail_fast,
                show_progress=not args.no_progress,
            )
            result["status"] = "ok"
        except Exception as exc:
            result = {"status": "error", "error": repr(exc)}
            if args.fail_fast:
                payload["methods"][method_name] = result
                break

        payload["methods"][method_name] = result
        print_method_summary(method_name, result)

    json_path, csv_path = write_summary_files(run_dir, payload)
    print("=" * 72)
    print(f"Summary JSON : {json_path}")
    print(f"Summary CSV  : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
