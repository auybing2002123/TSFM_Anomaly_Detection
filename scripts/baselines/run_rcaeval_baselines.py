"""
Run RCAEval baselines against the local RCAEval dataset copy used by this project.

Design goals:
- Reuse the local `code/RCAEval` implementation without editing it.
- Keep all outputs inside `results/baselines/`.
- Provide a single entrypoint to run a baseline suite and save structured summaries.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RCAEVAL_ROOT = PROJECT_ROOT.parent / "RCAEval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"
DATASET_ROOTS = {
    "re2-ob": PROJECT_ROOT / "datasets" / "RCAEval" / "RE2-OB",
    "re2-ss": PROJECT_ROOT / "datasets" / "RCAEval" / "RE2-SS",
    "re2-tt": PROJECT_ROOT / "datasets" / "RCAEval" / "RE2-TT",
}

# Methods that are meaningful and realistic to try first on RE2 datasets.
CORE_METHODS = ["baro", "circa", "microcause", "microrank", "tracerca"]
EXTENDED_METHODS = [
    "baro",
    "mmbaro",
    "circa",
    "microcause",
    "microrank",
    "tracerca",
    "run",
    "causalrca",
    "rcd",
    "mmrcd",
]

METHOD_SPECS = {
    "baro": ("RCAEval.e2e.baro", "baro"),
    "mmbaro": ("RCAEval.e2e.baro", "mmbaro"),
    "circa": ("RCAEval.e2e.circa", "circa"),
    "microcause": ("RCAEval.e2e.microcause", "microcause"),
    "microrank": ("RCAEval.e2e.microrank", "microrank"),
    "tracerca": ("RCAEval.e2e.tracerca", "tracerca"),
    "run": ("RCAEval.e2e.run", "run"),
    "causalrca": ("RCAEval.e2e.causalrca", "causalrca"),
    "rcd": ("RCAEval.e2e.rcd", "rcd"),
    "mmrcd": ("RCAEval.e2e.mmrcd", "mmrcd"),
    "dummy": ("RCAEval.e2e", "dummy"),
}

TRACE_DATA_METHODS = {"tracerca", "microrank"}
MULTISOURCE_DATA_METHODS = {"mmbaro", "mmrcd"}

FAULT_ORDER = ["cpu", "mem", "disk", "socket", "delay", "loss"]
FAULT_METRIC_MAP = {
    "cpu": "cpu",
    "mem": "mem",
    "disk": "diskio",
    "socket": "socket",
    "delay": "latency",
    "loss": "latency",
}


@dataclass
class CaseRecord:
    service: str
    fault: str
    case_id: str
    data_path: str
    elapsed_sec: float
    top5: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RCAEval baselines from the TSFM project.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS.keys()), default="re2-tt")
    parser.add_argument(
        "--profile",
        choices=["core", "extended", "custom"],
        default="core",
        help="Baseline suite to run. Use custom with --methods.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Explicit methods to run. Overrides the chosen profile when provided.",
    )
    parser.add_argument("--length", type=int, default=20, help="Window length in minutes, matching RCAEval main.py.")
    parser.add_argument("--limit-cases", type=int, default=None, help="Only run the first N cases for smoke tests.")
    parser.add_argument("--rcaeval-root", type=Path, default=DEFAULT_RCAEVAL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override the local dataset path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when a method/case fails.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-case progress display for long-running baselines.",
    )
    parser.add_argument(
        "--include-case-records",
        action="store_true",
        help="Persist per-case top-5 predictions for each method.",
    )
    return parser.parse_args()


def ensure_rcaeval_importable(rcaeval_root: Path) -> None:
    if not rcaeval_root.exists():
        raise FileNotFoundError(f"RCAEval root not found: {rcaeval_root}")
    root_str = str(rcaeval_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_method(method_name: str, rcaeval_root: Path) -> Callable:
    if method_name not in METHOD_SPECS:
        raise KeyError(f"Unsupported method: {method_name}")
    module_name, func_name = METHOD_SPECS[method_name]

    if module_name == "RCAEval.e2e":
        module = importlib.import_module(module_name)
    else:
        module_path = rcaeval_root.joinpath(*module_name.split(".")).with_suffix(".py")
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")
        spec = importlib.util.spec_from_file_location(f"_tsfm_{method_name}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    if not hasattr(module, func_name):
        raise AttributeError(f"{func_name} not found in {module_name}")
    return getattr(module, func_name)


def load_evaluator_types():
    benchmark = importlib.import_module("RCAEval.benchmark.evaluation")
    graph = importlib.import_module("RCAEval.classes.graph")
    return benchmark.Evaluator, graph.Node


def resolve_dataset_root(dataset: str, dataset_root: Optional[Path]) -> Path:
    root = dataset_root if dataset_root is not None else DATASET_ROOTS[dataset]
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def resolve_methods(profile: str, requested_methods: Optional[Sequence[str]]) -> List[str]:
    if requested_methods:
        return list(dict.fromkeys(requested_methods))
    if profile == "core":
        return CORE_METHODS.copy()
    if profile == "extended":
        return EXTENDED_METHODS.copy()
    raise ValueError("Profile 'custom' requires --methods.")


def discover_case_paths(dataset_root: Path, limit_cases: Optional[int]) -> List[Path]:
    paths = sorted(dataset_root.rglob("simple_metrics.csv"))
    if not paths:
        paths = sorted(dataset_root.rglob("data.csv"))
    if not paths:
        raise FileNotFoundError(f"No simple_metrics.csv or data.csv found under {dataset_root}")
    if limit_cases is not None:
        paths = paths[:limit_cases]
    return paths


def split_service_fault(group_name: str) -> Tuple[str, str]:
    if "_" not in group_name:
        raise ValueError(f"Cannot parse service/fault from '{group_name}'")
    service, fault = group_name.rsplit("_", 1)
    return service, fault


def infer_sli(data_path: Path, service: str, data: pd.DataFrame) -> str:
    path_str = str(data_path).replace("\\", "/")
    if "train-ticket" in path_str or "RE2-TT" in path_str:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        return sli
    if "online-boutique" in path_str or "RE2-OB" in path_str or "RE2-SS" in path_str:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        return sli
    raise ValueError(f"SLI mapping not implemented for {data_path}")


def load_case_dataframe(data_path: Path) -> Tuple[pd.DataFrame, int, str, str]:
    group_name = data_path.parent.parent.name
    case_id = data_path.parent.name
    service, fault = split_service_fault(group_name)

    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    inject_time_path = data_path.parent / "inject_time.txt"
    inject_time = int(inject_time_path.read_text(encoding="utf-8").strip())

    data = data.rename(
        columns={
            column: column.replace("_latency-90", "_latency")
            for column in data.columns
            if column.endswith("_latency-90")
        }
    )
    return data, inject_time, service, fault


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)


def trim_case_window(data: pd.DataFrame, inject_time: int, length_minutes: int) -> pd.DataFrame:
    half_window = length_minutes * 60 // 2
    normal_df = data[data["time"] < inject_time].tail(half_window)
    anomal_df = data[data["time"] >= inject_time].head(half_window)
    return pd.concat([normal_df, anomal_df], ignore_index=True)


def load_multisource_case_payload(case_dir: Path) -> Dict[str, object]:
    with (case_dir / "cluster_info.json").open("r", encoding="utf-8") as handle:
        cluster_info = json.load(handle)

    return {
        "metric": sanitize_dataframe(pd.read_csv(case_dir / "metrics.csv")),
        "logs": sanitize_dataframe(pd.read_csv(case_dir / "logs.csv")),
        "logts": sanitize_dataframe(pd.read_csv(case_dir / "logts.csv")),
        "traces": sanitize_dataframe(pd.read_csv(case_dir / "traces.csv")),
        "tracets_err": sanitize_dataframe(pd.read_csv(case_dir / "tracets_err.csv")),
        "tracets_lat": sanitize_dataframe(pd.read_csv(case_dir / "tracets_lat.csv")),
        "cluster_info": cluster_info,
    }


def ensure_causallearn_compatibility(method_name: str) -> None:
    if method_name != "mmrcd":
        return
    try:
        from causallearn.utils.PCUtils import SkeletonDiscovery
    except Exception:
        return

    if hasattr(SkeletonDiscovery, "local_skeleton_discovery"):
        return

    def _local_skeleton_discovery(np_data, f_node, alpha, indep_test, mi=None, labels=None, verbose=False):
        supported = inspect.signature(SkeletonDiscovery.skeleton_discovery).parameters
        kwargs = {
            "indep_test": indep_test,
            "background_knowledge": None,
            "stable": False,
            "verbose": verbose,
            "labels": labels,
            "show_progress": False,
        }
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        cg = SkeletonDiscovery.skeleton_discovery(np_data, alpha, **filtered_kwargs)
        if not hasattr(cg, "mi"):
            cg.mi = mi or []
        return cg

    SkeletonDiscovery.local_skeleton_discovery = _local_skeleton_discovery


def resolve_method_dataset(method_name: str, dataset: str) -> str:
    if method_name in MULTISOURCE_DATA_METHODS:
        if dataset == "re2-tt":
            return "mm-tt"
        if dataset == "re2-ob":
            return "mm-ob"
    return dataset


def load_method_input(method_name: str, data_path: Path, length_minutes: int) -> Tuple[object, int, str, str]:
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
        return trace_df, inject_time_us, service, fault

    if method_name in MULTISOURCE_DATA_METHODS:
        return load_multisource_case_payload(case_dir), inject_time, service, fault

    data, inject_time, service, fault = load_case_dataframe(data_path)
    data = trim_case_window(data, inject_time, length_minutes)
    return data, inject_time, service, fault


def to_metric_node(raw_rank: str, Node) -> "Node":
    parts = raw_rank.split("_", 1)
    if len(parts) == 1:
        return Node(parts[0], "unknown")
    return Node(parts[0], parts[1])


def to_service_nodes(raw_ranks: Iterable[str], Node) -> List["Node"]:
    unique: List["Node"] = []
    seen = set()
    for raw_rank in raw_ranks:
        metric_node = to_metric_node(raw_rank, Node)
        if metric_node.entity in seen:
            continue
        unique.append(Node(metric_node.entity, "unknown"))
        seen.add(metric_node.entity)
    return unique


def evaluate_records(case_records: Sequence[CaseRecord], Evaluator, Node) -> Dict[str, object]:
    service_eval_all = Evaluator()
    metric_eval_all = Evaluator()
    per_fault = {}

    for fault in FAULT_ORDER:
        service_eval = Evaluator()
        metric_eval = Evaluator()
        for record in case_records:
            if record.fault != fault:
                continue
            metric_ranks = [to_metric_node(raw_rank, Node) for raw_rank in record.top5]
            service_ranks = to_service_nodes(record.top5, Node)

            service_eval.add_case(service_ranks, Node(record.service, "unknown"))
            metric_eval.add_case(metric_ranks, Node(record.service, FAULT_METRIC_MAP[fault]))
            service_eval_all.add_case(service_ranks, Node(record.service, "unknown"))
            metric_eval_all.add_case(metric_ranks, Node(record.service, FAULT_METRIC_MAP[fault]))

        per_fault[fault] = {
            "num_cases": service_eval.num,
            "service": {
                "ac@1": service_eval.accuracy(1),
                "ac@3": service_eval.accuracy(3),
                "ac@5": service_eval.accuracy(5),
                "avg@5": service_eval.average(5),
            },
            "metric": {
                "ac@1": metric_eval.accuracy(1),
                "ac@3": metric_eval.accuracy(3),
                "ac@5": metric_eval.accuracy(5),
                "avg@5": metric_eval.average(5),
            },
        }

    return {
        "overall_service": {
            "ac@1": service_eval_all.accuracy(1),
            "ac@3": service_eval_all.accuracy(3),
            "ac@5": service_eval_all.accuracy(5),
            "avg@5": service_eval_all.average(5),
        },
        "overall_metric": {
            "ac@1": metric_eval_all.accuracy(1),
            "ac@3": metric_eval_all.accuracy(3),
            "ac@5": metric_eval_all.accuracy(5),
            "avg@5": metric_eval_all.average(5),
        },
        "per_fault": per_fault,
    }


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{rem:02d}s"


def print_case_progress(
    method_name: str,
    index: int,
    total: int,
    num_successful: int,
    num_failed: int,
    started_at: float,
    last_case_sec: float,
    case_path: Path,
) -> None:
    elapsed = time.perf_counter() - started_at
    avg = elapsed / max(index, 1)
    eta = avg * max(total - index, 0)
    width = 24
    filled = int(width * index / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"[{method_name}] [{bar}] {index}/{total}"
        f" ok={num_successful} fail={num_failed}"
        f" last={last_case_sec:.2f}s"
        f" elapsed={format_duration(elapsed)}"
        f" eta={format_duration(eta)}"
        f" case={case_path.parent.parent.name}/{case_path.parent.name}",
        flush=True,
    )


def run_single_method(
    method_name: str,
    method_fn: Callable,
    case_paths: Sequence[Path],
    dataset: str,
    length_minutes: int,
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
            data, inject_time, service, fault = load_method_input(method_name, data_path, length_minutes)
            sli_source = data["metric"] if isinstance(data, dict) else data
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
        except Exception as exc:  # pragma: no cover - error path depends on external methods
            error = {"data_path": str(data_path), "error": repr(exc)}
            errors.append(error)
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
        payload["case_records"] = [asdict(record) for record in case_records]
    return payload


def write_summary_files(output_dir: Path, payload: Dict[str, object]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "status",
                "num_cases",
                "num_successful",
                "num_failed",
                "avg_speed_sec",
                "service_ac@1",
                "service_ac@3",
                "service_ac@5",
                "service_avg@5",
                "metric_ac@1",
                "metric_ac@3",
                "metric_ac@5",
                "metric_avg@5",
            ]
        )
        for method_name, result in payload["methods"].items():
            overall_service = result.get("metrics", {}).get("overall_service", {})
            overall_metric = result.get("metrics", {}).get("overall_metric", {})
            writer.writerow(
                [
                    method_name,
                    result.get("status"),
                    result.get("num_cases"),
                    result.get("num_successful"),
                    result.get("num_failed"),
                    result.get("avg_speed_sec"),
                    overall_service.get("ac@1"),
                    overall_service.get("ac@3"),
                    overall_service.get("ac@5"),
                    overall_service.get("avg@5"),
                    overall_metric.get("ac@1"),
                    overall_metric.get("ac@3"),
                    overall_metric.get("ac@5"),
                    overall_metric.get("avg@5"),
                ]
            )
    return json_path, csv_path


def print_method_summary(method_name: str, result: Dict[str, object]) -> None:
    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    status = result["status"]
    print("-" * 72)
    print(f"{method_name}: {status}")
    if status != "ok":
        print(f"  error: {result.get('error')}")
        return

    overall_service = result["metrics"]["overall_service"]
    print(
        "  service metrics:"
        f" AC@1={fmt(overall_service['ac@1'])}"
        f" AC@3={fmt(overall_service['ac@3'])}"
        f" AC@5={fmt(overall_service['ac@5'])}"
        f" Avg@5={fmt(overall_service['avg@5'])}"
    )
    print(
        f"  speed: {fmt(result['avg_speed_sec'])} sec/case"
        f" | success={result['num_successful']}/{result['num_cases']}"
        f" | failed={result['num_failed']}"
    )


def main() -> int:
    args = parse_args()
    methods = resolve_methods(args.profile, args.methods)
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    case_paths = discover_case_paths(dataset_root, args.limit_cases)

    ensure_rcaeval_importable(args.rcaeval_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_dir / f"rcaeval_{args.dataset}_{timestamp}"
    payload: Dict[str, object] = {
        "metadata": {
            "dataset": args.dataset,
            "dataset_root": str(dataset_root),
            "rcaeval_root": str(args.rcaeval_root),
            "methods": methods,
            "length_minutes": args.length,
            "num_cases": len(case_paths),
            "timestamp": timestamp,
        },
        "methods": {},
    }

    print("=" * 72)
    print("RCAEval Baseline Runner")
    print("=" * 72)
    print(f"Dataset      : {args.dataset}")
    print(f"Dataset root : {dataset_root}")
    print(f"RCAEval root : {args.rcaeval_root}")
    print(f"Cases        : {len(case_paths)}")
    print(f"Methods      : {', '.join(methods)}")

    for method_name in methods:
        try:
            method_fn = load_method(method_name, args.rcaeval_root)
            result = run_single_method(
                method_name=method_name,
                method_fn=method_fn,
                case_paths=case_paths,
                dataset=args.dataset,
                length_minutes=args.length,
                rcaeval_root=args.rcaeval_root,
                include_case_records=args.include_case_records,
                fail_fast=args.fail_fast,
                show_progress=not args.no_progress,
            )
            result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - depends on external baseline availability
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






