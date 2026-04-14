from __future__ import annotations

import argparse
import contextlib
import ctypes
import ast
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SERVICEANOMALY_ROOT = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "ServiceAnomaly"
TRAIN_TICKET_ROOT = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "data" / "TrainTicket"
DEFAULT_BASELINE_DIR = TRAIN_TICKET_ROOT / "baseline" / "003"
DEFAULT_TEST_ROOT = TRAIN_TICKET_ROOT / "test"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "_batch"


TRAIN_TICKET_EDGES = [
    ("ts-travel-service.default", "ts-train-service.default"),
    ("ts-travel-service.default", "ts-seat-service.default"),
    ("ts-travel-service.default", "ts-route-service.default"),
    ("ts-travel-service.default", "ts-ticketinfo-service.default"),
    ("ts-travel2-service.default", "ts-ticketinfo-service.default"),
    ("ts-travel2-service.default", "ts-train-service.default"),
    ("ts-travel2-service.default", "ts-route-service.default"),
    ("ts-ticketinfo-service.default", "ts-basic-service.default"),
    ("ts-inside-payment-service.default", "ts-payment-service.default"),
    ("ts-inside-payment-service.default", "ts-order-service.default"),
    ("ts-inside-payment-service.default", "ts-order-other-service.default"),
    ("ts-preserve-service.default", "ts-user-service.default"),
    ("ts-preserve-service.default", "ts-security-service.default"),
    ("ts-preserve-service.default", "ts-contacts-service.default"),
    ("ts-preserve-service.default", "ts-travel-service.default"),
    ("ts-preserve-service.default", "ts-seat-service.default"),
    ("ts-preserve-service.default", "ts-assurance-service.default"),
    ("ts-preserve-service.default", "ts-food-service.default"),
    ("ts-preserve-service.default", "ts-ticketinfo-service.default"),
    ("ts-preserve-service.default", "ts-station-service.default"),
    ("ts-preserve-service.default", "ts-consign-service.default"),
    ("ts-preserve-service.default", "ts-order-service.default"),
    ("ts-seat-service.default", "ts-order-other-service.default"),
    ("ts-seat-service.default", "ts-travel-service.default"),
    ("ts-seat-service.default", "ts-config-service.default"),
    ("ts-seat-service.default", "ts-order-service.default"),
    ("ts-basic-service.default", "ts-train-service.default"),
    ("ts-basic-service.default", "ts-route-service.default"),
    ("ts-basic-service.default", "ts-station-service.default"),
    ("ts-food-service.default", "ts-station-service.default"),
    ("ts-food-service.default", "ts-travel-service.default"),
    ("ts-ui-dashboard.default", "ts-preserve-service.default"),
    ("ts-ui-dashboard.default", "ts-rebook-service.default"),
    ("ts-ui-dashboard.default", "ts-contacts-service.default"),
    ("ts-ui-dashboard.default", "ts-order-service.default"),
    ("ts-ui-dashboard.default", "ts-order-other-service.default"),
    ("ts-ui-dashboard.default", "ts-inside-payment-service.default"),
    ("ts-order-service.default", "ts-station-service.default"),
    ("ts-order-other-service.default", "ts-station-service.default"),
    ("ts-consign-service.default", "ts-consign-price-service.default"),
]


@dataclass
class CaseResult:
    case_id: str
    split: str
    label: int
    score: float
    graph_isomorphic: bool
    total_violations: int
    linear_violations: int
    nonlinear_violations: int
    runtime_ms: float
    status: str
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation for ServiceAnomaly on TrainTicket test cases.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap across normal + injected cases.")
    parser.add_argument("--source-service", default="ts-travel-service")
    parser.add_argument("--destination-service", default="ts-train-service")
    return parser.parse_args()


def install_headless_shims() -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import tkinter.filedialog as filedialog

    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.MessageBoxW = lambda *args, **kwargs: 1  # type: ignore[assignment]
    except Exception:
        pass

    filedialog.askdirectory = lambda: str(Path.cwd())  # type: ignore[assignment]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_trace(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(raw_text)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    raise ValueError(f"Unexpected trace format: {path}")


def load_case_relations(case_dir: Path) -> List[Dict[str, str]]:
    trace_file = case_dir / "trace.json"
    if not trace_file.exists():
        trace_file = case_dir / "trace" / "traces.json"
    if not trace_file.exists():
        trace_file = case_dir / "trace"
    trace_data = load_json_trace(trace_file)
    relations: List[Dict[str, str]] = []
    for item in trace_data:
        if "parent" in item and "child" in item:
            relations.append({"parent": item["parent"], "child": item["child"]})
    return relations


def iter_cases(test_root: Path) -> List[Tuple[str, Path, int]]:
    cases: List[Tuple[str, Path, int]] = []
    normal_root = test_root / "normal"
    injected_root = test_root / "injected faults"
    for case_dir in sorted([p for p in normal_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        cases.append((case_dir.name, case_dir, 0))
    for case_dir in sorted([p for p in injected_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        cases.append((case_dir.name, case_dir, 1))
    return cases


def collect_metric_files(case_dir: Path, destination: str) -> Dict[str, float]:
    metric_names = [
        "request_duration",
        "request_duratoin",
        "request_byte",
        "response_byte",
        "queue_size",
        "latency",
        "throughput",
    ]
    result: Dict[str, float] = {}
    for metric_name in metric_names:
        path = case_dir / f"{destination}_{metric_name}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, str):
            payload = json.loads(payload)
        values = payload["data"]["result"]
        merged: List[Sequence[Any]] = []
        for entry in values:
            merged.extend(entry["values"])
        if not merged:
            raise ValueError(f"No values in metric file: {path}")
        result[metric_name] = float(merged[0][1])
    if "request_duration" not in result and "request_duratoin" in result:
        result["request_duration"] = result["request_duratoin"]
    return result


def train_relations_for_baseline(baseline_dir: Path) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    if str(SERVICEANOMALY_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICEANOMALY_ROOT))

    from MetricLearning import MetricLearning
    from MetericsProfiling import MetericsProfiling

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline dir not found: {baseline_dir}")

    old_cwd = Path.cwd()
    os.chdir(baseline_dir)
    try:
        dag_edges = list(TRAIN_TICKET_EDGES)

        linear_relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        nonlinear_relations: Dict[Tuple[str, str], Dict[str, Any]] = {}

        metrics = MetericsProfiling()
        for source, destination in dag_edges:
            grouped = metrics.MetricGrouping(destination.split(".")[0])
            learner = MetricLearning()
            learner.LinearRelationship_Visualize(grouped, source, destination)
            learner.get_top_abs_correlations(grouped, 0.6)
            linear_relations[(source, destination)] = {
                "source": source,
                "destination": destination,
                "relations": list(learner.lin_ls),
            }
            learner.NonLinearRelationship_Visualize(grouped, source, destination)
            learner.get_non_linear(grouped, 0.47, 0.6)
            nonlinear_relations[(source, destination)] = {
                "source": source,
                "destination": destination,
                "relations": list(learner.nonlin_ls),
            }
    finally:
        os.chdir(old_cwd)

    return dag_edges, linear_relations, nonlinear_relations


def score_case(
    case_dir: Path,
    baseline_edges: Sequence[Tuple[str, str]],
    linear_relations: Mapping[Tuple[str, str], Dict[str, Any]],
    nonlinear_relations: Mapping[Tuple[str, str], Dict[str, Any]],
) -> Tuple[bool, int, int, int]:
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher

    relations = load_case_relations(case_dir)
    case_graph = nx.DiGraph()
    case_graph.add_edges_from((item["parent"], item["child"]) for item in relations)
    baseline_graph = nx.DiGraph()
    baseline_graph.add_edges_from(baseline_edges)
    graph_isomorphic = False
    try:
        graph_isomorphic = DiGraphMatcher(baseline_graph, case_graph).subgraph_is_isomorphic()
    except Exception:
        pass

    total_violations = 0
    linear_violations = 0
    nonlinear_violations = 0

    for source, destination in case_graph.edges:
        if (source, destination) not in linear_relations and (source, destination) not in nonlinear_relations:
            continue

        metrics = collect_metric_files(case_dir, destination.split(".")[0])

        lin_bundle = linear_relations.get((source, destination), {})
        for rel in lin_bundle.get("relations", []):
            try:
                metric1 = rel["metric1"]
                metric2 = rel["metric2"]
                coef = float(rel["coef"][0])
                intercept = float(rel["intercept"])
                error = float(rel["error"])
                y_test = float(metrics[metric1])
                x_test = float(metrics[metric2])
                y_predict = coef * x_test + intercept
                if abs(y_test - y_predict) > abs(error):
                    linear_violations += 1
                    total_violations += 1
            except Exception:
                continue

        nonlin_bundle = nonlinear_relations.get((source, destination), {})
        for rel in nonlin_bundle.get("relations", []):
            try:
                metric1 = rel["metric1"]
                metric2 = rel["metric2"]
                svr = rel["svr"]
                error = float(rel["error"])
                y_test = float(metrics[metric1])
                x_test = np.array([float(metrics[metric2])]).reshape(-1, 1)
                y_predict = svr.predict(x_test)
                if abs(y_test - y_predict) > abs(error):
                    nonlinear_violations += 1
                    total_violations += 1
            except Exception:
                continue

    return graph_isomorphic, total_violations, linear_violations, nonlinear_violations


def main() -> int:
    args = parse_args()
    install_headless_shims()

    baseline_dir = args.baseline_dir.resolve()
    test_root = args.test_root.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())

    baseline_edges, linear_relations, nonlinear_relations = train_relations_for_baseline(baseline_dir)
    cases = iter_cases(test_root)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    results: List[CaseResult] = []
    for case_name, case_dir, label in cases:
        start = time.perf_counter()
        try:
            graph_isomorphic, total_violations, linear_violations, nonlinear_violations = score_case(
                case_dir, baseline_edges, linear_relations, nonlinear_relations
            )
            status = "ok"
            error = None
            score = float(total_violations)
        except Exception as exc:
            graph_isomorphic = False
            total_violations = 0
            linear_violations = 0
            nonlinear_violations = 0
            status = "error"
            error = repr(exc)
            score = 0.0
        runtime_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            CaseResult(
                case_id=case_name,
                split="injected faults" if label else "normal",
                label=label,
                score=score,
                graph_isomorphic=graph_isomorphic,
                total_violations=total_violations,
                linear_violations=linear_violations,
                nonlinear_violations=nonlinear_violations,
                runtime_ms=runtime_ms,
                status=status,
                error=error,
            )
        )
        print(f"[{case_name}] label={label} score={score:.1f} graph_iso={graph_isomorphic} runtime={runtime_ms:.1f}ms status={status}")

    y_true = np.array([r.label for r in results], dtype=int)
    y_score = np.array([r.score for r in results], dtype=float)
    y_pred = (y_score > 0).astype(int)
    y_graph_pred = np.array([0 if r.graph_isomorphic else 1 for r in results], dtype=int)
    y_graph_score = np.array([1.0 if not r.graph_isomorphic else 0.0 for r in results], dtype=float)

    summary: Dict[str, Any] = {
        "baseline_dir": str(baseline_dir),
        "test_root": str(test_root),
        "num_cases": len(results),
        "num_normal": int((y_true == 0).sum()),
        "num_injected": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "graph_accuracy": float(accuracy_score(y_true, y_graph_pred)),
        "graph_precision": float(precision_score(y_true, y_graph_pred, zero_division=0)),
        "graph_recall": float(recall_score(y_true, y_graph_pred, zero_division=0)),
        "graph_f1": float(f1_score(y_true, y_graph_pred, zero_division=0)),
        "mean_runtime_ms": float(np.mean([r.runtime_ms for r in results])) if results else 0.0,
        "p95_runtime_ms": float(np.percentile([r.runtime_ms for r in results], 95)) if results else 0.0,
        "max_runtime_ms": float(np.max([r.runtime_ms for r in results])) if results else 0.0,
        "graph_isomorphic_rate": float(np.mean([1.0 if r.graph_isomorphic else 0.0 for r in results])) if results else 0.0,
    }
    try:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None
    except Exception:
        summary["roc_auc"] = None
    try:
        summary["graph_roc_auc"] = float(roc_auc_score(y_true, y_graph_score)) if len(set(y_true.tolist())) > 1 else None
    except Exception:
        summary["graph_roc_auc"] = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"serviceanomaly_trainticket_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    print("\nServiceAnomaly TrainTicket batch completed")
    print(f"Run dir : {run_dir}")
    print(f"Summary : {run_dir / 'summary.json'}")
    print(f"Metrics : {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
