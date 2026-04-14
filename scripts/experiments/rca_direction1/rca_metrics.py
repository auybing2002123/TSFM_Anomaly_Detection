from __future__ import annotations

from pathlib import Path
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RCAEVAL_ROOT = PROJECT_ROOT.parent / "RCAEval"
for path in (PROJECT_ROOT, RCAEVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from RCAEval.benchmark.evaluation import Evaluator  # noqa: E402
from RCAEval.classes.graph import Node  # noqa: E402


def _normalize_case_id(case_name: str) -> str:
    return case_name.rsplit("_w", 1)[0]


def _extract_window_index(case_name: str) -> int:
    suffix = case_name.rsplit("_w", 1)[-1]
    return int(suffix)


def _root_service_from_labels(gt_real: np.ndarray, service_names: Sequence[str]) -> str | None:
    root_indices = np.where(gt_real[:, 1] == 1)[0]
    if len(root_indices) == 0:
        return None
    return service_names[int(root_indices[0])]


def _select_case_windows(
    windows: List[Tuple[int, np.ndarray]],
    strategy: str,
) -> List[np.ndarray]:
    sorted_windows = [scores for _, scores in sorted(windows, key=lambda item: item[0])]

    if strategy == "all":
        return sorted_windows
    if strategy == "earliest":
        return sorted_windows[:1]
    if strategy.startswith("first_"):
        top_k = int(strategy.split("_", 1)[1])
        return sorted_windows[:top_k]
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


def aggregate_case_scores(
    case_names: Iterable[str],
    scores: np.ndarray,
    labels: np.ndarray,
    service_names: Sequence[str],
    strategy: str = "all",
) -> Dict[str, Dict[str, object]]:
    """
    Aggregate per-window service scores into case-level RCA scores.

    Only anomalous windows (those with a root label in `groundtruth_real`) are kept.
    For each case we average service scores across anomalous windows so the final
    RCA evaluation aligns with the official RCAEval "one ranking per case" setup.
    """
    grouped_scores: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    grouped_roots: Dict[str, str] = {}

    for case_name, score_row, label_row in zip(case_names, scores, labels):
        root_service = _root_service_from_labels(label_row, service_names)
        if root_service is None:
            continue
        case_id = _normalize_case_id(case_name)
        window_idx = _extract_window_index(case_name)
        grouped_scores[case_id].append((window_idx, np.asarray(score_row, dtype=np.float64)))
        grouped_roots[case_id] = root_service

    case_payload: Dict[str, Dict[str, object]] = {}
    for case_id, score_list in grouped_scores.items():
        selected_scores = _select_case_windows(score_list, strategy)
        case_payload[case_id] = {
            "service_scores": np.mean(np.stack(selected_scores, axis=0), axis=0),
            "root_service": grouped_roots[case_id],
            "num_windows": len(score_list),
            "selected_windows": len(selected_scores),
            "strategy": strategy,
        }
    return case_payload


def compute_service_rca_metrics(
    case_payload: Dict[str, Dict[str, object]],
    service_names: Sequence[str],
) -> Dict[str, object]:
    """
    Compute official RCAEval service-level AC@k / Avg@5 from case-level scores.
    """
    evaluator = Evaluator()
    per_case: List[Dict[str, object]] = []

    for case_id in sorted(case_payload.keys()):
        payload = case_payload[case_id]
        scores = np.asarray(payload["service_scores"], dtype=np.float64)
        root_service = str(payload["root_service"])
        sorted_indices = np.argsort(-scores)
        ranked_nodes = [Node(service_names[idx], "service") for idx in sorted_indices[:5]]
        answer = Node(root_service, "service")
        evaluator.add_case(ranked_nodes, answer)

        ranked_services = [service_names[idx] for idx in sorted_indices]
        try:
            rank = ranked_services.index(root_service) + 1
        except ValueError:
            rank = None
        per_case.append(
            {
                "case_id": case_id,
                "root_service": root_service,
                "rank": rank,
                "top5_services": ranked_services[:5],
                "top5_scores": [float(scores[idx]) for idx in sorted_indices[:5]],
                "num_windows": int(payload["num_windows"]),
                "selected_windows": int(payload.get("selected_windows", payload["num_windows"])),
                "strategy": payload.get("strategy", "all"),
            }
        )

    metrics = {
        "count": evaluator.num,
        "ac@1": evaluator.accuracy_service(1),
        "ac@3": evaluator.accuracy_service(3),
        "ac@5": evaluator.accuracy_service(5),
        "avg@5": evaluator.average_service(5),
        "per_case": per_case,
    }
    return metrics


def format_service_rca_metrics(metrics: Dict[str, object]) -> str:
    def _fmt(value: object) -> str:
        if value is None:
            return "null"
        return f"{float(value):.4f}"

    return (
        f"cases={metrics['count']}, "
        f"AC@1={_fmt(metrics['ac@1'])}, "
        f"AC@3={_fmt(metrics['ac@3'])}, "
        f"AC@5={_fmt(metrics['ac@5'])}, "
        f"Avg@5={_fmt(metrics['avg@5'])}"
    )


__all__ = [
    "aggregate_case_scores",
    "compute_service_rca_metrics",
    "format_service_rca_metrics",
]
