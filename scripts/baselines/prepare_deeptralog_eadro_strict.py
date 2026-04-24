from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEEPTRALOG_ROOT = WORKSPACE_ROOT / "external" / "DeepTraLog"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from data_eadro.preprocess_lazy import (  # noqa: E402
    CaseSource,
    enumerate_case_sources,
    normalize_service_name,
    read_json_from_zip,
)


DEFAULT_ZIP_PATH = PROJECT_ROOT / "datasets" / "Eadro" / "downloads" / "SN Dataset.zip"
DEFAULT_STRICT_EXPORT_DIR = PROJECT_ROOT / "artifacts" / "baselines" / "eadro_sn_strict_protocol_s42"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "baselines" / "deeptralog_eadro_strict_s42" / "ProcessedData"

RELATION_FILES = [
    "0_0_list.txt",
    "0_1_list.txt",
    "0_2_list.txt",
    "0_3_list.txt",
    "0_4_list.txt",
    "0_5_list.txt",
    "0_6_list.txt",
    "0_7_list.txt",
    "1_0_list.txt",
    "1_1_list.txt",
    "1_2_list.txt",
    "1_3_list.txt",
    "1_4_list.txt",
    "1_5_list.txt",
    "1_6_list.txt",
    "1_7_list.txt",
    "2_0_list.txt",
    "2_1_list.txt",
    "2_2_list.txt",
    "2_3_list.txt",
    "2_4_list.txt",
    "2_5_list.txt",
    "2_6_list.txt",
    "2_7_list.txt",
    "3_0_list.txt",
    "3_1_list.txt",
    "3_2_list.txt",
    "3_3_list.txt",
    "3_4_list.txt",
    "3_5_list.txt",
    "3_6_list.txt",
    "3_7_list.txt",
    "4_0_list.txt",
    "4_1_list.txt",
    "4_2_list.txt",
    "4_3_list.txt",
    "4_4_list.txt",
    "4_5_list.txt",
    "4_6_list.txt",
    "4_7_list.txt",
    "5_0_list.txt",
    "5_1_list.txt",
    "5_2_list.txt",
    "5_3_list.txt",
    "5_4_list.txt",
    "5_5_list.txt",
    "5_6_list.txt",
    "5_7_list.txt",
    "6_0_list.txt",
    "6_1_list.txt",
    "6_2_list.txt",
    "6_3_list.txt",
    "6_4_list.txt",
    "6_5_list.txt",
    "6_6_list.txt",
    "6_7_list.txt",
    "7_2_list.txt",
    "7_4_list.txt",
    "7_5_list.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an isolated DeepTraLog-style ProcessedData package for Eadro-SN strict split. "
            "This exporter reuses the current strict case split and only reads raw spans.json traces."
        )
    )
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--strict-export-dir", type=Path, default=DEFAULT_STRICT_EXPORT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variant",
        choices=["SN"],
        default="SN",
    )
    parser.add_argument(
        "--max-traces-per-case",
        type=int,
        default=128,
        help=(
            "Deterministic cap for smoke/feasibility runs. "
            "Use <=0 to export every trace from every selected case."
        ),
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_bucket(text: str, modulo: int) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 0x7FFFFFFF
    return value % modulo


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "feature_list").mkdir(parents=True, exist_ok=True)
    (path / "data_splits").mkdir(parents=True, exist_ok=True)


def load_case_split_map(strict_export_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        npz_path = strict_export_dir / f"{split_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing strict export split: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as payload:
            case_names = payload["case_names"].tolist()
        case_prefixes = {str(name).rsplit("_w", 1)[0] for name in case_names}
        for case_prefix in case_prefixes:
            existing = split_map.get(case_prefix)
            if existing is not None and existing != split_name:
                raise ValueError(f"Case {case_prefix} appears in both {existing} and {split_name}")
            split_map[case_prefix] = split_name
    return split_map


def load_case_spans_only(zf: zipfile.ZipFile, case: CaseSource) -> list[dict[str, Any]]:
    if case.source_kind == "zip_dir":
        payload = read_json_from_zip(zf, f"{case.case_prefix}spans.json")
        return payload if isinstance(payload, list) else []

    assert case.tar_member_path
    with zf.open(case.tar_member_path) as tar_fp:
        with tarfile.open(fileobj=tar_fp, mode="r|xz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = member.name.replace("\\", "/")
                if not member_name.endswith("/spans.json"):
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    break
                payload = json.loads(extracted.read())
                return payload if isinstance(payload, list) else []
    raise FileNotFoundError(f"spans.json not found for case {case.case_key}")


def select_trace_indices(total: int, max_items: int) -> list[int]:
    if max_items <= 0 or total <= max_items:
        return list(range(total))
    indices = np.linspace(0, total - 1, num=max_items, dtype=np.int64)
    deduped: list[int] = []
    seen = set()
    for idx in indices.tolist():
        if idx in seen:
            continue
        deduped.append(int(idx))
        seen.add(int(idx))
    return deduped


def _first_parent_span_id(span: dict[str, Any]) -> str | None:
    for ref in span.get("references", []) or []:
        if ref.get("refType") == "CHILD_OF" and ref.get("spanID"):
            return str(ref["spanID"])
    return None


def _trace_to_graph_rows(trace: dict[str, Any], gid: int, source_case: str, split_name: str, trace_bool: bool, error_trace_type: str) -> tuple[list[list[Any]], dict[str, list[str]], dict[str, Any]] | None:
    processes = trace.get("processes", {}) or {}
    spans = trace.get("spans", []) or []
    if not isinstance(processes, dict) or not isinstance(spans, list) or not spans:
        return None

    proc_to_service = {
        str(pid): normalize_service_name(proc.get("serviceName", "unknown"))
        for pid, proc in processes.items()
        if isinstance(proc, dict)
    }
    span_by_id = {str(span.get("spanID")): span for span in spans if span.get("spanID")}
    start_values = [int(span.get("startTime", 0) or 0) for span in spans]
    trace_start_us = min(start_values) if start_values else 0

    child_counts: Counter[str] = Counter()
    edges: list[tuple[int, int, str]] = []
    node_rows: list[list[Any]] = []
    relation_buffers: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    span_local_idx: dict[str, int] = {}

    for local_idx, span in enumerate(spans):
        span_id = str(span.get("spanID", f"span_{local_idx}"))
        span_local_idx[span_id] = local_idx

    for span in spans:
        child_id = str(span.get("spanID", ""))
        parent_id = _first_parent_span_id(span)
        if not child_id or not parent_id or parent_id not in span_local_idx:
            continue
        child_counts[parent_id] += 1
        parent_span = span_by_id.get(parent_id)
        if parent_span is None:
            continue
        src_idx = span_local_idx[parent_id]
        dst_idx = span_local_idx[child_id]
        src_service = proc_to_service.get(str(parent_span.get("processID", "")), "unknown")
        dst_service = proc_to_service.get(str(span.get("processID", "")), "unknown")
        relation_name = f"{stable_bucket(src_service, 8)}_{stable_bucket(dst_service, 8)}"
        if relation_name in RELATION_FILES:
            relation_buffers[relation_name][src_idx].append(dst_idx)
        edges.append((src_idx, dst_idx, relation_name))

    for local_idx, span in enumerate(spans):
        span_id = str(span.get("spanID", f"span_{local_idx}"))
        proc_id = str(span.get("processID", ""))
        service = proc_to_service.get(proc_id, "unknown")
        operation = str(span.get("operationName", "op"))
        start_us = int(span.get("startTime", 0) or 0)
        duration_us = int(span.get("duration", 0) or 0)
        start_ms = max(0.0, (start_us - trace_start_us) / 1000.0)
        duration_ms = max(0.0, duration_us / 1000.0)
        finish_ms = start_ms + duration_ms
        node_rows.append(
            [
                gid,
                local_idx,
                float(start_ms),
                float(finish_ms),
                float(duration_ms),
                float(child_counts.get(span_id, 0)),
                float(stable_bucket(service, 8)),
                float(stable_bucket(operation, 256)),
                float(stable_bucket(service + "::" + operation, 1024)),
            ]
        )

    relation_lines: dict[str, list[str]] = {}
    for relation_name, src_map in relation_buffers.items():
        relation_lines[relation_name] = [
            f"{gid}:{src_idx}:{','.join(str(x) for x in sorted(dst_list))}\n"
            for src_idx, dst_list in sorted(src_map.items())
        ]

    record = {
        "trace_id": gid,
        "trace_bool": bool(trace_bool),
        "error_trace_type": error_trace_type,
        "source_case": source_case,
        "preset_split": split_name,
        "source_trace_id": str(trace.get("traceID", "")),
        "num_nodes": len(node_rows),
        "num_edges": len(edges),
    }
    return node_rows, relation_lines, record


def write_space_separated_ids(path: Path, ids: Iterable[int]) -> None:
    values = [str(int(x)) for x in ids]
    path.write_text(" ".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    zip_path = args.zip_path.resolve()
    strict_export_dir = args.strict_export_dir.resolve()
    output_root = args.output_root.resolve()

    ensure_output_root(output_root, overwrite=args.overwrite)
    case_split_map = load_case_split_map(strict_export_dir)

    cases = enumerate_case_sources(zip_path, variant=args.variant, case_types="all")
    selected_cases = [case for case in cases if case.case_name_prefix in case_split_map]
    if args.max_cases > 0:
        selected_cases = selected_cases[: args.max_cases]
    if not selected_cases:
        raise ValueError("No Eadro cases matched the strict export split.")

    relation_buffers: dict[str, list[str]] = {name: [] for name in RELATION_FILES}
    trace_rows: list[dict[str, Any]] = []
    node_rows: list[list[Any]] = []
    export_case_rows: list[dict[str, Any]] = []

    gid = 0
    with zipfile.ZipFile(zip_path) as zf:
        for case in selected_cases:
            split_name = case_split_map[case.case_name_prefix]
            spans_payload = load_case_spans_only(zf, case)
            trace_indices = select_trace_indices(len(spans_payload), int(args.max_traces_per_case))
            exported_count = 0

            for trace_idx in trace_indices:
                graph_rows = _trace_to_graph_rows(
                    trace=spans_payload[trace_idx],
                    gid=gid,
                    source_case=case.case_name_prefix,
                    split_name=split_name,
                    trace_bool=case.is_normal,
                    error_trace_type=case.split_key,
                )
                if graph_rows is None:
                    continue
                node_chunk, relation_chunk, record = graph_rows
                node_rows.extend(node_chunk)
                for relation_name, lines in relation_chunk.items():
                    relation_buffers[relation_name].extend(lines)
                trace_rows.append(record)
                gid += 1
                exported_count += 1

            export_case_rows.append(
                {
                    "case_name_prefix": case.case_name_prefix,
                    "split": split_name,
                    "is_normal": bool(case.is_normal),
                    "total_raw_traces": int(len(spans_payload)),
                    "exported_traces": int(exported_count),
                }
            )
            print(
                f"Exported case {case.case_name_prefix}: "
                f"{exported_count}/{len(spans_payload)} traces -> split={split_name}",
                flush=True,
            )

    trace_info_path = output_root / "trace_info.csv"
    with trace_info_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trace_id",
                "trace_bool",
                "error_trace_type",
                "source_case",
                "preset_split",
                "source_trace_id",
                "num_nodes",
                "num_edges",
            ],
        )
        writer.writeheader()
        writer.writerows(trace_rows)

    node_embedding_path = output_root / "node_embedding.csv"
    with node_embedding_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gid", "dst_id", "f1", "f2", "f3", "f4", "f5", "f6", "f7"])
        writer.writerows(node_rows)

    for relation_name, lines in relation_buffers.items():
        (output_root / relation_name).write_text("".join(lines), encoding="utf-8")

    train_ids = [int(row["trace_id"]) for row in trace_rows if row["preset_split"] == "train" and bool(row["trace_bool"])]
    val_ids = [int(row["trace_id"]) for row in trace_rows if row["preset_split"] == "val"]
    test_ids = [int(row["trace_id"]) for row in trace_rows if row["preset_split"] == "test"]
    write_space_separated_ids(output_root / "data_splits" / "rep_model_train_gid_list.txt", train_ids)
    write_space_separated_ids(output_root / "data_splits" / "clf_eval_gid_list.txt", val_ids)
    write_space_separated_ids(output_root / "data_splits" / "clf_test_gid_list.txt", test_ids)

    split_counter = Counter(row["preset_split"] for row in trace_rows)
    normal_counter = Counter((row["preset_split"], bool(row["trace_bool"])) for row in trace_rows)
    summary = {
        "dataset": "Eadro-SN",
        "protocol_alignment": "strict case-level split reused from eadro_sn_strict_protocol_s42",
        "zip_path": str(zip_path),
        "strict_export_dir": str(strict_export_dir),
        "output_root": str(output_root),
        "variant": args.variant,
        "max_traces_per_case": int(args.max_traces_per_case),
        "num_cases": len(selected_cases),
        "num_graphs": len(trace_rows),
        "num_node_rows": len(node_rows),
        "split_graph_counts": {k: int(v) for k, v in sorted(split_counter.items())},
        "split_label_counts": {
            f"{split_name}:{'normal' if is_normal else 'anomaly'}": int(count)
            for (split_name, is_normal), count in sorted(normal_counter.items())
        },
        "train_benign_graphs": len(train_ids),
        "val_graphs": len(val_ids),
        "test_graphs": len(test_ids),
        "cases": export_case_rows,
        "artifacts": {
            "trace_info_csv": str(trace_info_path),
            "node_embedding_csv": str(node_embedding_path),
        },
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
