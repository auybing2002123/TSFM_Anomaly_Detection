#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(r"E:\code\paper")
DEEPTRALOG_ROOT = ROOT / "external" / "DeepTraLog"
OFFICIAL_GRAPH_ZIP = DEEPTRALOG_ROOT / "GraphData" / "graphdata_full.zip"
MSDS_TRACE_ROOT = (
    ROOT
    / "code"
    / "TSFM_Anomaly_Detection"
    / "datasets"
    / "MSDS"
    / "traces"
)
RESULT_ROOT = DEEPTRALOG_ROOT / "results" / "deeptralog_probe"


@dataclass
class ProbeSample:
    source: str
    trace_id: str
    payload: dict[str, Any]


def _load_first_line_from_zip(zip_path: Path, member_name: str) -> dict[str, Any]:
    with ZipFile(zip_path) as zf:
        with zf.open(member_name) as fh:
            first_line = fh.readline().decode("utf-8").strip()
    return json.loads(first_line)


def load_official_sample() -> ProbeSample:
    sample = _load_first_line_from_zip(OFFICIAL_GRAPH_ZIP, "process0.jsons")
    return ProbeSample(
        source="official_graphdata",
        trace_id=str(sample.get("trace_id", "")),
        payload=sample,
    )


def load_msds_sample() -> ProbeSample:
    candidates = []
    for subdir in sorted(MSDS_TRACE_ROOT.iterdir()):
        if subdir.is_dir():
            candidates.extend(sorted(subdir.glob("*.json")))
    if not candidates:
        raise FileNotFoundError(f"No MSDS trace files found under {MSDS_TRACE_ROOT}")
    sample_path = candidates[0]
    with sample_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return ProbeSample(
        source=f"msds:{sample_path.parent.name}",
        trace_id=sample_path.stem,
        payload=payload,
    )


def summarize_schema(sample: dict[str, Any]) -> dict[str, Any]:
    top_keys = sorted(sample.keys())
    node_types = []
    if "node_info" in sample and isinstance(sample["node_info"], list):
        for node in sample["node_info"]:
            if isinstance(node, list) and len(node) > 4:
                node_types.append(node[4])
    return {
        "top_keys": top_keys,
        "node_count": len(sample.get("node_info", [])) if isinstance(sample.get("node_info"), list) else None,
        "edge_count": len(sample.get("edge_index", [])) if isinstance(sample.get("edge_index"), list) else None,
        "node_types": sorted(set(node_types)),
    }


def stable_bucket(text: str, modulo: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def flatten_msds_tree(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []

    def visit(node: dict[str, Any], parent_idx: int | None, depth: int, sibling_idx: int) -> int:
        cur_idx = len(nodes)
        info = node.get("info", {})
        children = node.get("children", []) or []
        service = info.get("service", "unknown")
        name = info.get("name", "span")
        started = info.get("started", 0)
        duration = info.get("ended", info.get("finished", started)) if isinstance(info, dict) else 0
        if isinstance(info, dict):
            finished = info.get("finished", started)
            duration = max(0, float(finished) - float(started))
        else:
            finished = started
        node_type = stable_bucket(service, 8)
        nodes.append(
            {
                "service": service,
                "name": name,
                "depth": depth,
                "sibling_idx": sibling_idx,
                "started": float(started),
                "finished": float(finished),
                "duration": float(duration),
                "node_type": node_type,
                "child_count": len(children),
            }
        )
        if parent_idx is not None:
            edges.append((parent_idx, cur_idx))
        for child_idx, child in enumerate(children):
            if isinstance(child, dict):
                visit(child, cur_idx, depth + 1, child_idx)
        return cur_idx

    visit(trace, None, 0, 0)
    return nodes, edges


def convert_msds_trace(trace: dict[str, Any], trace_id: str, trace_bool: bool, error_trace_type: str) -> dict[str, Any]:
    nodes, edges = flatten_msds_tree(trace)
    node_info = []
    for idx, node in enumerate(nodes):
        node_info.append(
            [
                float(idx),
                float(node["depth"]),
                float(node["sibling_idx"]),
                float(node["duration"]),
                int(node["node_type"]),
                float(node["child_count"]),
                float(stable_bucket(node["service"], 2048)),
            ]
        )

    return {
        "edge_index": [[int(s), int(d)] for s, d in edges],
        "edge_attr": [0 for _ in edges],
        "node_info": node_info,
        "trace_id": trace_id,
        "trace_bool": bool(trace_bool),
        "error_trace_type": error_trace_type,
    }


def build_report(sample: ProbeSample) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source": sample.source,
        "trace_id": sample.trace_id,
        "schema": summarize_schema(sample.payload),
    }
    if sample.source.startswith("msds:"):
        converted = convert_msds_trace(
            sample.payload,
            trace_id=sample.trace_id,
            trace_bool=True,
            error_trace_type=sample.source.split(":", 1)[1],
        )
        report["converted_schema"] = summarize_schema(converted)
        report["converted_relation_types"] = sorted(
            {
                (converted["node_info"][s][4], converted["node_info"][d][4])
                for s, d in converted["edge_index"]
            }
        )
        report["converted_node_count"] = len(converted["node_info"])
        report["converted_edge_count"] = len(converted["edge_index"])
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        out_file = RESULT_ROOT / f"{sample.trace_id}_converted.json"
        out_file.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
        report["converted_path"] = str(out_file)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepTraLog compatibility probe")
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Only inspect the official GraphData sample",
    )
    args = parser.parse_args()

    official = load_official_sample()
    report = {"official": build_report(official)}

    if not args.official_only:
        msds = load_msds_sample()
        report["msds"] = build_report(msds)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if "msds" in report:
        print()
        print("Compatibility verdict:")
        print("- Official DeepTraLog sample parses successfully.")
        print("- MSDS trace trees can be flattened into DeepTraLog-style graph JSON.")
        print("- Full training is still blocked by DeepTraLog's hardcoded preprocessing/training assumptions.")
        print("  In particular: relation file expectations, fixed train/eval/test split sizes, and type remapping are not turnkey.")
    else:
        print()
        print("Compatibility verdict:")
        print("- Official DeepTraLog sample parses successfully.")
        print("- MSDS was not inspected in this run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
