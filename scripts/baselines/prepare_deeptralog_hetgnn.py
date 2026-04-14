from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEEPTRALOG_ROOT = WORKSPACE_ROOT / "external" / "DeepTraLog"
GRAPHDATA_ZIP = DEEPTRALOG_ROOT / "GraphData" / "graphdata_full.zip"
OUTPUT_ROOT = DEEPTRALOG_ROOT / "HetGNN" / "ProcessedData"

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
    parser = argparse.ArgumentParser(description="Prepare DeepTraLog HetGNN data from GraphData.")
    parser.add_argument("--graphdata-zip", type=Path, default=GRAPHDATA_ZIP)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-graphs", type=int, default=2000)
    parser.add_argument("--copy-meta", action="store_true", help="Copy id_service/id_url CSVs into the output directory.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "feature_list").mkdir(parents=True, exist_ok=True)
    (path / "data_splits").mkdir(parents=True, exist_ok=True)


def stable_bucket(text: str, modulo: int) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 0x7FFFFFFF
    return value % modulo


def graph_feature_row(node: Dict[str, Any]) -> list[float]:
    info = node.get("info", {}) if isinstance(node, dict) else {}
    children = node.get("children", []) if isinstance(node, dict) else []
    started = float(info.get("started", 0.0) or 0.0)
    finished = float(info.get("finished", started) or started)
    duration = max(0.0, finished - started)
    service = str(info.get("service", "unknown"))
    name = str(info.get("name", "span"))
    return [
        started,
        finished,
        duration,
        float(len(children)),
        float(stable_bucket(service, 8)),
        float(stable_bucket(name, 256)),
        float(stable_bucket(service + "::" + name, 1024)),
    ]


def flatten_trace(node: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []

    def visit(cur: Dict[str, Any], parent_idx: int | None) -> int:
        idx = len(nodes)
        nodes.append(cur)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in cur.get("children", []) or []:
            if isinstance(child, dict):
                visit(child, idx)
        return idx

    visit(node, None)
    return nodes, edges


def iter_graph_records(graphdata_zip: Path, max_graphs: int) -> Iterable[tuple[int, dict[str, Any], str]]:
    with ZipFile(graphdata_zip) as zf:
        graph_files = [name for name in zf.namelist() if name.startswith("process") and name.endswith(".jsons")]
        graph_files.sort()
        gid = 0
        for member in graph_files:
            with zf.open(member) as fh:
                for raw in fh:
                    if gid >= max_graphs:
                        return
                    payload = json.loads(raw.decode("utf-8"))
                    yield gid, payload, member
                    gid += 1


def main() -> int:
    args = parse_args()
    if not args.graphdata_zip.exists():
        raise FileNotFoundError(f"GraphData zip not found: {args.graphdata_zip}")

    ensure_output_root(args.output_root, args.overwrite)
    relation_buffers: dict[str, list[str]] = {name: [] for name in RELATION_FILES}
    trace_rows: list[list[Any]] = []
    node_rows: list[list[Any]] = []
    graph_count = 0

    for gid, payload, member in iter_graph_records(args.graphdata_zip, args.max_graphs):
        trace_id = int(gid)
        trace_bool = bool(payload.get("trace_bool", False))
        error_trace_type = str(payload.get("error_trace_type", member.replace(".jsons", "")))
        trace_rows.append([trace_id, trace_bool, error_trace_type, member])

        nodes, edges = flatten_trace(payload)
        node_types: list[int] = []
        for local_id, node in enumerate(nodes):
            features = graph_feature_row(node)
            node_rows.append([trace_id, local_id, *features])
            info = node.get("info", {}) if isinstance(node, dict) else {}
            node_types.append(stable_bucket(str(info.get("service", "unknown")), 8))

        relation_map: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for src_idx, dst_idx in edges:
            src_type = node_types[src_idx]
            dst_type = node_types[dst_idx]
            relation_map[f"{src_type}_{dst_type}"][src_idx].append(dst_idx)

        for relation_name, src_map in relation_map.items():
            if relation_name not in relation_buffers:
                continue
            for src_idx, dst_list in src_map.items():
                relation_buffers[relation_name].append(
                    f"{trace_id}:{src_idx}:{','.join(str(x) for x in dst_list)}\n"
                )

        graph_count += 1
        if graph_count % 500 == 0:
            print(f"Processed {graph_count} graphs ...")

    # Write trace info.
    trace_info_path = args.output_root / "trace_info.csv"
    with trace_info_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trace_id", "trace_bool", "error_trace_type", "source_member"])
        writer.writerows(trace_rows)

    # Write node embeddings.
    node_embedding_path = args.output_root / "node_embedding.csv"
    with node_embedding_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gid", "dst_id", "f1", "f2", "f3", "f4", "f5", "f6", "f7"])
        writer.writerows(node_rows)

    # Write relation files.
    for relation_name, lines in relation_buffers.items():
        relation_path = args.output_root / relation_name
        relation_path.write_text("".join(lines), encoding="utf-8")

    if args.copy_meta:
        with ZipFile(args.graphdata_zip) as zf:
            for meta_name in ["id_service.csv", "id_url+temp.csv", "id_url+type.csv"]:
                target = args.output_root / meta_name
                target.write_bytes(zf.read(meta_name))

    summary = {
        "graph_count": graph_count,
        "trace_info_path": str(trace_info_path),
        "node_embedding_path": str(node_embedding_path),
        "relation_files": len(RELATION_FILES),
        "output_root": str(args.output_root),
    }
    (args.output_root / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
