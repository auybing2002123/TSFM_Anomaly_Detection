from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SERVICEANOMALY_ROOT = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "ServiceAnomaly"
DEFAULT_SAMPLE_DIR = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "data" / "TrainTicket" / "baseline" / "003"
DEFAULT_TRACE_FILE = DEFAULT_SAMPLE_DIR / "trace" / "traces.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "external" / "ServiceAnomaly" / "_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the ServiceAnomaly baseline.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-source", default="ts-travel-service")
    parser.add_argument("--edge-destination", default="ts-train-service")
    return parser.parse_args()


def install_headless_shims(sample_dir: Path) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.MessageBoxW = lambda *args, **kwargs: 1  # type: ignore[assignment]
    except Exception:
        pass

    import tkinter.filedialog as filedialog

    filedialog.askdirectory = lambda: str(sample_dir)  # type: ignore[assignment]


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    trace_file = args.trace_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SERVICEANOMALY_ROOT.exists():
        raise FileNotFoundError(f"ServiceAnomaly repo not found: {SERVICEANOMALY_ROOT}")
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    if str(SERVICEANOMALY_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICEANOMALY_ROOT))

    install_headless_shims(sample_dir)

    from DAG import DAG
    from MetericsProfiling import MetericsProfiling
    from MetricLearning import MetricLearning
    from Test import Test

    old_cwd = Path.cwd()
    os.chdir(sample_dir)
    try:
        with trace_file.open("r", encoding="utf-8") as f:
            trace = json.load(f)
        trace = json.loads(trace)
        relations = trace["data"]

        dag = DAG()
        dag.GrphGeneration(relations)
        dag.GraphVisualize()

        if not dag.edges:
            raise RuntimeError("No edges were generated from the trace graph.")

        source = args.edge_source
        destination = args.edge_destination

        metrics = MetericsProfiling()
        grouped = metrics.MetricGrouping(destination)

        ml = MetricLearning()
        ml.LinearRelationship_Visualize(grouped, source, destination)
        ml.get_top_abs_correlations(grouped, 0.6)
        ml.NonLinearRelationship_Visualize(grouped, source, destination)
        ml.get_non_linear(grouped, 0.47, 0.6)

        smoke_test = Test()
        relation_list = [{"parent": f"{source}.default", "child": f"{destination}.default"}]
        smoke_test.test_set_relation(relation_list)
        test_metrics = smoke_test.test_metric_gathering(destination)

        summary = {
            "baseline": "ServiceAnomaly",
            "sample_dir": str(sample_dir),
            "trace_file": str(trace_file),
            "graph_edges": len(dag.edges),
            "source": source,
            "destination": destination,
            "metric_columns": list(grouped.columns),
            "linear_relations": len(getattr(ml, "lin_ls", [])),
            "nonlinear_relations": len(getattr(ml, "nonlin_ls", [])),
            "test_metrics_keys": sorted(test_metrics.keys()),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = output_dir / f"serviceanomaly_smoke_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("ServiceAnomaly smoke completed")
        print(f"Run dir : {run_dir}")
        print(f"Summary : {run_dir / 'summary.json'}")
        print(f"Summary data: {json.dumps(summary, ensure_ascii=False)}")
        return 0
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
