from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_rcaeval.dataset_loader import load_rcaeval_lazy_stratified  # noqa: E402
from scripts.baselines.run_rcaeval_baselines import (  # noqa: E402
    ensure_rcaeval_importable,
    load_method,
    print_method_summary,
    resolve_dataset_root,
    run_single_method,
    write_summary_files,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "rca_direction1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RCAEval baselines on the exact RE2-TT stratified split used by TSFM RCA experiments."
    )
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["re2-tt"], default="re2-tt")
    parser.add_argument("--methods", nargs="+", default=["baro", "tracerca"])
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--rcaeval-root", type=Path, default=PROJECT_ROOT.parent / "RCAEval")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--include-case-records",
        action="store_true",
        help="Persist per-case top-5 predictions for downstream strict aggregation evals.",
    )
    return parser.parse_args()


def get_split_case_ids(data_dir: str, split: str, seed: int) -> List[str]:
    splits = load_rcaeval_lazy_stratified(data_dir, seed=seed)
    if split == "all":
        entries = []
        for key in ("train", "val", "test"):
            entries.extend(splits[key].entries)
    else:
        entries = splits[split].entries

    case_ids = sorted({entry["case_name"].rsplit("_w", 1)[0] for entry in entries})
    return case_ids


def resolve_case_path(dataset_root: Path, case_id: str) -> Path:
    fault_dir, case_num = case_id.split("/", 1)
    candidate = dataset_root / fault_dir / case_num / "simple_metrics.csv"
    if candidate.exists():
        return candidate

    fallback = dataset_root / fault_dir / case_num / "data.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"No simple_metrics.csv/data.csv found for case {case_id} under {dataset_root}")


def main() -> int:
    args = parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    case_ids = get_split_case_ids(args.data_dir, args.split, args.seed)
    case_paths = [resolve_case_path(dataset_root, case_id) for case_id in case_ids]

    ensure_rcaeval_importable(args.rcaeval_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_dir / f"rcaeval_{args.dataset}_{args.split}_aligned_{timestamp}"
    payload: Dict[str, object] = {
        "metadata": {
            "dataset": args.dataset,
            "dataset_root": str(dataset_root),
            "rcaeval_root": str(args.rcaeval_root),
            "processed_data_dir": args.data_dir,
            "split": args.split,
            "seed": args.seed,
            "methods": args.methods,
            "length_minutes": args.length,
            "num_cases": len(case_paths),
            "timestamp": timestamp,
            "note": "Case set is aligned to the TSFM stratified split used by RCA Direction 1 experiments.",
            "case_ids": case_ids,
        },
        "methods": {},
    }

    print("=" * 72)
    print("RCAEval Baseline Runner (Aligned RE2-TT Split)")
    print("=" * 72)
    print(f"Dataset      : {args.dataset}")
    print(f"Split        : {args.split}")
    print(f"Seed         : {args.seed}")
    print(f"Dataset root : {dataset_root}")
    print(f"RCAEval root : {args.rcaeval_root}")
    print(f"Cases        : {len(case_paths)}")
    print(f"Methods      : {', '.join(args.methods)}")

    for method_name in args.methods:
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
