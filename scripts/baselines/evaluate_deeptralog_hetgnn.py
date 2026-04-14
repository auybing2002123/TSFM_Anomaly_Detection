from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPTRALOG_CODE_ROOT = PROJECT_ROOT.parents[1] / "external" / "DeepTraLog" / "HetGNN" / "code"
if str(DEEPTRALOG_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPTRALOG_CODE_ROOT))

from HetGNN import model_class  # type: ignore  # noqa: E402
from args import read_args  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepTraLog HetGNN evaluation smoke")
    parser.add_argument("--data-path", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\ProcessedData"))
    parser.add_argument("--model-path", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke\HetGNN_0.pt"))
    parser.add_argument("--center", type=Path, default=Path(r"E:\code\paper\external\DeepTraLog\HetGNN\model_save_msds_smoke\HetGNN_SVDD_Center.pt"))
    parser.add_argument("--split", choices=["eval", "test"], default="test")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    sys.argv = [
        "evaluate_deeptralog_hetgnn.py",
        "--data_path", str(cli.data_path),
        "--model_path", str(cli.model_path),
        "--preprocess", "1",
        "--train_iter_n", "1",
        "--save_model_freq", "1",
        "--embed_d", "7",
        "--cuda", "0",
    ]
    args = read_args()

    model_object = model_class(args)
    model_object.model.load_state_dict(torch.load(cli.checkpoint, map_location="cpu"))
    model_object.model.svdd_center = torch.load(cli.center, map_location="cpu")
    model_object.model.eval()

    _, eval_gid_list, test_gid_list = model_object.train_eval_test_split()
    target_list = eval_gid_list if cli.split == "eval" else test_gid_list
    roc_auc, ap = model_object.eval_model(target_list)

    summary = {
        "split": cli.split,
        "checkpoint": str(cli.checkpoint),
        "center": str(cli.center),
        "roc_auc": float(roc_auc),
        "avg_precision": float(ap),
        "sample_count": int(len(target_list)),
    }

    out_dir = PROJECT_ROOT / "results" / "baselines" / "deeptralog_hetgnn_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cli.split}_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
