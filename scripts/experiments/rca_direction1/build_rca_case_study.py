from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RCA direction-1 case study markdown")
    parser.add_argument("--root-head-summary", type=str, required=True)
    parser.add_argument("--prop-summary", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="first_3")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_map(summary: Dict, split: str, strategy: str, branch: str) -> Dict[str, Dict]:
    cases = summary["splits"][split][strategy][branch]["per_case"]
    return {case["case_id"]: case for case in cases}


def top3(case: Dict) -> str:
    return ", ".join(case["top5_services"][:3])


def compare_cases(
    root_head_cases: Dict[str, Dict],
    anomaly_cases: Dict[str, Dict],
    propagation_cases: Dict[str, Dict],
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    improved = []
    worsened = []
    propagation_worse = []

    for case_id in sorted(root_head_cases.keys()):
        rh = root_head_cases[case_id]
        an = anomaly_cases[case_id]
        pr = propagation_cases[case_id]

        if rh["rank"] < an["rank"]:
            improved.append((case_id, rh, an, pr))
        elif rh["rank"] > an["rank"]:
            worsened.append((case_id, rh, an, pr))

        if pr["rank"] > rh["rank"]:
            propagation_worse.append((case_id, rh, an, pr))

    return improved, worsened, propagation_worse


def render_case_block(title: str, cases: List[Tuple], limit: int = 3) -> List[str]:
    lines = [f"## {title}", ""]
    if not cases:
        lines.append("无。")
        lines.append("")
        return lines

    for case_id, rh, an, pr in cases[:limit]:
        lines.append(f"### `{case_id}`")
        lines.append("")
        lines.append(f"- `root service`: `{rh['root_service']}`")
        lines.append(
            f"- `anomaly-sort rank={an['rank']}`: {top3(an)}"
        )
        lines.append(
            f"- `root-head rank={rh['rank']}`: {top3(rh)}"
        )
        lines.append(
            f"- `propagation rank={pr['rank']}`: {top3(pr)}"
        )
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    root_summary = load_json(args.root_head_summary)
    prop_summary = load_json(args.prop_summary)

    root_head_cases = case_map(root_summary, args.split, args.strategy, "root_head")
    anomaly_cases = case_map(root_summary, args.split, args.strategy, "anomaly_sorting")
    propagation_cases = case_map(prop_summary, args.split, args.strategy, "root_head")

    improved, worsened, propagation_worse = compare_cases(
        root_head_cases,
        anomaly_cases,
        propagation_cases,
    )

    lines: List[str] = [
        "# RCA 方向1 Case Study",
        "",
        f"- split: `{args.split}`",
        f"- strategy: `{args.strategy}`",
        f"- root-head improved cases: `{len(improved)}`",
        f"- root-head worsened cases: `{len(worsened)}`",
        f"- propagation worse-than-root-head cases: `{len(propagation_worse)}`",
        "",
        "这份 case study 关注三个问题：",
        "- `root-head-only` 是否真的能把 root 提前，而不是只是复制 anomaly score。",
        "- 它在哪些 case 上比 anomaly-sort 更好。",
        "- propagation 版是如何把已经修正的 case 再次拉坏的。",
        "",
    ]

    lines.extend(render_case_block("Root Head 修正成功的代表案例", improved))
    lines.extend(render_case_block("Root Head 仍然失败的案例", worsened))
    lines.extend(render_case_block("Propagation 拉坏的代表案例", propagation_worse))

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved case study markdown to: {args.output}")


if __name__ == "__main__":
    main()
