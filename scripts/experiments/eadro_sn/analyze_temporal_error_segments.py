from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual temporal errors from Eadro-SN replay events.")
    parser.add_argument("--events-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def window_index(source_id: str, fallback: int) -> int:
    match = re.search(r"_w(\d+)$", source_id)
    return int(match.group(1)) if match else fallback


def case_id(source_id: str) -> str:
    return source_id.rsplit("_w", 1)[0]


def binary_metrics(preds: list[int], labels: list[int]) -> dict:
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def contiguous_segments(rows: list[dict], key: str) -> list[dict]:
    segments: list[dict] = []
    if not rows:
        return segments
    start = 0
    value = rows[0][key]
    for idx in range(1, len(rows) + 1):
        if idx == len(rows) or rows[idx][key] != value:
            segment_rows = rows[start:idx]
            segments.append(
                {
                    "value": int(value),
                    "start_step": int(segment_rows[0]["step"]),
                    "end_step": int(segment_rows[-1]["step"]),
                    "start_window": int(segment_rows[0]["window_index"]),
                    "end_window": int(segment_rows[-1]["window_index"]),
                    "length": len(segment_rows),
                }
            )
            if idx < len(rows):
                start = idx
                value = rows[idx][key]
    return segments


def positive_segments(rows: list[dict]) -> list[dict]:
    return [segment for segment in contiguous_segments(rows, "label") if segment["value"] == 1]


def relation_to_segments(window: int, segments: list[dict]) -> dict:
    best: dict | None = None
    for segment in segments:
        if segment["start_window"] <= window <= segment["end_window"]:
            return {
                "relation": "inside",
                "distance": 0,
                "segment_start": segment["start_window"],
                "segment_end": segment["end_window"],
                "offset_from_start": window - segment["start_window"],
                "offset_to_end": segment["end_window"] - window,
            }
        if window < segment["start_window"]:
            distance = segment["start_window"] - window
            relation = "before"
        else:
            distance = window - segment["end_window"]
            relation = "after"
        if best is None or distance < best["distance"]:
            best = {
                "relation": relation,
                "distance": int(distance),
                "segment_start": segment["start_window"],
                "segment_end": segment["end_window"],
                "offset_from_start": None,
                "offset_to_end": None,
            }
    return best or {
        "relation": "no_positive_segment",
        "distance": None,
        "segment_start": None,
        "segment_end": None,
        "offset_from_start": None,
        "offset_to_end": None,
    }


def read_events(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        source_id = str(event["source_id"])
        rows.append(
            {
                "step": int(event["step"]),
                "source_id": source_id,
                "case_id": case_id(source_id),
                "window_index": window_index(source_id, int(event["step"])),
                "label": int(bool(event["window_anomaly_label"])),
                "prediction": int(bool(event["window_anomaly_prediction"])),
                "raw_prediction": int(bool(event.get("raw_window_anomaly_prediction", event["window_anomaly_prediction"]))),
                "window_score": float(event["window_score"]),
                "root_cause_score": float(event.get("root_cause_score", 0.0)),
                "root_cause_service": event.get("root_cause_service"),
            }
        )
    return rows


def summarize(rows: list[dict], summary_json: Path | None) -> dict:
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    case_summaries = []
    all_positive_segments: dict[str, list[dict]] = {}
    for cid, case_rows in by_case.items():
        segments = positive_segments(case_rows)
        all_positive_segments[cid] = segments
        labels = [row["label"] for row in case_rows]
        preds = [row["prediction"] for row in case_rows]
        case_summaries.append(
            {
                "case_id": cid,
                "num_windows": len(case_rows),
                "metrics": binary_metrics(preds, labels),
                "positive_segments": segments,
            }
        )

    errors = []
    for row in rows:
        if row["label"] == row["prediction"]:
            continue
        relation = relation_to_segments(row["window_index"], all_positive_segments[row["case_id"]])
        errors.append(
            {
                "step": row["step"],
                "window_index": row["window_index"],
                "case_id": row["case_id"],
                "type": "FP" if row["prediction"] == 1 else "FN",
                "label": row["label"],
                "prediction": row["prediction"],
                "raw_prediction": row["raw_prediction"],
                "window_score": row["window_score"],
                "root_cause_score": row["root_cause_score"],
                "root_cause_service": row["root_cause_service"],
                **relation,
            }
        )

    fp_errors = [error for error in errors if error["type"] == "FP"]
    fn_errors = [error for error in errors if error["type"] == "FN"]
    boundary_fp_counts = {
        f"within_{radius}": sum(
            1
            for error in fp_errors
            if error["distance"] is not None and error["distance"] <= radius
        )
        for radius in (1, 2, 3)
    }
    fn_categories = {
        "segment_onset_offset_0_1": sum(
            1
            for error in fn_errors
            if error["relation"] == "inside" and error["offset_from_start"] is not None and error["offset_from_start"] <= 1
        ),
        "segment_onset_offset_0_2": sum(
            1
            for error in fn_errors
            if error["relation"] == "inside" and error["offset_from_start"] is not None and error["offset_from_start"] <= 2
        ),
        "segment_interior": sum(
            1
            for error in fn_errors
            if error["relation"] == "inside" and error["offset_from_start"] is not None and error["offset_from_start"] > 2
        ),
    }

    labels = [row["label"] for row in rows]
    preds = [row["prediction"] for row in rows]
    metrics = binary_metrics(preds, labels)

    boundary_adjusted = {}
    for radius in (1, 2):
        adjusted_preds = preds.copy()
        for error in fp_errors:
            if error["distance"] is not None and error["distance"] <= radius:
                adjusted_preds[error["step"]] = 0
        boundary_adjusted[f"remove_fp_within_{radius}_window_of_positive_segment"] = binary_metrics(
            adjusted_preds,
            labels,
        )

    summary_payload = None
    if summary_json is not None and summary_json.exists():
        summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))

    return {
        "summary_json": None if summary_json is None else str(summary_json),
        "summary_metrics": None if summary_payload is None else summary_payload.get("replay_detection_metrics"),
        "computed_metrics": metrics,
        "case_summaries": case_summaries,
        "num_errors": len(errors),
        "num_fp": len(fp_errors),
        "num_fn": len(fn_errors),
        "boundary_fp_counts": boundary_fp_counts,
        "fn_categories": fn_categories,
        "boundary_adjusted_diagnostic_only": boundary_adjusted,
        "fn_scores_desc": sorted(
            [
                {
                    "step": error["step"],
                    "window_index": error["window_index"],
                    "window_score": error["window_score"],
                    "offset_from_start": error["offset_from_start"],
                    "offset_to_end": error["offset_to_end"],
                    "root_cause_service": error["root_cause_service"],
                }
                for error in fn_errors
            ],
            key=lambda item: item["window_score"],
            reverse=True,
        ),
        "fp_scores_desc": sorted(
            [
                {
                    "step": error["step"],
                    "window_index": error["window_index"],
                    "window_score": error["window_score"],
                    "relation": error["relation"],
                    "distance": error["distance"],
                    "root_cause_service": error["root_cause_service"],
                }
                for error in fp_errors
            ],
            key=lambda item: item["window_score"],
            reverse=True,
        ),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    events_path = Path(args.events_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else None
    rows = read_events(events_path)
    report = summarize(rows, summary_path)
    output_path = Path(args.output_json) if args.output_json else events_path.with_name(events_path.stem + "_error_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "output_json": str(output_path),
            "metrics": report["computed_metrics"],
            "num_errors": report["num_errors"],
            "boundary_fp_counts": report["boundary_fp_counts"],
            "fn_categories": report["fn_categories"],
            "boundary_adjusted_diagnostic_only": report["boundary_adjusted_diagnostic_only"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
