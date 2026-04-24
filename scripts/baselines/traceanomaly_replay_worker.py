import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resident-model paced replay worker for TraceAnomaly strict baseline."
    )
    parser.add_argument("--traceanomaly-root", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=["aggregate", "flatten_time"], required=True)
    parser.add_argument("--flow-type", choices=["rnvp", "planar_nf", "none"], required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-summary-path", type=Path, required=True)
    parser.add_argument("--output-events-path", type=Path, required=True)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def summarize_series(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": percentile(arr, 50),
        "p95_ms": percentile(arr, 95),
        "p99_ms": percentile(arr, 99),
        "max_ms": float(arr.max()),
    }


def summarize_latency_records(records):
    metrics = [
        "sample_load_ms",
        "tensorize_ms",
        "transfer_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
    ]
    return {
        name: summarize_series([float(record[name]) for record in records]) for name in metrics
    }


def summarize_response_times(values):
    return {"response_time_ms": summarize_series(values)}


def compute_binary_metrics(preds, labels):
    preds_arr = np.asarray(preds, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
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


def score_to_prediction(score, direction, threshold):
    if direction == "low_is_abnormal":
        return bool(score <= threshold)
    return bool(score >= threshold)


def _prepare_imports(traceanomaly_root):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(traceanomaly_root) not in sys.path:
        sys.path.insert(0, str(traceanomaly_root))


def _load_npz(path):
    with np.load(str(path), allow_pickle=True) as payload:
        result = {}
        for name in payload.files:
            result[name] = payload[name]
        return result


def _build_vector_from_trace(trace_tensor, representation):
    if representation == "flatten_time":
        return trace_tensor.reshape(-1).astype(np.float32)

    counts = trace_tensor[..., 0].astype(np.float32)
    mean_durations = trace_tensor[..., 1].astype(np.float32)
    agg_counts = counts.sum(axis=0)
    active_mask = counts > 0
    active_steps = active_mask.sum(axis=0).astype(np.float32)
    duration_sum = np.where(active_mask, mean_durations, 0.0).sum(axis=0)
    agg_mean_duration = np.divide(
        duration_sum,
        np.maximum(active_steps, 1.0),
        out=np.zeros_like(duration_sum, dtype=np.float32),
        where=active_steps > 0,
    )
    return np.concatenate([agg_counts.reshape(-1), agg_mean_duration.reshape(-1)], axis=0).astype(np.float32)


def _vectorize_split(payload, representation, split_name):
    trace_tensor = payload["traces"]
    labels = payload["window_labels"].astype(np.int64)
    case_names = payload["case_names"]
    vectors = []
    ids = []
    for idx in range(trace_tensor.shape[0]):
        vectors.append(_build_vector_from_trace(trace_tensor[idx], representation))
        case_name = str(case_names[idx])
        ids.append("{}_{:05d}_{}".format(split_name, idx, case_name.replace("/", "_")))
    return {
        "ids": ids,
        "labels": labels,
        "vectors": np.stack(vectors, axis=0),
    }


def _load_strict_arrays(export_dir, representation):
    train_split = _vectorize_split(_load_npz(export_dir / "train.npz"), representation, "train")
    test_split = _vectorize_split(_load_npz(export_dir / "test.npz"), representation, "test")
    return (
        train_split["vectors"],
        test_split["vectors"],
        list(test_split["ids"]),
        test_split["labels"].astype(np.int64),
    )


def _compute_valid_columns(train_raw):
    if train_raw.size == 0:
        return []
    positive_mask = train_raw > 0
    return np.where(np.any(positive_mask, axis=0))[0].astype(np.int64).tolist()


def _get_mean_std(matrix):
    means = []
    stds = []
    for column in matrix.T:
        positive = column[column > 0.00001]
        if positive.size == 0:
            means.append(0.0)
            stds.append(1.0)
            continue
        means.append(float(np.mean(positive)))
        stds.append(float(max(1.0, np.std(positive))))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def _normalize(matrix, mean, std):
    arr = np.asarray(matrix, dtype=np.float32)
    normalized = np.where(arr < 0.00001, -1.0, (arr - mean) / std)
    return normalized.astype(np.float32)


def _fit_and_transform(train_raw, test_raw):
    valid_columns = _compute_valid_columns(train_raw)
    if not valid_columns:
        raise ValueError("TraceAnomaly replay found no valid columns from train split.")
    train_filtered = train_raw[:, valid_columns].astype(np.float32)
    test_filtered = test_raw[:, valid_columns].astype(np.float32)
    mean, std = _get_mean_std(train_filtered)
    return _normalize(train_filtered, mean, std), _normalize(test_filtered, mean, std), len(valid_columns)


def _build_graph(model_dir, flow_type, input_dim):
    import tfsnippet as spt
    from traceanomaly.MLConfig import set_global_config
    import traceanomaly.main as tamain

    tf.reset_default_graph()
    cfg = tamain.ExpConfig()
    set_global_config(cfg)
    cfg.debug_level = -1
    cfg.flow_type = None if flow_type == "none" else flow_type
    cfg.x_dim = int(input_dim)

    spt.utils.set_assertion_enabled(False)

    input_x = tf.placeholder(dtype=tf.float32, shape=(None, cfg.x_dim), name="input_x")
    if cfg.flow_type is None:
        posterior_flow = None
    elif cfg.flow_type == "planar_nf":
        posterior_flow = spt.layers.planar_normalizing_flows(cfg.n_planar_nf_layers)
    else:
        if cfg.flow_type != "rnvp":
            raise ValueError("Unsupported flow_type: {}".format(cfg.flow_type))
        with tf.variable_scope("posterior_flow"):
            flows = []
            for _ in range(cfg.n_rnvp_layers):
                flows.append(spt.layers.ActNorm())
                flows.append(
                    spt.layers.CouplingLayer(
                        tf.make_template(
                            "coupling",
                            tamain.coupling_layer_shift_and_scale,
                            create_scope_now_=True,
                        ),
                        scale_type="sigmoid",
                    )
                )
                flows.append(spt.layers.InvertibleDense(strict_invertible=True))
            posterior_flow = spt.layers.SequentialFlow(flows=flows)

    with tf.name_scope("testing"):
        test_q_net = tamain.q_net(input_x, posterior_flow, n_z=cfg.test_n_z)
        test_chain = test_q_net.chain(tamain.p_net, latent_axis=0, observed={"x": input_x})
        test_logp = test_chain.vi.evaluation.is_loglikelihood()

    session = spt.utils.create_session(lock_memory=False)
    with session.as_default():
        var_dict = spt.utils.get_variables_as_dict()
        saver = spt.VariableSaver(var_dict, str(model_dir))
        saver.restore()
    return session, input_x, test_logp, cfg


def _prefetch_samples(ids, vectors, labels, start_index, count):
    prefetched = {}
    upper = min(len(vectors), start_index + count)
    for sample_idx in range(start_index, upper):
        prefetched[sample_idx] = (
            ids[sample_idx],
            np.asarray(vectors[sample_idx], dtype=np.float32).copy(),
            int(labels[sample_idx]),
        )
    return prefetched


def _print_step(event):
    state = "OK" if event["deadline_met"] else "MISS"
    print(
        "[step={:04d}] sample={:05d} score={:.6f} label={} pred={} proc={:.2f}ms resp={:.2f}ms deadline={}".format(
            event["step"],
            event["sample_idx"],
            event["anomaly_score"],
            int(event["label"]),
            int(event["prediction"]),
            event["processing_ms"],
            event["response_time_ms"],
            state,
        ),
        flush=True,
    )


def main():
    args = parse_args()
    _prepare_imports(args.traceanomaly_root)

    offline_summary = load_json(args.summary_json)
    train_raw, test_raw, test_ids, test_labels = _load_strict_arrays(args.export_dir, args.representation)
    _, test_vectors, effective_dim = _fit_and_transform(train_raw, test_raw)

    available = max(0, len(test_vectors) - args.start_index)
    max_steps = min(args.max_steps, available)
    if max_steps <= 0:
        raise ValueError("No samples available for TraceAnomaly strict replay.")

    session, input_x, test_logp, cfg = _build_graph(
        model_dir=args.model_dir,
        flow_type=args.flow_type,
        input_dim=effective_dim,
    )

    prefetched = None
    prefetch_wall_ms = 0.0
    if args.prefetch:
        prefetch_start = time.perf_counter()
        prefetched = _prefetch_samples(
            ids=test_ids,
            vectors=test_vectors,
            labels=test_labels,
            start_index=args.start_index,
            count=max_steps,
        )
        prefetch_wall_ms = (time.perf_counter() - prefetch_start) * 1000.0

    direction = str(offline_summary["validation_selection"]["direction"])
    threshold = float(offline_summary["validation_selection"]["threshold"])

    print("TraceAnomaly strict replay worker loaded {} test windows".format(len(test_vectors)), flush=True)
    print(
        "Representation={}, flow_type={}, effective_dim={}, threshold={:.6f} ({})".format(
            args.representation,
            args.flow_type,
            effective_dim,
            threshold,
            direction,
        ),
        flush=True,
    )

    with session.as_default():
        for offset in range(args.warmup_samples):
            sample_idx = args.start_index + offset
            if sample_idx >= len(test_vectors):
                break
            sample = test_vectors[sample_idx : sample_idx + 1]
            _ = session.run(test_logp, feed_dict={input_x: sample})

        events = []
        interval_ms = float(args.interval_ms)
        deadline_ms = float(args.deadline_ms)
        interval_s = interval_ms / 1000.0
        base_release = time.perf_counter()

        for step in range(max_steps):
            sample_idx = args.start_index + step
            scheduled_release = base_release + step * interval_s if args.pace else time.perf_counter()
            if args.pace:
                sleep_time = scheduled_release - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_start = time.perf_counter()
            sample_load_start = time.perf_counter()
            sample = None if prefetched is None else prefetched.get(sample_idx)
            if sample is None:
                sample = (
                    test_ids[sample_idx],
                    np.asarray(test_vectors[sample_idx], dtype=np.float32),
                    int(test_labels[sample_idx]),
                )
            sample_load_ms = (time.perf_counter() - sample_load_start) * 1000.0

            tensorize_start = time.perf_counter()
            source_id, sample_vector, label_int = sample
            batch_x = np.expand_dims(sample_vector.astype(np.float32), axis=0)
            tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

            inference_start = time.perf_counter()
            logp = session.run(test_logp, feed_dict={input_x: batch_x})
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            postprocess_start = time.perf_counter()
            score = float(np.asarray(logp).reshape(-1)[0] / cfg.x_dim)
            prediction = score_to_prediction(score, direction, threshold)
            postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
            finish_time = time.perf_counter()

            processing_ms = (finish_time - total_start) * 1000.0
            response_time_ms = (finish_time - scheduled_release) * 1000.0
            event = {
                "step": step,
                "sample_idx": sample_idx,
                "source_id": source_id,
                "label": bool(label_int > 0),
                "prediction": bool(prediction),
                "threshold": threshold,
                "direction": direction,
                "anomaly_score": score,
                "scheduled_release_ms": scheduled_release * 1000.0,
                "finish_ms": finish_time * 1000.0,
                "release_lag_ms": max(0.0, (total_start - scheduled_release) * 1000.0),
                "sample_load_ms": sample_load_ms,
                "tensorize_ms": tensorize_ms,
                "transfer_ms": 0.0,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": processing_ms,
                "processing_ms": processing_ms,
                "response_time_ms": response_time_ms,
                "deadline_ms": deadline_ms,
                "deadline_met": response_time_ms <= deadline_ms,
            }
            events.append(event)

            if (
                args.print_every <= 1
                or step == 0
                or (step + 1) % args.print_every == 0
                or step == max_steps - 1
            ):
                _print_step(event)

    deadline_misses = sum(1 for event in events if not event["deadline_met"])
    replay_detection_metrics = compute_binary_metrics(
        preds=[int(bool(event["prediction"])) for event in events],
        labels=[int(bool(event["label"])) for event in events],
    )
    summary = {
        "run": {
            "baseline": "TraceAnomaly",
            "dataset": "Eadro-SN",
            "protocol": "strict case-level split + train-normal-only + val-threshold + test-final",
            "summary_json": str(args.summary_json),
            "traceanomaly_root": str(args.traceanomaly_root),
            "model_dir": str(args.model_dir),
            "device": "cpu",
            "device_name": "docker-tf1-cpu",
            "step_size_ms_assumption": 100.0,
            "replay_note": "Official TraceAnomaly graph restored once inside Docker TF1 worker; routing diagnostics are N/A.",
        },
        "mode": "paced_replay" if args.pace else "as_fast_as_possible",
        "prefetch_enabled": bool(args.prefetch),
        "pin_memory_enabled": False,
        "pin_memory_requested": bool(args.pin_memory),
        "prefetch_wall_ms": prefetch_wall_ms,
        "num_steps": max_steps,
        "start_index": args.start_index,
        "interval_ms": float(args.interval_ms),
        "deadline_ms": float(args.deadline_ms),
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate_pct": deadline_misses / max_steps * 100.0,
        "offline_threshold": {
            "direction": direction,
            "threshold": threshold,
        },
        "offline_test_metrics": offline_summary["test_result"]["metrics"],
        "replay_detection_metrics": replay_detection_metrics,
        "processing_latency": summarize_latency_records(events),
        "response_latency": summarize_response_times([event["response_time_ms"] for event in events]),
        "baseline_metadata": {
            "representation": args.representation,
            "flow_type": args.flow_type,
            "effective_dim": effective_dim,
            "full_vector_dim": int(test_raw.shape[1]),
            "model_dir": str(args.model_dir),
            "runtime_note": "pin_memory is not applicable in the TensorFlow/NumPy CPU worker.",
        },
        "coverage": {
            "replayed_steps": max_steps,
            "total_test_steps": int(len(test_vectors)),
            "coverage_pct": max_steps / max(len(test_vectors), 1) * 100.0,
        },
    }

    save_json(args.output_summary_path, summary)
    save_jsonl(args.output_events_path, events)

    print("\n" + "=" * 72, flush=True)
    print("TraceAnomaly Strict Replay Summary", flush=True)
    print("=" * 72, flush=True)
    print("Mode         : {}".format(summary["mode"]), flush=True)
    print(
        "Prefetch     : {} (pin_memory_effective={})".format(
            summary["prefetch_enabled"],
            summary["pin_memory_enabled"],
        ),
        flush=True,
    )
    if summary["prefetch_enabled"]:
        print("Prefetch Cost: {:.2f} ms".format(summary["prefetch_wall_ms"]), flush=True)
    print("Steps        : {}".format(summary["num_steps"]), flush=True)
    print("Miss Rate    : {:.2f}%".format(summary["deadline_miss_rate_pct"]), flush=True)
    print(
        "Response     : mean={:.2f} ms  p95={:.2f} ms  p99={:.2f} ms  max={:.2f} ms".format(
            summary["response_latency"]["response_time_ms"]["mean_ms"],
            summary["response_latency"]["response_time_ms"]["p95_ms"],
            summary["response_latency"]["response_time_ms"]["p99_ms"],
            summary["response_latency"]["response_time_ms"]["max_ms"],
        ),
        flush=True,
    )
    print(
        "Processing   : mean={:.2f} ms  p95={:.2f} ms  p99={:.2f} ms  max={:.2f} ms".format(
            summary["processing_latency"]["total_ms"]["mean_ms"],
            summary["processing_latency"]["total_ms"]["p95_ms"],
            summary["processing_latency"]["total_ms"]["p99_ms"],
            summary["processing_latency"]["total_ms"]["max_ms"],
        ),
        flush=True,
    )
    print(
        "Replay Detect: F1={:.4f}  P={:.4f}  R={:.4f}  Acc={:.4f}".format(
            summary["replay_detection_metrics"]["f1"],
            summary["replay_detection_metrics"]["precision"],
            summary["replay_detection_metrics"]["recall"],
            summary["replay_detection_metrics"]["accuracy"],
        ),
        flush=True,
    )
    print("=" * 72, flush=True)
    print("Summary saved to: {}".format(args.output_summary_path), flush=True)
    print("Event logs saved to: {}".format(args.output_events_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
