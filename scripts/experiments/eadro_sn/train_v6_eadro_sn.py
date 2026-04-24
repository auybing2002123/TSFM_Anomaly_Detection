from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_rcaeval.v6.config import V6RCAEvalConfig  # noqa: E402
from models_rcaeval.v6.model import MultiModalV6_RCAEval  # noqa: E402
from scripts.experiments.eadro_sn.eadro_sn_lazy_loader import (  # noqa: E402
    create_eadro_sn_lazy_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated V6 3-layer raw training on Eadro-SN")
    parser.add_argument("--data-dir", type=str, default="data_eadro/processed/sn_lazy")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="root",
        choices=["root", "anomaly"],
        help="root: 只监督 root service; anomaly: 将 root+affected 都视为异常服务",
    )
    parser.add_argument(
        "--selection-target",
        type=str,
        default="auto",
        choices=["auto", "root_service", "service_anomaly", "window_anomaly"],
        help="验证集上用哪个指标选 best checkpoint",
    )

    parser.add_argument("--gpt2-layers", type=int, default=3)
    parser.add_argument("--freeze-gpt2", action="store_true", default=True)
    parser.add_argument("--no-freeze-gpt2", dest="freeze_gpt2", action="store_false")
    parser.add_argument("--train-ln", action="store_true", default=True)
    parser.add_argument("--no-train-ln", dest="train_ln", action="store_false")
    parser.add_argument("--train-wpe", action="store_true", default=True)
    parser.add_argument("--no-train-wpe", dest="train_wpe", action="store_false")

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument(
        "--trace-no-graph",
        action="store_true",
        help="Disable trace graph propagation (equivalent to --num-gat-layers 0)",
    )
    parser.add_argument(
        "--adjacency-mode",
        type=str,
        default="two_hop",
        choices=["two_hop", "raw", "dense"],
        help="two_hop: raw+2hop closure, raw: direct topology, dense: fully connected (without self-loop)",
    )
    parser.add_argument("--cls-hidden-dim", type=int, default=256)

    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", type=str, default="qv", choices=["qv", "qkv", "all"])

    parser.add_argument("--abnormal-weight", type=float, default=4.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--disable-traces", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/eadro/experiments/v6_3layer_raw_eadro_sn_s42",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/experiments/eadro_sn/v6_3layer_raw_eadro_sn_s42",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def maybe_disable_modalities(
    metrics: torch.Tensor,
    logs: torch.Tensor,
    traces: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.disable_metrics:
        metrics = torch.zeros_like(metrics)
    if args.disable_logs:
        logs = torch.zeros_like(logs)
    if args.disable_traces:
        traces = torch.zeros_like(traces)
    return metrics, logs, traces


def convert_training_labels(gt_cls: torch.Tensor, label_mode: str) -> torch.Tensor:
    if label_mode == "root":
        return gt_cls
    if label_mode != "anomaly":
        raise ValueError(f"未知 label_mode: {label_mode}")

    abnormal = ((gt_cls[..., 1] + gt_cls[..., 2]) > 0).float().unsqueeze(-1)
    normal = 1.0 - abnormal
    unknown = torch.zeros_like(normal)
    return torch.cat([normal, abnormal, unknown], dim=-1)


def resolve_selection_target(label_mode: str, selection_target: str) -> str:
    if selection_target != "auto":
        return selection_target
    if label_mode == "anomaly":
        return "window_anomaly"
    return "root_service"


def compute_binary_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds = (probs >= threshold).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    limit_batches: int = 0,
) -> dict[str, dict[str, float] | str]:
    model.eval()
    all_root_probs = []
    all_root_labels = []
    all_service_anomaly_labels = []
    all_window_probs = []
    all_window_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches > 0 and batch_idx >= limit_batches:
                break

            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)
            metrics, logs, traces = maybe_disable_modalities(metrics, logs, traces, args)
            gt_cls = convert_training_labels(gt_cls_raw, args.label_mode)

            cls_probs, _ = model(metrics, logs, traces, gt_cls, evaluate=True)
            positive_probs = cls_probs[..., 1].detach().cpu()
            root_labels = gt_real[..., 1].detach().cpu()
            service_anomaly_labels = ((gt_cls_raw[..., 1] + gt_cls_raw[..., 2]) > 0).long().detach().cpu()
            window_probs = positive_probs.max(dim=1).values
            window_labels = service_anomaly_labels.max(dim=1).values

            all_root_probs.append(positive_probs.flatten())
            all_root_labels.append(root_labels.flatten())
            all_service_anomaly_labels.append(service_anomaly_labels.flatten())
            all_window_probs.append(window_probs)
            all_window_labels.append(window_labels)

    root_probs = torch.cat(all_root_probs, dim=0)
    root_labels = torch.cat(all_root_labels, dim=0)
    service_anomaly_labels = torch.cat(all_service_anomaly_labels, dim=0)
    window_probs = torch.cat(all_window_probs, dim=0)
    window_labels = torch.cat(all_window_labels, dim=0)

    root_service_metrics = compute_binary_metrics(root_probs, root_labels)
    service_anomaly_metrics = compute_binary_metrics(root_probs, service_anomaly_labels)
    window_anomaly_metrics = compute_binary_metrics(window_probs, window_labels)
    selection_target = resolve_selection_target(args.label_mode, args.selection_target)
    selected_metrics = {
        "root_service": root_service_metrics,
        "service_anomaly": service_anomaly_metrics,
        "window_anomaly": window_anomaly_metrics,
    }[selection_target]

    return {
        "selection_target": selection_target,
        "selected_metrics": selected_metrics,
        "root_service_metrics": root_service_metrics,
        "service_anomaly_metrics": service_anomaly_metrics,
        "window_anomaly_metrics": window_anomaly_metrics,
    }


def build_adjacency(metadata: dict, mode: str = "two_hop") -> torch.Tensor:
    num_services = int(metadata["num_services"])

    if mode == "dense":
        adjacency_matrix = torch.ones(num_services, num_services, dtype=torch.float32)
        adjacency_matrix.fill_diagonal_(0)
        return adjacency_matrix

    if mode not in {"two_hop", "raw"}:
        raise ValueError(f"Unsupported adjacency mode: {mode}")

    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        if mode == "raw":
            return adj_raw
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
        return adjacency_matrix

    adjacency_matrix = torch.ones(num_services, num_services, dtype=torch.float32)
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def summary_name(args: argparse.Namespace) -> str:
    parts = [
        f"v6_{args.gpt2_layers}layer",
        f"seed{args.seed}",
        f"bs{args.batch_size}",
        f"lr{args.lr:g}".replace(".", "p"),
    ]
    if args.disable_metrics:
        parts.append("wo_metrics")
    if args.disable_logs:
        parts.append("wo_logs")
    if args.disable_traces:
        parts.append("wo_traces")
    if args.num_gat_layers == 0:
        parts.append("trace_no_graph")
    if args.adjacency_mode != "two_hop":
        parts.append(f"adj_{args.adjacency_mode}")
    if args.label_mode != "root":
        parts.append(f"{args.label_mode}_label")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须 >= 1")
    if args.trace_no_graph:
        args.num_gat_layers = 0

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    result_dir = (PROJECT_ROOT / args.result_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading Eadro-SN data from: {data_dir}")
    loaders = create_eadro_sn_lazy_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=False,
    )
    metadata = loaders["metadata"]
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    adjacency_matrix = build_adjacency(metadata, mode=args.adjacency_mode)
    num_edges = int((adjacency_matrix > 0).sum().item())
    print("\n[Data Summary]")
    print(f"  services: {metadata['num_services']}")
    print(f"  metric_dim: {metadata['num_metrics']}")
    print(f"  log_dim: {metadata['log_dim']}")
    print(f"  trace_dim: {metadata['trace_dim']}")
    print(f"  adjacency mode: {args.adjacency_mode}")
    print(f"  trace graph layers: {args.num_gat_layers}")
    print(f"  adjacency edges (2-hop): {num_edges}")
    print(f"  train/val/test: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")

    config = V6RCAEvalConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["num_metrics"],
        log_dim=metadata["log_dim"],
        trace_dim=metadata["trace_dim"],
        embed_dim=args.embed_dim,
        gpt2_layers=args.gpt2_layers,
        freeze_gpt2=args.freeze_gpt2,
        train_ln=args.train_ln,
        train_wpe=args.train_wpe,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        cls_hidden_dim=args.cls_hidden_dim,
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight,
        pred_loss_weight=args.pred_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
    )

    model = MultiModalV6_RCAEval(config, adjacency_matrix).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, dict[str, float] | str] | None = None
    patience_counter = 0
    run_name = summary_name(args)
    checkpoint_path = save_dir / "best_model.pth"
    summary_path = result_dir / f"{run_name}_summary.json"

    start_time = time.perf_counter()
    print(f"\nStarting training: epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_cls = 0.0
        epoch_pred = 0.0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Eadro-SN V6 {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break

            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls_raw = batch["groundtruth_cls"].float().to(device)
            metrics, logs, traces = maybe_disable_modalities(metrics, logs, traces, args)
            gt_cls = convert_training_labels(gt_cls_raw, args.label_mode)

            loss, rec_loss, cls_loss, pred_loss = model(metrics, logs, traces, gt_cls)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            limited_last = args.limit_train_batches > 0 and (batch_idx + 1) == args.limit_train_batches
            epoch_last = (batch_idx + 1) == len(train_loader)
            if should_step or limited_last or epoch_last:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item())
            epoch_rec += float(rec_loss.item())
            epoch_cls += float(cls_loss.item())
            epoch_pred += float(pred_loss.item())
            num_steps += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "rec": f"{rec_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "pred": f"{pred_loss.item():.4f}",
                }
            )

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_rec = epoch_rec / max(num_steps, 1)
        avg_cls = epoch_cls / max(num_steps, 1)
        avg_pred = epoch_pred / max(num_steps, 1)
        val_selected = val_metrics["selected_metrics"]
        val_root = val_metrics["root_service_metrics"]
        val_service_anomaly = val_metrics["service_anomaly_metrics"]
        val_window = val_metrics["window_anomaly_metrics"]
        print(
            f"Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, Rec={avg_rec:.4f}, Cls={avg_cls:.4f}, Pred={avg_pred:.4f}, "
            f"Val[{val_metrics['selection_target']}] F1={val_selected['f1']:.4f}, "
            f"P={val_selected['precision']:.4f}, R={val_selected['recall']:.4f}, "
            f"RootF1={val_root['f1']:.4f}, ServiceAnomF1={val_service_anomaly['f1']:.4f}, "
            f"WindowF1={val_window['f1']:.4f}"
        )

        if val_selected["f1"] > best_f1:
            best_f1 = val_selected["f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "f1": best_f1,
                        "selection_target": val_metrics["selection_target"],
                        "metadata": metadata,
                        "args": vars(args),
                    },
                checkpoint_path,
            )
            print(f"  [BEST] saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError(f"训练未产生 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_metrics = evaluate(model, train_loader, device, args, limit_batches=args.limit_eval_batches)
    val_metrics = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches)
    test_metrics = evaluate(model, test_loader, device, args, limit_batches=args.limit_eval_batches)
    train_selected = train_metrics["selected_metrics"]
    val_selected = val_metrics["selected_metrics"]
    test_selected = test_metrics["selected_metrics"]

    elapsed_sec = time.perf_counter() - start_time
    summary = {
        "run_name": run_name,
        "elapsed_sec": elapsed_sec,
        "checkpoint_path": str(checkpoint_path),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "result_dir": str(result_dir),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "selection_target": resolve_selection_target(args.label_mode, args.selection_target),
        "train_metrics": train_selected,
        "val_metrics": val_selected,
        "test_metrics": test_selected,
        "train_metric_suite": train_metrics,
        "val_metric_suite": val_metrics,
        "test_metric_suite": test_metrics,
        "metadata": {
            "dataset_name": metadata["dataset_name"],
            "num_services": metadata["num_services"],
            "num_metrics": metadata["num_metrics"],
            "log_dim": metadata["log_dim"],
            "trace_dim": metadata["trace_dim"],
            "num_samples": metadata["num_samples"],
            "num_cases": metadata["num_cases"],
        },
        "config": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Final Metrics")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Selection target: {resolve_selection_target(args.label_mode, args.selection_target)}")
    print(f"Best val F1: {best_f1:.4f}")
    print(
        f"Train selected: F1={train_selected['f1']:.4f}, P={train_selected['precision']:.4f}, "
        f"R={train_selected['recall']:.4f}, Acc={train_selected['accuracy']:.4f}"
    )
    print(
        f"Val selected:   F1={val_selected['f1']:.4f}, P={val_selected['precision']:.4f}, "
        f"R={val_selected['recall']:.4f}, Acc={val_selected['accuracy']:.4f}"
    )
    print(
        f"Test selected:  F1={test_selected['f1']:.4f}, P={test_selected['precision']:.4f}, "
        f"R={test_selected['recall']:.4f}, Acc={test_selected['accuracy']:.4f}"
    )
    print(
        f"Test root-service: F1={test_metrics['root_service_metrics']['f1']:.4f}, "
        f"P={test_metrics['root_service_metrics']['precision']:.4f}, "
        f"R={test_metrics['root_service_metrics']['recall']:.4f}"
    )
    print(
        f"Test service-anomaly: F1={test_metrics['service_anomaly_metrics']['f1']:.4f}, "
        f"P={test_metrics['service_anomaly_metrics']['precision']:.4f}, "
        f"R={test_metrics['service_anomaly_metrics']['recall']:.4f}"
    )
    print(
        f"Test window-anomaly: F1={test_metrics['window_anomaly_metrics']['f1']:.4f}, "
        f"P={test_metrics['window_anomaly_metrics']['precision']:.4f}, "
        f"R={test_metrics['window_anomaly_metrics']['recall']:.4f}"
    )
    print(
        f"Confusion (test selected): TP={test_selected['tp']}, TN={test_selected['tn']}, "
        f"FP={test_selected['fp']}, FN={test_selected['fn']}"
    )
    print(f"Summary saved to: {summary_path}")
    print(f"Elapsed: {elapsed_sec / 60.0:.2f} min")


if __name__ == "__main__":
    main()
