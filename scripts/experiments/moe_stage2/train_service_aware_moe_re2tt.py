from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import (  # noqa: E402
    create_re2tt_lazy_dataloaders,
)
from scripts.experiments.moe_stage2.service_aware_moe_config import (  # noqa: E402
    ServiceAwareMoERCAEvalConfig,
)
from scripts.experiments.moe_stage2.service_aware_moe_model import (  # noqa: E402
    MultiModalServiceAwareMoE_RCAEval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 service-aware MoE training (RE2-TT)")
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/rcaeval/experiments/moe_stage2/service_aware_re2tt",
    )
    parser.add_argument("--abnormal-weight", type=float, default=6.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)

    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-target", choices=["qv", "qkv", "all"], default="qv")
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--moe-router-temperature", type=float, default=0.7)
    parser.add_argument("--moe-balance-weight", type=float, default=0.05)

    parser.add_argument("--service-prior-enabled", action="store_true", default=True)
    parser.add_argument("--no-service-prior", dest="service_prior_enabled", action="store_false")
    parser.add_argument("--service-prior-mode", choices=["cyclic", "block"], default="cyclic")
    parser.add_argument("--service-prior-strength", type=float, default=0.75)
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--disable-traces", action="store_true")
    parser.add_argument("--trace-encoder-mode", choices=["gat", "no_graph"], default="gat")
    parser.add_argument("--adjacency-mode", choices=["original", "dense"], default="original")
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    limit_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches > 0 and batch_idx >= limit_batches:
                break
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            cls_probs, _ = model(metrics, logs, traces, gt_cls, evaluate=True)
            all_preds.append(cls_probs.argmax(dim=-1).cpu())
            all_labels.append(gt_real.argmax(dim=-1).cpu())

    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须 >= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loaders = create_re2tt_lazy_dataloaders(
            str((PROJECT_ROOT / args.data_dir).resolve()),
            batch_size=args.batch_size,
            num_workers=0,
            seed=args.seed,
            pin_memory=args.pin_memory,
        )

    metadata = loaders["metadata"]
    if metadata.get("adjacency_matrix"):
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
    else:
        adjacency_matrix = torch.ones(metadata["num_services"], metadata["num_services"])
        adjacency_matrix.fill_diagonal_(0)

    config = ServiceAwareMoERCAEvalConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["metrics_per_service"],
        log_dim=metadata["log_dim"],
        trace_dim=metadata["trace_dim"],
        abnormal_weight=args.abnormal_weight,
        cls_weight=args.cls_weight,
        pred_loss_weight=args.pred_loss_weight,
        moe_num_experts=args.moe_num_experts,
        moe_rank=args.moe_rank,
        moe_alpha=args.moe_alpha,
        moe_dropout=args.moe_dropout,
        moe_target=args.moe_target,
        moe_adapter_layers=args.moe_adapter_layers,
        moe_router_hidden=args.moe_router_hidden,
        moe_router_topk=args.moe_router_topk,
        moe_router_temperature=args.moe_router_temperature,
        moe_balance_weight=args.moe_balance_weight,
        service_prior_enabled=args.service_prior_enabled,
        service_prior_mode=args.service_prior_mode,
        service_prior_strength=args.service_prior_strength,
        disable_metrics=args.disable_metrics,
        disable_logs=args.disable_logs,
        disable_traces=args.disable_traces,
        trace_encoder_mode=args.trace_encoder_mode,
        adjacency_mode=args.adjacency_mode,
    )

    model = MultiModalServiceAwareMoE_RCAEval(config, adjacency_matrix).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    patience_counter = 0
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Starting Stage-2 service-aware MoE training for {args.epochs} epochs...")
    print(
        "Low-memory config: "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
        f"effective_batch_size={args.batch_size * args.grad_accum_steps}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"Stage2 ServiceAware {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            if args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)

            loss, rec_loss, cls_loss, pred_loss = model(metrics, logs, traces, gt_cls)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            is_last_limited_batch = (
                args.limit_train_batches > 0 and (batch_idx + 1) == args.limit_train_batches
            )
            is_last_epoch_batch = (batch_idx + 1) == len(loaders["train"])
            if should_step or is_last_limited_batch or is_last_epoch_batch:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "moe": f"{float(model.latest_moe_balance_loss.item()):.4f}",
                }
            )

        scheduler.step()
        val_metrics = evaluate(model, loaders["val"], device, limit_batches=args.limit_eval_batches)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"Epoch {epoch}: "
            f"F1={val_metrics['f1']:.4f}, "
            f"P={val_metrics['precision']*100:.1f}%, "
            f"R={val_metrics['recall']*100:.1f}%"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "f1": best_f1,
                },
                save_dir / "best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    checkpoint_path = save_dir / "best_model.pth"
    if not checkpoint_path.exists():
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "f1": best_f1,
            },
            checkpoint_path,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, limit_batches=args.limit_eval_batches)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"Test: F1={test_metrics['f1']:.4f}, "
        f"P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
