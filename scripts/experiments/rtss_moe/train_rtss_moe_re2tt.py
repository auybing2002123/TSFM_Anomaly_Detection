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

from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import create_re2tt_lazy_dataloaders  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_config import RTSSMoERCAEvalConfig  # noqa: E402
from scripts.experiments.rtss_moe.rtss_moe_model import RTSSMultiModalV6MoEAdapter_RCAEval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSS-MoE P0 training (RE2-TT)")
    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints/rcaeval/experiments/rtss_moe/p0_re2tt")
    parser.add_argument("--abnormal-weight", type=float, default=6.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-rank", type=int, default=4)
    parser.add_argument("--moe-alpha", type=float, default=8.0)
    parser.add_argument("--moe-dropout", type=float, default=0.05)
    parser.add_argument("--moe-adapter-layers", type=int, default=1)
    parser.add_argument("--moe-router-hidden", type=int, default=64)
    parser.add_argument("--rtss-budget-topk", type=int, default=2)
    parser.add_argument("--moe-router-temperature", type=float, default=0.7)
    parser.add_argument("--moe-balance-weight", type=float, default=0.05)
    parser.add_argument("--rtss-latency-cost-weight", type=float, default=0.0)
    parser.add_argument("--rtss-route-switch-weight", type=float, default=0.0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)
            cls_probs, _ = model(metrics, logs, traces, gt_cls, evaluate=True)
            all_preds.append(cls_probs.argmax(dim=-1).cpu())
            all_labels.append(gt_real.argmax(dim=-1).cpu())

    preds = torch.cat(all_preds, dim=0).flatten()
    labels = torch.cat(all_labels, dim=0).flatten()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def main() -> None:
    args = parse_args()
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

    config = RTSSMoERCAEvalConfig(
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
        moe_adapter_layers=args.moe_adapter_layers,
        moe_router_hidden=args.moe_router_hidden,
        moe_router_temperature=args.moe_router_temperature,
        moe_balance_weight=args.moe_balance_weight,
        rtss_budget_topk=args.rtss_budget_topk,
        rtss_latency_cost_weight=args.rtss_latency_cost_weight,
        rtss_route_switch_weight=args.rtss_route_switch_weight,
    )
    model = RTSSMultiModalV6MoEAdapter_RCAEval(config, adjacency_matrix).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    patience_counter = 0
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"RTSS-MoE RE2-TT {epoch}/{args.epochs}")
        for batch in pbar:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)

            optimizer.zero_grad()
            loss, rec_loss, cls_loss, pred_loss = model(metrics, logs, traces, gt_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bal": f"{model.latest_rtss_aux.balance_loss:.4f}",
                    "cost": f"{model.latest_rtss_aux.cost_loss:.4f}",
                    "switch": f"{model.latest_rtss_aux.switch_loss:.4f}",
                }
            )

        scheduler.step()
        val_metrics = evaluate(model, loaders["val"], device)
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

    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)
    print(
        f"Test: F1={test_metrics['f1']:.4f}, "
        f"P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
