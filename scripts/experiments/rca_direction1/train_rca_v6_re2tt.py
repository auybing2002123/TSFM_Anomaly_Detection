from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.backbone_efficiency.re2tt_lazy_loader import (  # noqa: E402
    create_re2tt_lazy_dataloaders,
)
from scripts.experiments.rca_direction1.rca_v6_config import (  # noqa: E402
    V6RootHeadRCAEvalConfig,
)
from scripts.experiments.rca_direction1.rca_metrics import (  # noqa: E402
    aggregate_case_scores,
    compute_service_rca_metrics,
    format_service_rca_metrics,
)
from scripts.experiments.rca_direction1.rca_v6_model import (  # noqa: E402
    MultiModalV6RootHead_RCAEval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCA direction-1: V6 + root head (RE2-TT)")

    parser.add_argument("--data-dir", type=str, default="data_rcaeval/processed/re2-tt_lazy")
    parser.add_argument("--save-dir", type=str, default="checkpoints/experiments/rca_direction1/v6_root_head_re2tt")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--eval-only-checkpoint", type=str, default="")

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
    parser.add_argument("--cls-hidden-dim", type=int, default=256)

    parser.add_argument("--root-hidden-dim", type=int, default=128)
    parser.add_argument("--root-dropout", type=float, default=0.10)
    parser.add_argument("--root-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--root-pos-weight", type=float, default=8.0)
    parser.add_argument("--use-propagation", action="store_true", default=False)
    parser.add_argument("--propagation-mode", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--propagation-hidden-dim", type=int, default=128)
    parser.add_argument("--propagation-temperature", type=float, default=1.0)
    parser.add_argument("--victim-penalty-weight", type=float, default=1.0)
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-3)

    parser.add_argument("--abnormal-weight", type=float, default=6.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--root-head-only", action="store_true", default=False)

    # Prototype/smoke controls.
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def maybe_limit(loader: Iterable, limit: int):
    if limit <= 0:
        yield from loader
        return

    for idx, batch in enumerate(loader):
        if idx >= limit:
            break
        yield batch


def build_adjacency(metadata: Dict) -> torch.Tensor:
    num_services = metadata["num_services"]
    if "adjacency_matrix" in metadata and metadata["adjacency_matrix"]:
        adj_raw = torch.tensor(metadata["adjacency_matrix"], dtype=torch.float32)
        adj_raw.fill_diagonal_(0)
        adj_2hop = (torch.mm(adj_raw, adj_raw) > 0).float()
        adjacency_matrix = ((adj_raw + adj_2hop) > 0).float()
        adjacency_matrix.fill_diagonal_(0)
        return adjacency_matrix
    adjacency_matrix = torch.ones(num_services, num_services)
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def evaluate(
    model: MultiModalV6RootHead_RCAEval,
    dataloader,
    device: torch.device,
    limit_batches: int = 0,
    service_names: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    anomaly_preds = []
    anomaly_labels = []
    root_scores = []
    anomaly_scores = []
    root_labels = []
    case_names: List[str] = []

    with torch.no_grad():
        for batch in maybe_limit(dataloader, limit_batches):
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            outputs = model(
                metrics,
                logs,
                traces,
                gt_cls,
                groundtruth_real=gt_real,
                evaluate=True,
            )

            anomaly_prob = outputs["anomaly_probs"][..., 1]
            root_prob = outputs["root_probs"]

            preds = outputs["anomaly_probs"].argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            anomaly_preds.append(preds.cpu())
            anomaly_labels.append(labels.cpu())

            root_scores.append(root_prob.cpu())
            anomaly_scores.append(anomaly_prob.cpu())
            root_labels.append(gt_real.cpu())
            case_names.extend(list(batch["case_name"]))

    anomaly_preds = torch.cat(anomaly_preds, dim=0).flatten()
    anomaly_labels = torch.cat(anomaly_labels, dim=0).flatten()

    tp = ((anomaly_preds == 1) & (anomaly_labels == 1)).sum().item()
    tn = ((anomaly_preds == 0) & (anomaly_labels == 0)).sum().item()
    fp = ((anomaly_preds == 1) & (anomaly_labels == 0)).sum().item()
    fn = ((anomaly_preds == 0) & (anomaly_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    root_scores = torch.cat(root_scores, dim=0)
    anomaly_scores = torch.cat(anomaly_scores, dim=0)
    root_labels = torch.cat(root_labels, dim=0)

    if service_names is None:
        raise ValueError("service_names are required for official RCAEval metrics")

    root_case_payload = aggregate_case_scores(
        case_names,
        root_scores.numpy(),
        root_labels.numpy(),
        service_names,
    )
    anomaly_case_payload = aggregate_case_scores(
        case_names,
        anomaly_scores.numpy(),
        root_labels.numpy(),
        service_names,
    )
    root_head_metrics = compute_service_rca_metrics(root_case_payload, service_names)
    anomaly_sort_metrics = compute_service_rca_metrics(anomaly_case_payload, service_names)

    return {
        "anomaly": {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "root_head": root_head_metrics,
        "anomaly_sorting": anomaly_sort_metrics,
    }


def build_config(args: argparse.Namespace, metadata: Dict) -> V6RootHeadRCAEvalConfig:
    return V6RootHeadRCAEvalConfig(
        num_hosts=metadata["num_services"],
        metric_dim=metadata["metrics_per_service"],
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
        root_hidden_dim=args.root_hidden_dim,
        root_dropout=args.root_dropout,
        root_loss_weight=args.root_loss_weight,
        rank_loss_weight=args.rank_loss_weight,
        root_pos_weight=args.root_pos_weight,
        use_propagation=args.use_propagation,
        propagation_mode=args.propagation_mode,
        propagation_hidden_dim=args.propagation_hidden_dim,
        propagation_temperature=args.propagation_temperature,
        victim_penalty_weight=args.victim_penalty_weight,
        sparse_loss_weight=args.sparse_loss_weight,
    )


def load_init_checkpoint(
    model: MultiModalV6RootHead_RCAEval,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint": checkpoint_path,
    }


def freeze_except_rca_head(model: MultiModalV6RootHead_RCAEval) -> None:
    for param in model.parameters():
        param.requires_grad = False
    trainable_modules = [model.root_projector, model.root_head]
    if model.propagation_scorer is not None:
        trainable_modules.append(model.propagation_scorer)
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = create_re2tt_lazy_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=args.pin_memory,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    metadata = loaders["metadata"]

    print("\n[数据信息]")
    print(f"  服务数: {metadata['num_services']}")
    print(f"  每服务指标: {metadata['metrics_per_service']}")
    print(f"  日志维度: {metadata['log_dim']}")
    print(f"  追踪维度: {metadata['trace_dim']}")

    adjacency_matrix = build_adjacency(metadata)
    config = build_config(args, metadata)
    model = MultiModalV6RootHead_RCAEval(config, adjacency_matrix).to(device)

    init_info = None
    if args.init_checkpoint:
        init_info = load_init_checkpoint(model, args.init_checkpoint, device)
        print(f"\nInitialized from checkpoint: {args.init_checkpoint}")
        if init_info["missing_keys"]:
            print(f"  Missing keys ({len(init_info['missing_keys'])}): {init_info['missing_keys'][:6]}")
        if init_info["unexpected_keys"]:
            print(f"  Unexpected keys ({len(init_info['unexpected_keys'])}): {init_info['unexpected_keys'][:6]}")

    if args.root_head_only:
        freeze_except_rca_head(model)
        rca_modules = "root_projector + root_head"
        if model.propagation_scorer is not None:
            rca_modules += " + propagation_scorer"
        print(f"  Training mode: root-head-only (frozen anomaly backbone; train {rca_modules})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = save_dir / "best_model.pth"
    summary_path = save_dir / "summary.json"

    best_tuple = (-1.0, -1.0, -1.0)  # maximize AC@1 / AC@3 / Avg@5
    patience_counter = 0

    service_names = list(metadata["services"])
    source_metrics = evaluate(
        model,
        val_loader,
        device,
        limit_batches=args.limit_eval_batches,
        service_names=service_names,
    )
    print("\n[Source checkpoint / epoch 0 validation]")
    print(f"  Root head:    {format_service_rca_metrics(source_metrics['root_head'])}")
    print(f"  Anomaly sort: {format_service_rca_metrics(source_metrics['anomaly_sorting'])}")

    if args.eval_only_checkpoint:
        eval_checkpoint = torch.load(args.eval_only_checkpoint, map_location=device)
        model.load_state_dict(eval_checkpoint["model_state_dict"])
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            limit_batches=args.limit_eval_batches,
            service_names=service_names,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            limit_batches=args.limit_eval_batches,
            service_names=service_names,
        )
        summary = {
            "save_dir": str(save_dir),
            "best_epoch": eval_checkpoint.get("epoch"),
            "config": eval_checkpoint.get("config", config.__dict__),
            "init_info": init_info,
            "source_metrics": source_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "notes": {
                "ranking_metric": "Official RCAEval service-level metrics: AC@k / Avg@5 (one ranking per case)",
                "comparison": "root_head vs anomaly_score_sorting",
                "prototype": "with propagation branch" if args.use_propagation else "no propagation branch yet",
                "training_mode": "eval-only",
                "propagation": "enabled" if args.use_propagation else "disabled",
                "propagation_mode": args.propagation_mode if args.use_propagation else "none",
                "evaluated_checkpoint": args.eval_only_checkpoint,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 100)
        print("评估完成（eval-only）")
        print("=" * 100)
        print(f"Val root head:    {format_service_rca_metrics(val_metrics['root_head'])}")
        print(f"Val anomaly sort: {format_service_rca_metrics(val_metrics['anomaly_sorting'])}")
        print(f"Test root head:   {format_service_rca_metrics(test_metrics['root_head'])}")
        print(f"Test anomaly sort:{format_service_rca_metrics(test_metrics['anomaly_sorting'])}")
        print(f"Summary saved: {summary_path}")
        return

    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loss_parts = {
            "rec_loss": 0.0,
            "cls_loss": 0.0,
            "pred_loss": 0.0,
            "root_loss": 0.0,
            "rank_loss": 0.0,
            "sparse_loss": 0.0,
        }
        num_steps = 0

        pbar = tqdm(maybe_limit(train_loader, args.limit_train_batches), desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            metrics = batch["metrics"].float().to(device)
            logs = batch["logs"].float().to(device)
            traces = batch["traces"].float().to(device)
            gt_cls = batch["groundtruth_cls"].float().to(device)
            gt_real = batch["groundtruth_real"].float().to(device)

            optimizer.zero_grad()
            losses, _ = model(
                metrics,
                logs,
                traces,
                gt_cls,
                groundtruth_real=gt_real,
                evaluate=False,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += losses["total_loss"].item()
            for name in loss_parts:
                loss_parts[name] += losses[name].item()
            num_steps += 1

            pbar.set_postfix(
                {
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "root": f"{losses['root_loss'].item():.4f}",
                    "rank": f"{losses['rank_loss'].item():.4f}",
                }
            )

        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            limit_batches=args.limit_eval_batches,
            service_names=service_names,
        )
        root_head = val_metrics["root_head"]
        current_tuple = (
            root_head["ac@1"] if root_head["ac@1"] is not None else -1.0,
            root_head["ac@3"] if root_head["ac@3"] is not None else -1.0,
            root_head["avg@5"] if root_head["avg@5"] is not None else -999.0,
        )

        avg_total = total_loss / max(1, num_steps)
        avg_losses = {k: v / max(1, num_steps) for k, v in loss_parts.items()}

        print(
            f"Epoch {epoch}: "
            f"Loss={avg_total:.4f}, "
            f"AnomalyF1={val_metrics['anomaly']['f1']:.4f}, "
            f"RootHead[{format_service_rca_metrics(val_metrics['root_head'])}], "
            f"AnomalySort[{format_service_rca_metrics(val_metrics['anomaly_sorting'])}]"
        )

        if current_tuple > best_tuple:
            best_tuple = current_tuple
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "metadata": metadata,
                    "val_metrics": val_metrics,
                    "avg_losses": avg_losses,
                    "init_info": init_info,
                    "source_metrics": source_metrics,
                },
                best_state_path,
            )
            print("  [BEST] updated by root-head validation ranking")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    if not best_state_path.exists():
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "metadata": metadata,
                "val_metrics": source_metrics,
                "avg_losses": {},
                "init_info": init_info,
                "source_metrics": source_metrics,
            },
            best_state_path,
        )
        print("  [FALLBACK] no comparable RCA case in limited validation; saved current model")

    checkpoint = torch.load(best_state_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        limit_batches=args.limit_eval_batches,
        service_names=service_names,
    )

    summary = {
        "save_dir": str(save_dir),
        "best_epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "init_info": checkpoint.get("init_info"),
        "source_metrics": checkpoint.get("source_metrics"),
        "val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "notes": {
            "ranking_metric": "Official RCAEval service-level metrics: AC@k / Avg@5 (one ranking per case)",
            "comparison": "root_head vs anomaly_score_sorting",
            "prototype": "with propagation branch" if args.use_propagation else "no propagation branch yet",
            "training_mode": "root-head-only" if args.root_head_only else "full-finetune",
            "propagation": "enabled" if args.use_propagation else "disabled",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("测试集评估")
    print("=" * 100)
    print(
        f"Best epoch={checkpoint['epoch']}, "
        f"Anomaly F1={test_metrics['anomaly']['f1']:.4f}"
    )
    print(f"Root head:     {format_service_rca_metrics(test_metrics['root_head'])}")
    print(f"Anomaly sort:  {format_service_rca_metrics(test_metrics['anomaly_sorting'])}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
