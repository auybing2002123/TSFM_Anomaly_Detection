"""
V3.1 训练脚本：融合版（跨模态架构 + 跨主机因果）
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_msds.v1 import MSDSV1Dataset
from data_msds.dataset_loader import load_msds_temporal_split
from models_msds.v3.config import V3Config
from models_msds.v3.model_v3_1 import MultiModalV3_1_MSDS


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3.1 Model')
    parser.add_argument('--data-dir', type=str, default='data_msds/processed')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--gat-heads', type=int, default=4)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    causal_matrix = None
    
    with torch.no_grad():
        for batch in dataloader:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)

            cls_probs, _, causal_matrix = model(
                data_node, data_log, data_edge, gt_cls, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    preds = torch.cat(all_preds).flatten()
    labels = torch.cat(all_labels).flatten()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, recall, precision, tp, tn, fp, fn, causal_matrix


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据（60/10/30 时序划分）
    print(f"Loading data from {args.data_dir}...")
    splits = load_msds_temporal_split(args.data_dir)
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']
    full_dataset = splits['full']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 邻接矩阵
    adj = splits['adjacency']
    if adj is not None:
        adj = torch.from_numpy(adj).float()

    # 配置
    config = V3Config(
        embedding_dim=args.embed_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        num_temporal_layers=args.num_temporal_layers,
        temporal_heads=args.temporal_heads,
        causal_loss_weight=args.causal_loss_weight,
        sparse_loss_weight=args.sparse_loss_weight,
    )
    
    # 模型
    model = MultiModalV3_1_MSDS(config, adj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 保存路径
    save_dir = Path('checkpoints/msds/v3_1')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_rec, total_cls, total_causal = 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, _ = model(
                data_node, data_log, data_edge, gt_cls
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_cls += cls_loss.item()
            total_causal += causal_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        n = len(train_loader)

        # 评估
        f1, recall, prec, tp, tn, fp, fn, causal_mat = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train: loss={total_loss/n:.4f}, rec={total_rec/n:.4f}, cls={total_cls/n:.4f}, causal={total_causal/n:.4f}")
        print(f"Val: F1={f1:.4f}, Recall={recall*100:.2f}%, Precision={prec*100:.2f}%")
        print(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # 打印因果矩阵
        if causal_mat is not None:
            print("Causal Matrix (5×5):")
            mat = causal_mat.cpu().numpy()
            header = "     " + "  ".join([f"{h[-3:]}" for h in host_names])
            print(header)
            for i, name in enumerate(host_names):
                row = f"{name[-3:]}  [" + ", ".join([f"{mat[i,j]:.2f}" for j in range(5)]) + "]"
                print(row)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"⭐ New best F1: {f1:.4f}")
        print()
    
    # 根因定位（使用测试集）
    print("=" * 60)
    print("加载最佳模型进行根因定位分析...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
    
    root_cause_counts = {name: 0 for name in host_names}
    total_anomalies = 0
    
    with torch.no_grad():
        for batch in test_loader:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            labels = gt_real.argmax(dim=-1)
            has_anomaly = labels.sum(dim=1) > 0
            if has_anomaly.sum() == 0:
                continue
            
            results = model.locate_root_cause(
                data_node[has_anomaly], data_log[has_anomaly],
                data_edge[has_anomaly], gt_cls[has_anomaly]
            )
            for r in results:
                if r['root_cause_name'] is not None:
                    total_anomalies += 1
                    root_cause_counts[r['root_cause_name']] += 1
    
    print(f"根因定位分析 ({total_anomalies} 个异常样本):")
    print("-" * 40)
    for name in host_names:
        cnt = root_cause_counts[name]
        pct = cnt / total_anomalies * 100 if total_anomalies > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"{name}: {cnt:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"模型保存至: {save_dir / 'best_model.pth'}")
    
    # 加载最佳模型，在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint)
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"      (TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    
    print(f"\n最终结果: Val F1={best_f1:.4f}, Test F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
