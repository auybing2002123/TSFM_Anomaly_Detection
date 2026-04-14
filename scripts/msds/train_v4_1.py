"""
V4.1 训练脚本：日志编码器 → TinyBERT

与 V3 跨主机因果版本对比，唯一改动是日志编码方式：
- V3: 模板计数 (256维) → Linear → 64维
- V4.1: TinyBERT 嵌入 (312维) → Linear → 64维

用法:
    # 1. 先预计算日志嵌入（只需运行一次）
    python data_msds/precompute_log_embeddings.py --data-dir datasets/MSDS
    
    # 2. 训练 V4.1 模型
    python scripts/msds/train_v4_1.py --epochs 30
    
    # 或者使用模板模式（与 V3 相同，用于对比）
    python scripts/msds/train_v4_1.py --log-encoder template
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_msds.v4.config import V4Config
from models_msds.v4.model import MultiModalV4_MSDS
from data_msds.v4_dataset import MSDSV4Dataset
from data_msds.dataset_loader import load_msds_temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train V4.1 Model (TinyBERT Log Encoder)')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default='data_msds/processed',
                        help='预处理数据目录')
    parser.add_argument('--log-embeddings', type=str, 
                        default='data_msds/processed/log_embeddings/log_embeddings.pkl',
                        help='预计算日志嵌入路径（bert/bert_attn 模式）')
    parser.add_argument('--log-embeddings-attn', type=str, 
                        default='data_msds/processed/log_embeddings_attn/log_embeddings.pkl',
                        help='Attention Pooling 模式的日志嵌入路径')
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 日志编码器
    parser.add_argument('--log-encoder', type=str, default='bert',
                        choices=['bert', 'bert_attn', 'bert_cluster', 'template'],
                        help='日志编码器类型: bert=mean pooling, bert_attn=attention pooling, bert_cluster=语义聚类+计数, template=模板计数')
    parser.add_argument('--bert-model', type=str, 
                        default='huawei-noah/TinyBERT_General_4L_312D',
                        help='BERT 模型名称')
    parser.add_argument('--bert-freeze', action='store_true', default=True,
                        help='冻结 BERT 参数')
    parser.add_argument('--attn-pool-heads', type=int, default=4,
                        help='Attention Pooling heads 数量')
    parser.add_argument('--max-logs-per-host', type=int, default=10,
                        help='每主机每时间点最大日志数')
    parser.add_argument('--num-clusters', type=int, default=64,
                        help='语义聚类数量（bert_cluster 模式）')
    parser.add_argument('--log-embeddings-cluster', type=str, 
                        default='data_msds/processed/log_embeddings_cluster/log_embeddings.pkl',
                        help='语义聚类模式的日志嵌入路径')
    
    # 模型
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=8)
    parser.add_argument('--num-gat-layers', type=int, default=2)
    parser.add_argument('--num-temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=4)
    
    # 因果
    parser.add_argument('--use-causal', action='store_true', default=True)
    parser.add_argument('--no-causal', action='store_false', dest='use_causal')
    parser.add_argument('--causal-init-off-diag', type=float, default=0.1)
    parser.add_argument('--sparse-loss-weight', type=float, default=1.0)
    parser.add_argument('--dag-loss-weight', type=float, default=0.5)
    parser.add_argument('--causal-loss-weight', type=float, default=0.1)
    parser.add_argument('--use-gumbel', action='store_true', default=True)
    parser.add_argument('--gumbel-temperature', type=float, default=0.5)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_causal_matrices = []
    
    with torch.no_grad():
        for batch in dataloader:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            gt_real = batch['groundtruth_real'].float().to(device)
            
            # Attention Pooling 模式需要 log_mask
            log_mask = None
            if 'log_mask' in batch:
                log_mask = batch['log_mask'].float().to(device)
            
            cls_probs, _, causal_matrix = model(
                data_node, data_log, data_edge, gt_cls, log_mask=log_mask, evaluate=True
            )
            
            preds = cls_probs.argmax(dim=-1)
            labels = gt_real.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            if causal_matrix is not None:
                all_causal_matrices.append(causal_matrix.cpu())
    
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
    
    avg_causal = None
    if all_causal_matrices:
        avg_causal = torch.stack(all_causal_matrices).mean(dim=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'causal_matrix': avg_causal
    }


def print_causal_matrix(causal_matrix, host_names=None):
    """打印因果矩阵"""
    if causal_matrix is None:
        return
    
    if host_names is None:
        host_names = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    
    print("\n因果矩阵:")
    print("         " + "  ".join([f"{n[-3:]}" for n in host_names]))
    
    for i, name in enumerate(host_names):
        row = causal_matrix[i].numpy()
        row_str = "  ".join([f"{v:.2f}" for v in row])
        print(f"{name[-3:]}  [{row_str}]")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确定日志嵌入路径
    log_emb_path = None
    if args.log_encoder == 'bert':
        log_emb_path = args.log_embeddings
        if not Path(log_emb_path).exists():
            print(f"\n⚠️  预计算日志嵌入不存在: {log_emb_path}")
            print("请先运行: python data_msds/precompute_log_embeddings.py --pooling mean")
            print("或使用模板模式: --log-encoder template")
            return
    elif args.log_encoder == 'bert_attn':
        log_emb_path = args.log_embeddings_attn
        if not Path(log_emb_path).exists():
            print(f"\n⚠️  Attention Pooling 日志嵌入不存在: {log_emb_path}")
            print("请先运行: python data_msds/precompute_log_embeddings.py --pooling none --output-dir data_msds/processed/log_embeddings_attn")
            print("或使用其他模式: --log-encoder bert 或 --log-encoder template")
            return
    elif args.log_encoder == 'bert_cluster':
        log_emb_path = args.log_embeddings_cluster
        if not Path(log_emb_path).exists():
            print(f"\n⚠️  语义聚类日志嵌入不存在: {log_emb_path}")
            print(f"请先运行: python data_msds/precompute_log_embeddings.py --pooling cluster --num-clusters {args.num_clusters} --output-dir data_msds/processed/log_embeddings_cluster")
            print("或使用其他模式: --log-encoder bert 或 --log-encoder template")
            return
    
    # 加载数据（时序划分：60% 训练 / 10% 验证 / 30% 测试）
    print(f"\n加载数据: {args.data_dir}")
    
    # 使用 V4 数据集（支持日志嵌入）
    full_dataset = MSDSV4Dataset(
        args.data_dir,
        log_embeddings_path=log_emb_path
    )
    
    # 时序划分（不随机打乱）
    total_size = len(full_dataset)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")
    
    # 获取邻接矩阵
    adjacency_matrix = full_dataset.get_adjacency_matrix()
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
    
    # 创建配置
    config = V4Config(
        # 日志编码器配置
        log_encoder_type=args.log_encoder,
        bert_model_name=args.bert_model,
        bert_freeze=args.bert_freeze,
        precomputed_log_embeddings=log_emb_path,
        attn_pool_heads=args.attn_pool_heads,
        max_logs_per_host=args.max_logs_per_host,
        num_clusters=args.num_clusters,
        
        # 模型配置
        embedding_dim=args.embed_dim,
        gat_heads=args.gat_heads,
        num_gat_layers=args.num_gat_layers,
        num_temporal_layers=args.num_temporal_layers,
        temporal_heads=args.temporal_heads,
        
        # 因果配置
        use_causal=args.use_causal,
        causal_init_off_diag=args.causal_init_off_diag,
        sparse_loss_weight=args.sparse_loss_weight,
        dag_loss_weight=args.dag_loss_weight,
        causal_loss_weight=args.causal_loss_weight,
        use_gumbel_softmax=args.use_gumbel,
        gumbel_temperature=args.gumbel_temperature
    )
    
    # 创建模型
    model = MultiModalV4_MSDS(config, adjacency_matrix).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 保存目录
    save_dir = Path(f'checkpoints/msds/v4_1_{args.log_encoder}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    best_f1 = 0.0
    patience_counter = 0
    
    print(f"\n开始训练 {args.epochs} 轮...")
    print(f"日志编码器: {args.log_encoder.upper()}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rec_loss_sum = 0.0
        cls_loss_sum = 0.0
        causal_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            gt_cls = batch['groundtruth_cls'].float().to(device)
            
            # Attention Pooling 模式需要 log_mask
            log_mask = None
            if 'log_mask' in batch:
                log_mask = batch['log_mask'].float().to(device)
            
            optimizer.zero_grad()
            loss, rec_loss, cls_loss, causal_loss, _ = model(
                data_node, data_log, data_edge, gt_cls, log_mask=log_mask
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            rec_loss_sum += rec_loss.item()
            cls_loss_sum += cls_loss.item()
            causal_loss_sum += causal_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{rec_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 在验证集上评估（用于 early stopping）
        metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Recall={metrics['recall']*100:.1f}%, "
              f"Precision={metrics['precision']*100:.1f}%")
        
        print_causal_matrix(metrics['causal_matrix'])
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'f1': best_f1,
                'causal_matrix': metrics['causal_matrix'],
                'log_encoder': args.log_encoder
            }, save_dir / 'best_model.pth')
            
            print(f"  ⭐ New best Val F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # 在测试集上最终评估
    print(f"\n{'='*60}")
    print("Final Test on held-out test set")
    print(f"{'='*60}")
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"📊 Test Results (on {len(test_dataset)} samples):")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Confusion: TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    
    print_causal_matrix(test_metrics['causal_matrix'])
    
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"  Best Val F1:  {best_f1:.4f}")
    print(f"  Test F1:      {test_metrics['f1']:.4f}")
    print(f"  模型保存至: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
