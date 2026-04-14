"""
V1 训练脚本

用法:
    python scripts/gaia/train_v1.py --config configs/gaia/v1/default.yaml
"""
import argparse
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_gaia.v1.dataset import GAIAV1Dataset
from models_gaia.v1.model import GAIAV1Model
from models_gaia.v1.config import V1Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='V1 训练')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gaia/v1/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='数据文件路径（覆盖配置文件中的设置）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='训练设备'
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, data_path_override: str = None) -> tuple:
    """创建数据加载器"""
    data_config = config['data']
    
    # 使用命令行参数覆盖配置文件
    if data_path_override:
        data_path = project_root / data_path_override
    else:
        data_path = project_root / data_config['data_path']
    
    logger.info(f"加载数据: {data_path}")
    
    # 先加载数据，自动检测 n_metrics
    data_dict = torch.load(data_path, weights_only=False)
    actual_n_metrics = data_dict['metrics'].shape[1]
    
    # 更新配置中的 n_metrics
    if config['model']['n_metrics'] != actual_n_metrics:
        logger.warning(f"配置文件中 n_metrics={config['model']['n_metrics']}，实际数据 n_metrics={actual_n_metrics}")
        logger.info(f"自动更新为 n_metrics={actual_n_metrics}")
        config['model']['n_metrics'] = actual_n_metrics
    
    # 创建数据集
    train_dataset = GAIAV1Dataset(
        data_path=str(data_path),
        seq_len=data_config['seq_len'],
        stride=data_config['stride'],
        split='train',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio']
    )
    
    val_dataset = GAIAV1Dataset(
        data_path=str(data_path),
        seq_len=data_config['seq_len'],
        stride=data_config['stride'],
        split='val',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: dict) -> GAIAV1Model:
    """创建模型"""
    model_config = config['model']
    
    v1_config = V1Config(
        n_metrics=model_config['n_metrics'],
        log_features=model_config['log_features'],
        trace_features=model_config['trace_features'],
        hidden_dim=model_config['hidden_dim'],
        gpt_layers=model_config['gpt_layers'],
        freeze_gpt2=model_config['freeze_gpt2'],
        train_ln=model_config['train_ln'],
        train_wpe=model_config['train_wpe'],
        head_hidden_dim=model_config['head_hidden_dim'],
        dropout=model_config['dropout'],
        cache_dir=model_config.get('cache_dir')
    )
    
    model = GAIAV1Model(v1_config)
    
    # 打印参数统计
    param_stats = model.count_parameters()
    logger.info("模型参数统计:")
    for key, value in param_stats.items():
        logger.info(f"  {key}: {value:,}")
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    max_grad_norm: float = 1.0
) -> dict:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (metrics, logs, traces, labels) in enumerate(pbar):
        # 移动到设备
        metrics = metrics.to(device)
        logs = logs.to(device)
        traces = traces.to(device)
        labels = labels.to(device).float()
        
        # 检查输入是否有 NaN/Inf
        if torch.isnan(metrics).any() or torch.isnan(logs).any() or torch.isnan(traces).any():
            logger.warning(f"Batch {batch_idx}: 输入包含 NaN，跳过")
            continue
        if torch.isinf(metrics).any() or torch.isinf(logs).any() or torch.isinf(traces).any():
            logger.warning(f"Batch {batch_idx}: 输入包含 Inf，跳过")
            continue
        
        # 前向传播
        scores, info = model(metrics, logs, traces)
        
        # 检查输出是否有 NaN/Inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.error(f"Batch {batch_idx}: 模型输出包含 NaN/Inf")
            logger.error(f"  Encoder std: {info}")
            continue
        
        # 计算损失
        loss = criterion(scores, labels)
        
        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Batch {batch_idx}: 损失为 NaN/Inf")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 统计
        batch_size = metrics.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / total_samples:.4f}'
        })
    
    return {
        'loss': total_loss / total_samples if total_samples > 0 else float('inf')
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """验证"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for metrics, logs, traces, labels in tqdm(val_loader, desc="Validation"):
            metrics = metrics.to(device)
            logs = logs.to(device)
            traces = traces.to(device)
            labels = labels.to(device).float()
            
            scores, _ = model(metrics, logs, traces)
            loss = criterion(scores, labels)
            
            batch_size = metrics.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
    
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算分类指标
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    import numpy as np
    
    # 转为概率
    probs = torch.sigmoid(all_scores).numpy()
    labels_np = all_labels.numpy()
    
    # 确保是 1D 数组（二分类）
    if probs.ndim > 1:
        probs = probs.flatten()
    if labels_np.ndim > 1:
        labels_np = labels_np.flatten()
    
    # 🔍 调试信息：查看预测分数分布
    logger.info(f"  Score stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    
    # 🎯 最优阈值搜索：在验证集上搜索最大化 F1 的阈值
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    # 尝试 81 个不同的阈值（从 0.1 到 0.9）
    for threshold in np.linspace(0.1, 0.9, 81):
        preds_temp = (probs > threshold).astype(int)
        precision_temp, recall_temp, f1_temp, _ = precision_recall_fscore_support(
            labels_np, preds_temp, average='binary', zero_division=0
        )
        
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
            best_precision = precision_temp
            best_recall = recall_temp
    
    logger.info(f"  Optimal threshold: {best_threshold:.4f} (searched from 0.1 to 0.9)")
    logger.info(f"  At optimal threshold - F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    
    # 使用最优阈值进行预测
    preds = (probs > best_threshold).astype(int)
    
    logger.info(f"  Pred distribution: {preds.sum()} / {len(probs)} predicted as anomaly")
    logger.info(f"  True distribution: {labels_np.sum()} / {len(labels_np)} are anomaly")
    
    # 计算最终指标（使用最优阈值）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels_np, probs)
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / total_samples,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'best_threshold': best_threshold,  # 保存最优阈值
        'scores': all_scores,
        'labels': all_labels
    }


def main():
    args = parse_args()
    
    # 加载配置
    config_path = project_root / args.config
    config = load_config(config_path)
    
    logger.info(f"配置文件: {config_path}")
    logger.info(f"训练设备: {args.device}")
    
    # 创建输出目录
    output_dir = project_root / config['training']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器（自动检测 n_metrics）
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(config, args.data_path)
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config)
    model = model.to(args.device)
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 创建损失函数（加类别权重）
    # 异常比例 12.46%，正常比例 87.54%
    # pos_weight = 正常数 / 异常数 ≈ 7.0
    pos_weight = torch.tensor([7.0]).to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logger.info(f"使用类别权重: pos_weight={pos_weight.item():.1f}")
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, args.device)
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于 F1）
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_auc': val_metrics['auc'],
                'best_threshold': val_metrics['best_threshold'],  # 保存最优阈值
                'config': config
            }, checkpoint_path)
            logger.info(f"✅ 保存最佳模型 (F1={val_metrics['f1']:.4f}, Threshold={val_metrics['best_threshold']:.4f}): {checkpoint_path}")
        
        # 定期保存检查点
        if epoch % config['training'].get('save_interval', 10) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")
    
    logger.info("\n🎉 训练完成！")
    logger.info(f"最佳验证 F1: {best_val_f1:.4f}")
    logger.info(f"最佳验证 Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
