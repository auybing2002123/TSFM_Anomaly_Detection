"""
V2 超参数搜索脚本

策略：分阶段搜索
1. 第一阶段：粗搜关键参数（embed_dim, num_layers, lr）
2. 第二阶段：细调最佳配置周围的参数

运行方式：
    python scripts/msds/hyperparam_search_v2.py --phase 1  # 粗搜
    python scripts/msds/hyperparam_search_v2.py --phase 2  # 细调（需要先跑完 phase 1）
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_msds.v1 import MSDSV1Dataset
from models_msds.v2 import MultiModalV2_MSDS, V2Config


def train_and_evaluate(config_dict, train_loader, val_loader, full_dataset, 
                       device, epochs=20, patience=5, verbose=False):
    """训练并评估一个配置，返回最佳 F1"""
    
    sample = full_dataset[0]
    log_dim = sample['data_log'].shape[-1]
    trace_dim = sample['data_edge'].shape[-1]
    window_size = sample['data_node'].shape[0]
    
    adjacency_matrix = full_dataset.get_adjacency_matrix()
    if adjacency_matrix is not None:
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
    
    config = V2Config(
        metric_dim=5,
        log_dim=log_dim,
        trace_dim=trace_dim,
        window_size=window_size,
        embedding_dim=config_dict['embed_dim'],
        num_gat_layers=config_dict['num_gat_layers'],
        gat_heads=config_dict['gat_heads'],
        num_temporal_layers=config_dict['num_temporal_layers'],
        temporal_heads=config_dict['temporal_heads'],
        gat_dropout=config_dict.get('dropout', 0.1),
        temporal_dropout=config_dict.get('dropout', 0.1),
        abnormal_weight=config_dict.get('abnormal_weight', 5.0),
        learning_rate=config_dict['lr'],
        batch_size=config_dict['batch_size'],
        device=device
    )
    
    model = MultiModalV2_MSDS(config, adjacency_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'])
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            data_node = batch['data_node'].float().to(device)
            data_log = batch['data_log'].float().to(device)
            data_edge = batch['data_edge'].float().to(device)
            groundtruth_cls = batch['groundtruth_cls'].float().to(device)
            groundtruth_real = batch['groundtruth_real'].float().to(device)
            
            loss, _, _ = model(data_node, data_log, data_edge,
                              groundtruth_cls, groundtruth_real, evaluate=False)
            
            if torch.isnan(loss):
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                data_node = batch['data_node'].float().to(device)
                data_log = batch['data_log'].float().to(device)
                data_edge = batch['data_edge'].float().to(device)
                groundtruth_cls = batch['groundtruth_cls'].float().to(device)
                groundtruth_real = batch['groundtruth_real'].float().to(device)
                
                cls_result, labels = model(data_node, data_log, data_edge,
                                          groundtruth_cls, groundtruth_real, evaluate=True)
                all_preds.append(cls_result.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        pred_labels = torch.argmax(all_preds, dim=-1).reshape(-1)
        true_labels = torch.argmax(all_labels[:, :, :2], dim=-1).reshape(-1)
        unknown_mask = all_labels[:, :, 2].reshape(-1)
        valid_mask = unknown_mask == 0
        pred_labels = pred_labels[valid_mask].numpy()
        true_labels = true_labels[valid_mask].numpy()
        
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        if verbose:
            print(f"  Epoch {epoch+1}: F1={f1:.4f}, Best={best_f1:.4f}")
    
    # 清理显存
    del model, optimizer
    torch.cuda.empty_cache()
    
    return best_f1


def get_phase1_configs():
    """第一阶段：粗搜关键参数"""
    configs = []
    
    # 基础配置
    base = {
        'num_temporal_layers': 2,
        'temporal_heads': 4,
        'dropout': 0.1,
        'abnormal_weight': 5.0,
    }
    
    # 搜索空间
    embed_dims = [64, 128, 256]
    num_gat_layers_list = [2, 3]
    gat_heads_list = [4, 8]
    lrs = [5e-5, 1e-4, 5e-4]
    batch_sizes = [16, 32]
    
    for embed_dim in embed_dims:
        for num_gat_layers in num_gat_layers_list:
            for gat_heads in gat_heads_list:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        config = base.copy()
                        config.update({
                            'embed_dim': embed_dim,
                            'num_gat_layers': num_gat_layers,
                            'gat_heads': gat_heads,
                            'lr': lr,
                            'batch_size': batch_size,
                        })
                        configs.append(config)
    
    return configs


def get_phase2_configs(best_config):
    """第二阶段：在最佳配置周围细调"""
    configs = []
    
    # 在最佳配置周围微调
    base = best_config.copy()
    
    # 微调 dropout
    for dropout in [0.05, 0.1, 0.15, 0.2, 0.3]:
        config = base.copy()
        config['dropout'] = dropout
        configs.append(config)
    
    # 微调 abnormal_weight
    for weight in [3.0, 5.0, 7.0, 10.0]:
        config = base.copy()
        config['abnormal_weight'] = weight
        configs.append(config)
    
    # 微调 temporal layers
    for num_temporal_layers in [1, 2, 3]:
        for temporal_heads in [2, 4, 8]:
            config = base.copy()
            config['num_temporal_layers'] = num_temporal_layers
            config['temporal_heads'] = temporal_heads
            configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--data-path', type=str, default='data_msds/processed')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=15, help='每个配置训练的轮数')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='results/msds/v2_hyperparam')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    full_dataset = MSDSV1Dataset(args.data_path)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"数据集: 训练 {train_size}, 验证 {val_size}")
    
    if args.phase == 1:
        configs = get_phase1_configs()
        result_file = output_dir / 'phase1_results.json'
    else:
        # 读取 phase1 最佳配置
        phase1_file = output_dir / 'phase1_results.json'
        if not phase1_file.exists():
            print("错误：请先运行 phase 1")
            return
        with open(phase1_file) as f:
            phase1_results = json.load(f)
        best_config = phase1_results['best_config']
        print(f"Phase 1 最佳配置: F1={phase1_results['best_f1']:.4f}")
        print(f"配置: {best_config}")
        configs = get_phase2_configs(best_config)
        result_file = output_dir / 'phase2_results.json'
    
    print(f"\nPhase {args.phase}: 共 {len(configs)} 个配置")
    print(f"每个配置训练 {args.epochs} 轮，patience={args.patience}")
    print("=" * 60)
    
    results = []
    best_f1 = 0
    best_config = None
    
    for i, config in enumerate(configs):
        # 为每个配置创建新的 DataLoader（因为 batch_size 可能不同）
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                               shuffle=False, num_workers=0)
        
        print(f"\n[{i+1}/{len(configs)}] 测试配置:")
        print(f"  embed_dim={config['embed_dim']}, gat_layers={config['num_gat_layers']}, "
              f"gat_heads={config['gat_heads']}, lr={config['lr']}, bs={config['batch_size']}")
        
        try:
            f1 = train_and_evaluate(
                config, train_loader, val_loader, full_dataset,
                args.device, args.epochs, args.patience, verbose=False
            )
            
            results.append({'config': config, 'f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = config.copy()
                print(f"  ✓ F1={f1:.4f} (新最佳!)")
            else:
                print(f"  F1={f1:.4f}")
                
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append({'config': config, 'f1': 0, 'error': str(e)})
    
    # 保存结果
    output = {
        'phase': args.phase,
        'timestamp': datetime.now().isoformat(),
        'best_f1': best_f1,
        'best_config': best_config,
        'all_results': sorted(results, key=lambda x: x['f1'], reverse=True)
    }
    
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Phase {args.phase} 完成!")
    print(f"最佳 F1: {best_f1:.4f}")
    print(f"最佳配置: {best_config}")
    print(f"结果保存到: {result_file}")
    
    # 打印 Top 5
    print("\nTop 5 配置:")
    for i, r in enumerate(output['all_results'][:5]):
        print(f"  {i+1}. F1={r['f1']:.4f} - {r['config']}")


if __name__ == '__main__':
    main()
