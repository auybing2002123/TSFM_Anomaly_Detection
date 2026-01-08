#!/usr/bin/env python
"""
诊断脚本：分析模型为什么 F1 和 AUC 很低

检查项目：
1. 数据标签分布
2. Anomaly score 分布（正常 vs 异常样本）
3. 重构误差 vs 预测误差
4. Backbone embedding 质量
5. Score fusion 权重
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_data(datasets):
    """诊断数据集"""
    print("\n" + "="*60)
    print("1. 数据诊断")
    print("="*60)
    
    test_dataset = datasets['test_dataset']
    labels = test_dataset.get_all_labels().numpy()
    
    anomaly_ratio = labels.mean()
    print(f"  测试集大小: {len(labels)}")
    print(f"  异常比例: {anomaly_ratio*100:.2f}%")
    print(f"  正常样本: {(labels==0).sum()}")
    print(f"  异常样本: {(labels==1).sum()}")
    
    # 检查标签是否全是 0 或全是 1
    if anomaly_ratio == 0:
        print("  ⚠️ 警告: 测试集没有异常样本！")
    elif anomaly_ratio == 1:
        print("  ⚠️ 警告: 测试集全是异常样本！")
    
    return labels


def diagnose_scores(model, test_loader, device, labels, max_batches=20):
    """诊断 anomaly score 分布"""
    print("\n" + "="*60)
    print("2. Anomaly Score 诊断")
    print("="*60)
    
    model.eval()
    all_scores = []
    all_recon_scores = []
    all_pred_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            x = batch['data'].to(device)
            output = model(x)
            
            all_scores.append(output['anomaly_score'].cpu().numpy())
            all_recon_scores.append(output['recon_score'].cpu().numpy())
            all_pred_scores.append(output['pred_score'].cpu().numpy())
            if 'label' in batch:
                all_labels.append(batch['label'].numpy())
    
    scores = np.concatenate(all_scores)
    recon_scores = np.concatenate(all_recon_scores)
    pred_scores = np.concatenate(all_pred_scores)
    
    # 使用收集到的标签
    if all_labels:
        labels = np.concatenate(all_labels)
    else:
        # 如果没有标签，使用传入的
        labels = labels[:len(scores)]
    
    # 分离正常和异常样本的 score
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    print(f"\n  总体 Score 统计:")
    print(f"    范围: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"    均值: {scores.mean():.6f}, 标准差: {scores.std():.6f}")
    
    print(f"\n  正常样本 Score:")
    print(f"    均值: {scores[normal_mask].mean():.6f}")
    print(f"    标准差: {scores[normal_mask].std():.6f}")
    
    print(f"\n  异常样本 Score:")
    print(f"    均值: {scores[anomaly_mask].mean():.6f}")
    print(f"    标准差: {scores[anomaly_mask].std():.6f}")
    
    # 关键指标：正常和异常的 score 差异
    score_diff = scores[anomaly_mask].mean() - scores[normal_mask].mean()
    print(f"\n  ★ Score 差异 (异常 - 正常): {score_diff:.6f}")
    
    if score_diff <= 0:
        print("  ⚠️ 严重问题: 异常样本的 score 不比正常样本高！")
        print("     模型无法区分正常和异常。")
    elif score_diff < scores.std() * 0.5:
        print("  ⚠️ 问题: Score 差异太小，区分度不够。")
    
    # 计算 AUC
    try:
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)
        print(f"\n  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")
        
        if auc_roc < 0.55:
            print("  ⚠️ AUC-ROC 接近随机 (0.5)，模型没有学到有效特征。")
    except:
        print("  无法计算 AUC")
    
    # 分析重构 vs 预测
    print(f"\n  重构 Score 统计:")
    print(f"    正常均值: {recon_scores[normal_mask].mean():.6f}")
    print(f"    异常均值: {recon_scores[anomaly_mask].mean():.6f}")
    print(f"    差异: {recon_scores[anomaly_mask].mean() - recon_scores[normal_mask].mean():.6f}")
    
    print(f"\n  预测 Score 统计:")
    print(f"    正常均值: {pred_scores[normal_mask].mean():.6f}")
    print(f"    异常均值: {pred_scores[anomaly_mask].mean():.6f}")
    print(f"    差异: {pred_scores[anomaly_mask].mean() - pred_scores[normal_mask].mean():.6f}")
    
    # 单独计算 recon 和 pred 的 AUC
    try:
        recon_auc = roc_auc_score(labels, recon_scores)
        pred_auc = roc_auc_score(labels, pred_scores)
        print(f"\n  重构 AUC-ROC: {recon_auc:.4f}")
        print(f"  预测 AUC-ROC: {pred_auc:.4f}")
    except:
        pass
    
    return scores, recon_scores, pred_scores


def diagnose_fusion_weights(model):
    """诊断 score fusion 权重"""
    print("\n" + "="*60)
    print("3. Score Fusion 权重")
    print("="*60)
    
    model._init_detection_head()
    alpha, beta = model.detection_head.score_fusion.get_weights()
    print(f"  α (重构权重): {alpha.item():.4f}")
    print(f"  β (预测权重): {beta.item():.4f}")


def diagnose_embeddings(model, test_loader, device, labels):
    """诊断 backbone embedding 质量"""
    print("\n" + "="*60)
    print("4. Backbone Embedding 诊断")
    print("="*60)
    
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # 只取前 10 个 batch
                break
            x = batch['data'].to(device)
            output = model(x)
            all_embeddings.append(output['embeddings'].cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding 范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"  Embedding 均值: {embeddings.mean():.4f}")
    print(f"  Embedding 标准差: {embeddings.std():.4f}")
    
    # 检查 embedding 是否有变化
    if embeddings.std() < 0.01:
        print("  ⚠️ 警告: Embedding 方差很小，可能 backbone 没有提取有效特征。")
    
    # 检查不同样本的 embedding 是否相似
    sample_var = embeddings.var(axis=0).mean()
    print(f"  样本间方差: {sample_var:.6f}")
    
    if sample_var < 0.001:
        print("  ⚠️ 警告: 不同样本的 embedding 几乎相同！")


def diagnose_reconstruction(model, test_loader, device):
    """诊断重构质量"""
    print("\n" + "="*60)
    print("5. 重构质量诊断")
    print("="*60)
    
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch['data'].to(device)
        output = model(x)
        
        recon = output['recon']
        
        # 计算重构误差
        mse = ((recon - x) ** 2).mean().item()
        mae = (recon - x).abs().mean().item()
        
        print(f"  重构 MSE: {mse:.6f}")
        print(f"  重构 MAE: {mae:.6f}")
        
        # 检查重构是否接近输入
        correlation = np.corrcoef(
            x.cpu().numpy().flatten()[:1000],
            recon.cpu().numpy().flatten()[:1000]
        )[0, 1]
        print(f"  输入-重构相关性: {correlation:.4f}")
        
        if correlation < 0.5:
            print("  ⚠️ 重构质量差，模型可能没有学好。")


def suggest_fixes(scores, labels, auc_roc=None):
    """根据诊断结果给出修复建议"""
    print("\n" + "="*60)
    print("6. 修复建议")
    print("="*60)
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    score_diff = scores[anomaly_mask].mean() - scores[normal_mask].mean()
    
    suggestions = []
    
    if score_diff <= 0:
        suggestions.append("1. 【严重】异常样本 score 不高于正常样本")
        suggestions.append("   - 检查标签是否正确")
        suggestions.append("   - 尝试只用重构误差作为 score（不用 fusion）")
        suggestions.append("   - 检查 backbone embedding 是否有区分度")
    
    if auc_roc and auc_roc < 0.55:
        suggestions.append("2. 【严重】AUC 接近随机")
        suggestions.append("   - Chronos 可能不适合这个任务")
        suggestions.append("   - 尝试更简单的 baseline（如 AutoEncoder）")
        suggestions.append("   - 检查数据预处理是否正确")
    
    if labels.mean() < 0.01:
        suggestions.append("3. 异常比例很低 (<1%)")
        suggestions.append("   - 这是正常的，但需要调整阈值策略")
        suggestions.append("   - 使用 anomaly_ratio 方法设置阈值")
    
    if not suggestions:
        suggestions.append("未发现明显问题，可能需要更多训练或调参。")
    
    for s in suggestions:
        print(f"  {s}")


def main():
    parser = argparse.ArgumentParser(description='诊断模型问题')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default='amazon/chronos-t5-mini')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TSFM-AD 模型诊断")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载数据
    loader = TSFMADDataLoader(config={
        'window_size': args.window_size,
        'stride': args.window_size,  # 使用 stride = window_size
        'val_ratio': 0.2,
    })
    datasets = loader.load_dataset(args.dataset)
    
    test_loader = DataLoader(
        datasets['test_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    n_features = datasets['metadata']['n_features']
    
    # 创建模型（未训练）
    model_config = {
        'model': {
            'backbone': args.backbone,
            'freeze_backbone': True,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'pooling': 'attention',
        },
        'data': {
            'n_features': n_features,
            'window_size': args.window_size,
        },
        'paths': {
            'cache_dir': 'cache',
        },
    }
    
    model = TSFMADModel(config=model_config, use_lora=False)
    model = model.to(device)
    
    # 运行诊断
    labels = diagnose_data(datasets)
    scores, recon_scores, pred_scores = diagnose_scores(model, test_loader, device, labels)
    diagnose_fusion_weights(model)
    diagnose_embeddings(model, test_loader, device, labels)
    diagnose_reconstruction(model, test_loader, device)
    
    try:
        auc_roc = roc_auc_score(labels, scores)
    except:
        auc_roc = None
    
    suggest_fixes(scores, labels, auc_roc)
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    main()
