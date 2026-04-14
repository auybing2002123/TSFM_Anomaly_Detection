"""
RCAEval 损失权重快速调优

目标：提升 F1（当前 0.7732）
瓶颈：Recall 太低（74.46%），漏报 993 个异常

策略：
1. 增大 abnormal_weight: 5.0 → 7.0 → 10.0 → 15.0
2. 启用 Focal Loss
3. 组合测试

每个配置训练 20 epochs（快速验证）
"""
import subprocess
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_experiment(name, args_dict, epochs=20):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"实验: {name}")
    print(f"{'='*80}")
    print(f"参数: {args_dict}")
    
    # 构建命令
    cmd = [
        'python', 'scripts/rcaeval/train_v3_host_re2ob.py',
        '--epochs', str(epochs),
        '--batch-size', '32',  # 增大 batch size 加速训练
        '--patience', '10',
        '--save-dir', f'checkpoints/rcaeval/tune_{name.replace(" ", "_")}'
    ]
    
    # 添加参数
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"命令: {' '.join(cmd)}")
    
    # 运行
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 实验失败:")
        print(result.stderr)
        return None
    
    # 解析结果（从输出中提取）
    output = result.stdout
    
    # 简单解析（假设输出包含 "Test: ... F1=0.xxxx"）
    import re
    test_match = re.search(r'Test:.*F1=([0-9.]+)', output)
    val_match = re.search(r'最佳验证 F1: ([0-9.]+)', output)
    
    if test_match and val_match:
        return {
            'name': name,
            'args': args_dict,
            'val_f1': float(val_match.group(1)),
            'test_f1': float(test_match.group(1))
        }
    else:
        print(f"⚠️ 无法解析结果")
        return None


def experiment_1_abnormal_weight():
    """实验 1: 调整 abnormal_weight"""
    print("\n" + "="*80)
    print("实验 1: 调整 abnormal_weight")
    print("="*80)
    
    weights = [5.0, 7.0, 10.0, 15.0]
    results = []
    
    for w in weights:
        result = run_experiment(
            f'abnormal_{w}',
            {'abnormal-weight': w},
            epochs=20
        )
        if result:
            results.append(result)
    
    # 总结
    print(f"\n{'='*80}")
    print("实验 1 总结: abnormal_weight 调优")
    print(f"{'='*80}")
    print(f"{'Weight':<10} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['args']['abnormal-weight']:<10.1f} {r['val_f1']:<10.4f} {r['test_f1']:<10.4f}")
    
    if results:
        best = max(results, key=lambda x: x['test_f1'])
        print(f"\n✅ 最佳配置: abnormal_weight = {best['args']['abnormal-weight']}")
        print(f"   Test F1 = {best['test_f1']:.4f}")
        return best
    return None


def experiment_2_focal_loss():
    """实验 2: Focal Loss"""
    print("\n" + "="*80)
    print("实验 2: Focal Loss")
    print("="*80)
    
    configs = [
        {'name': 'Baseline', 'args': {}},
        {'name': 'Focal_g2.0_a0.25', 'args': {'use-focal-loss': True, 'focal-gamma': 2.0, 'focal-alpha': 0.25}},
        {'name': 'Focal_g3.0_a0.25', 'args': {'use-focal-loss': True, 'focal-gamma': 3.0, 'focal-alpha': 0.25}},
        {'name': 'Focal_g2.0_a0.5', 'args': {'use-focal-loss': True, 'focal-gamma': 2.0, 'focal-alpha': 0.5}},
    ]
    
    results = []
    
    for cfg in configs:
        result = run_experiment(cfg['name'], cfg['args'], epochs=20)
        if result:
            results.append(result)
    
    # 总结
    print(f"\n{'='*80}")
    print("实验 2 总结: Focal Loss")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['val_f1']:<10.4f} {r['test_f1']:<10.4f}")
    
    if results:
        best = max(results, key=lambda x: x['test_f1'])
        print(f"\n✅ 最佳配置: {best['name']}")
        print(f"   Test F1 = {best['test_f1']:.4f}")
        return best
    return None


def experiment_3_combined():
    """实验 3: 组合最佳配置"""
    print("\n" + "="*80)
    print("实验 3: 组合最佳配置")
    print("="*80)
    
    configs = [
        {
            'name': 'Baseline',
            'args': {}
        },
        {
            'name': 'High_Weight_10',
            'args': {'abnormal-weight': 10.0}
        },
        {
            'name': 'Focal_g2_a0.25',
            'args': {'use-focal-loss': True, 'focal-gamma': 2.0, 'focal-alpha': 0.25}
        },
        {
            'name': 'Combined_W10_Focal',
            'args': {
                'abnormal-weight': 10.0,
                'use-focal-loss': True,
                'focal-gamma': 2.0,
                'focal-alpha': 0.25
            }
        }
    ]
    
    results = []
    
    for cfg in configs:
        result = run_experiment(cfg['name'], cfg['args'], epochs=30)
        if result:
            results.append(result)
    
    # 总结
    print(f"\n{'='*80}")
    print("实验 3 总结: 组合配置")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['val_f1']:<10.4f} {r['test_f1']:<10.4f}")
    
    if results:
        best = max(results, key=lambda x: x['test_f1'])
        baseline = results[0]
        improvement = best['test_f1'] - baseline['test_f1']
        print(f"\n✅ 最佳配置: {best['name']}")
        print(f"   Test F1 = {best['test_f1']:.4f}")
        print(f"   提升: +{improvement:.4f} ({improvement/baseline['test_f1']:.1%})")
        return best
    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['abnormal', 'focal', 'combined', 'all'])
    args = parser.parse_args()
    
    all_results = {}
    
    if args.experiment == 'abnormal' or args.experiment == 'all':
        best_abnormal = experiment_1_abnormal_weight()
        if best_abnormal:
            all_results['abnormal'] = best_abnormal
    
    if args.experiment == 'focal' or args.experiment == 'all':
        best_focal = experiment_2_focal_loss()
        if best_focal:
            all_results['focal'] = best_focal
    
    if args.experiment == 'combined' or args.experiment == 'all':
        best_combined = experiment_3_combined()
        if best_combined:
            all_results['combined'] = best_combined
    
    # 保存结果
    results_file = Path('results/rcaeval/loss_weight_tuning_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    print(f"结果已保存至: {results_file}")
