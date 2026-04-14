"""测试最优阈值搜索功能"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# 模拟预测分数和真实标签
np.random.seed(42)
n_samples = 1000
n_anomalies = 120  # 12% 异常

# 生成模拟数据
labels = np.zeros(n_samples)
labels[:n_anomalies] = 1
np.random.shuffle(labels)

# 生成预测分数（异常样本分数稍高）
probs = np.random.rand(n_samples) * 0.3 + 0.3  # 基础分数 0.3-0.6
probs[labels == 1] += 0.1  # 异常样本加 0.1

print("=" * 60)
print("测试最优阈值搜索")
print("=" * 60)

print(f"\n数据统计:")
print(f"总样本数: {n_samples}")
print(f"异常样本数: {n_anomalies} ({n_anomalies/n_samples*100:.1f}%)")
print(f"分数范围: {probs.min():.4f} ~ {probs.max():.4f}")

# 方法 1: 固定阈值 0.5
print("\n" + "=" * 60)
print("方法 1: 固定阈值 0.5")
print("=" * 60)
threshold_fixed = 0.5
preds_fixed = (probs > threshold_fixed).astype(int)
p1, r1, f1_1, _ = precision_recall_fscore_support(
    labels, preds_fixed, average='binary', zero_division=0
)
print(f"阈值: {threshold_fixed:.4f}")
print(f"预测异常数: {preds_fixed.sum()}")
print(f"Precision: {p1:.4f}, Recall: {r1:.4f}, F1: {f1_1:.4f}")

# 方法 2: 动态阈值（中位数）
print("\n" + "=" * 60)
print("方法 2: 动态阈值（中位数）")
print("=" * 60)
threshold_median = float(np.median(probs))
preds_median = (probs > threshold_median).astype(int)
p2, r2, f1_2, _ = precision_recall_fscore_support(
    labels, preds_median, average='binary', zero_division=0
)
print(f"阈值: {threshold_median:.4f}")
print(f"预测异常数: {preds_median.sum()}")
print(f"Precision: {p2:.4f}, Recall: {r2:.4f}, F1: {f1_2:.4f}")

# 方法 3: 最优阈值搜索
print("\n" + "=" * 60)
print("方法 3: 最优阈值搜索")
print("=" * 60)

best_f1 = 0
best_threshold = 0.5
best_precision = 0
best_recall = 0

for threshold in np.linspace(0.1, 0.9, 81):
    preds_temp = (probs > threshold).astype(int)
    p_temp, r_temp, f1_temp, _ = precision_recall_fscore_support(
        labels, preds_temp, average='binary', zero_division=0
    )
    
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_threshold = threshold
        best_precision = p_temp
        best_recall = r_temp

preds_optimal = (probs > best_threshold).astype(int)
print(f"最优阈值: {best_threshold:.4f}")
print(f"预测异常数: {preds_optimal.sum()}")
print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")

# 对比总结
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"\n{'方法':<20} {'阈值':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 60)
print(f"{'固定阈值 0.5':<20} {threshold_fixed:<10.4f} {f1_1:<10.4f} {p1:<12.4f} {r1:<10.4f}")
print(f"{'动态阈值(median)':<20} {threshold_median:<10.4f} {f1_2:<10.4f} {p2:<12.4f} {r2:<10.4f}")
print(f"{'最优阈值搜索':<20} {best_threshold:<10.4f} {best_f1:<10.4f} {best_precision:<12.4f} {best_recall:<10.4f}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print(f"\n✅ 最优阈值搜索 F1 提升: {(best_f1 - f1_2) / f1_2 * 100:.1f}% (相比动态阈值)")
print(f"✅ 最优阈值搜索 F1 提升: {(best_f1 - f1_1) / max(f1_1, 0.0001) * 100:.1f}% (相比固定阈值)")
