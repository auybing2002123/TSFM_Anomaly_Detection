"""
多模态归因器

整合多种归因方法，提供统一的接口。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .integrated_gradients import IntegratedGradients, AttributionResult


@dataclass
class FeatureImportance:
    """特征重要性结果"""
    # Top-K 重要特征
    top_metric_features: List[Tuple[str, float]]  # [(特征名, 贡献), ...]
    top_log_templates: List[Tuple[int, float]]    # [(模板ID, 贡献), ...]
    top_trace_edges: List[Tuple[Tuple[int, int], float]]  # [((src, dst), 贡献), ...]
    
    # 原始归因
    raw_attribution: AttributionResult


class MultiModalAttributor:
    """
    多模态归因器
    
    提供多层次的异常解释：
    1. 主机级：哪个主机是异常源
    2. 模态级：哪个模态贡献最大
    3. 时间级：异常从什么时候开始
    4. 特征级：哪些具体特征导致异常
    """
    
    # MSDS 数据集的特征名
    METRIC_NAMES = ['cpu.user', 'mem.used', 'load.min1', 'load.min5', 'load.min15']
    HOST_NAMES = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    MODALITY_NAMES = ['Metrics', 'Logs', 'Traces']
    
    def __init__(
        self,
        model: nn.Module,
        baseline_type: str = 'zero',
        steps: int = 50,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 训练好的异常检测模型
            baseline_type: baseline 类型 ('zero' 或 'mean')
            steps: Integrated Gradients 的积分步数
            device: 计算设备
        """
        self.model = model
        self.device = device
        
        # 初始化 Integrated Gradients
        self.ig = IntegratedGradients(
            model=model,
            baseline_type=baseline_type,
            steps=steps,
            device=device
        )
    
    def explain(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_host: Optional[int] = None,
        top_k: int = 5
    ) -> Dict:
        """
        生成异常解释
        
        Args:
            data_node: (B, T, N, metric_dim) Metrics
            data_log: (B, T, N, log_dim) Logs
            data_edge: (B, T, N, N, trace_dim) Traces
            target_host: 目标主机（如果为 None，则解释所有主机）
            top_k: 返回 Top-K 重要特征
        
        Returns:
            解释结果字典
        """
        # 计算 Integrated Gradients
        attr_result = self.ig.attribute(
            data_node, data_log, data_edge, target_host
        )
        
        # 生成多层次解释
        explanations = []
        B = data_node.shape[0]
        
        for b in range(B):
            exp = self._generate_explanation(
                attr_result, b, top_k
            )
            explanations.append(exp)
        
        return {
            'explanations': explanations,
            'raw_attribution': attr_result
        }
    
    def _generate_explanation(
        self,
        attr_result: AttributionResult,
        batch_idx: int,
        top_k: int
    ) -> Dict:
        """为单个样本生成解释"""
        
        # 异常分数和预测
        anomaly_score = attr_result.anomaly_score[batch_idx].cpu().numpy()
        prediction = attr_result.prediction[batch_idx].cpu().numpy()
        
        # Level 1: 主机级
        host_attr = attr_result.host_attribution[batch_idx].cpu().numpy()
        host_ranking = self._rank_hosts(host_attr)
        
        # Level 2: 模态级
        modality_attr = attr_result.modality_attribution[batch_idx].cpu().numpy()
        modality_ranking = self._rank_modalities(modality_attr)
        
        # Level 3: 时间级
        temporal_attr = attr_result.temporal_attribution[batch_idx].cpu().numpy()
        temporal_analysis = self._analyze_temporal(temporal_attr)
        
        # Level 4: 特征级
        feature_importance = self._get_feature_importance(
            attr_result, batch_idx, top_k
        )
        
        return {
            'anomaly_score': anomaly_score,
            'prediction': prediction,
            'host_ranking': host_ranking,
            'modality_ranking': modality_ranking,
            'temporal_analysis': temporal_analysis,
            'feature_importance': feature_importance
        }
    
    def _rank_hosts(self, host_attr: np.ndarray) -> List[Dict]:
        """排序主机贡献"""
        ranking = []
        sorted_indices = np.argsort(host_attr)[::-1]
        
        for idx in sorted_indices:
            ranking.append({
                'host': self.HOST_NAMES[idx],
                'host_idx': int(idx),
                'contribution': float(host_attr[idx]),
                'is_anomaly': host_attr[idx] > 0.3  # 简单阈值
            })
        
        return ranking
    
    def _rank_modalities(self, modality_attr: np.ndarray) -> List[Dict]:
        """排序模态贡献"""
        ranking = []
        sorted_indices = np.argsort(modality_attr)[::-1]
        
        for idx in sorted_indices:
            ranking.append({
                'modality': self.MODALITY_NAMES[idx],
                'contribution': float(modality_attr[idx])
            })
        
        return ranking
    
    def _analyze_temporal(self, temporal_attr: np.ndarray) -> Dict:
        """分析时间模式"""
        T = len(temporal_attr)
        
        # 找到异常开始时间（贡献开始显著增加的时间点）
        threshold = temporal_attr.mean() + temporal_attr.std()
        anomaly_start = None
        for t in range(T):
            if temporal_attr[t] > threshold:
                anomaly_start = t
                break
        
        # 找到异常峰值时间
        anomaly_peak = int(np.argmax(temporal_attr))
        
        # 生成时间模式可视化（ASCII）
        pattern = self._generate_temporal_pattern(temporal_attr)
        
        return {
            'anomaly_start': anomaly_start,
            'anomaly_peak': anomaly_peak,
            'pattern': pattern,
            'attribution': temporal_attr.tolist()
        }
    
    def _generate_temporal_pattern(self, temporal_attr: np.ndarray) -> str:
        """生成时间模式的 ASCII 可视化"""
        # 归一化到 0-8
        normalized = (temporal_attr - temporal_attr.min()) / (temporal_attr.max() - temporal_attr.min() + 1e-8)
        levels = (normalized * 8).astype(int)
        
        # ASCII 字符
        chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '█']
        pattern = ''.join([chars[l] for l in levels])
        
        return f'[{pattern}]'
    
    def _get_feature_importance(
        self,
        attr_result: AttributionResult,
        batch_idx: int,
        top_k: int
    ) -> Dict:
        """获取特征级重要性"""
        
        # Metric 特征
        metric_attr = attr_result.metric_attribution[batch_idx].abs()  # (T, N, metric_dim)
        metric_importance = metric_attr.sum(dim=0).cpu().numpy()  # (N, metric_dim)
        top_metrics = self._get_top_metric_features(metric_importance, top_k)
        
        # Log 模板
        log_attr = attr_result.log_attribution[batch_idx].abs()  # (T, N, log_dim)
        log_importance = log_attr.sum(dim=0).cpu().numpy()  # (N, log_dim)
        top_logs = self._get_top_log_templates(log_importance, top_k)
        
        # Trace 边
        trace_attr = attr_result.trace_attribution[batch_idx].abs()  # (T, N, N, trace_dim)
        trace_importance = trace_attr.sum(dim=(0, 3)).cpu().numpy()  # (N, N)
        top_traces = self._get_top_trace_edges(trace_importance, top_k)
        
        return {
            'top_metrics': top_metrics,
            'top_logs': top_logs,
            'top_traces': top_traces
        }
    
    def _get_top_metric_features(
        self,
        importance: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """获取 Top-K 重要的 Metric 特征"""
        N, D = importance.shape
        
        # 展平并排序
        flat_importance = importance.flatten()
        top_indices = np.argsort(flat_importance)[::-1][:top_k]
        
        results = []
        total = flat_importance.sum()
        
        for idx in top_indices:
            host_idx = idx // D
            feat_idx = idx % D
            
            results.append({
                'host': self.HOST_NAMES[host_idx],
                'feature': self.METRIC_NAMES[feat_idx] if feat_idx < len(self.METRIC_NAMES) else f'metric_{feat_idx}',
                'contribution': float(flat_importance[idx]),
                'percentage': float(flat_importance[idx] / total * 100) if total > 0 else 0
            })
        
        return results
    
    def _get_top_log_templates(
        self,
        importance: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """获取 Top-K 重要的 Log 模板"""
        N, D = importance.shape
        
        # 展平并排序
        flat_importance = importance.flatten()
        top_indices = np.argsort(flat_importance)[::-1][:top_k]
        
        results = []
        total = flat_importance.sum()
        
        for idx in top_indices:
            host_idx = idx // D
            template_idx = idx % D
            
            results.append({
                'host': self.HOST_NAMES[host_idx],
                'template_id': int(template_idx),
                'contribution': float(flat_importance[idx]),
                'percentage': float(flat_importance[idx] / total * 100) if total > 0 else 0
            })
        
        return results
    
    def _get_top_trace_edges(
        self,
        importance: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """获取 Top-K 重要的 Trace 边"""
        N = importance.shape[0]
        
        # 展平并排序
        flat_importance = importance.flatten()
        top_indices = np.argsort(flat_importance)[::-1][:top_k]
        
        results = []
        total = flat_importance.sum()
        
        for idx in top_indices:
            src_idx = idx // N
            dst_idx = idx % N
            
            if flat_importance[idx] > 0:  # 只返回有贡献的边
                results.append({
                    'source': self.HOST_NAMES[src_idx],
                    'target': self.HOST_NAMES[dst_idx],
                    'contribution': float(flat_importance[idx]),
                    'percentage': float(flat_importance[idx] / total * 100) if total > 0 else 0
                })
        
        return results
    
    def generate_report(
        self,
        explanation: Dict,
        sample_idx: int = 0
    ) -> str:
        """
        生成人类可读的解释报告
        
        Args:
            explanation: explain() 的返回结果
            sample_idx: 样本索引
        
        Returns:
            格式化的报告字符串
        """
        exp = explanation['explanations'][sample_idx]
        
        lines = []
        lines.append("=" * 65)
        lines.append("                    异常检测解释报告")
        lines.append("=" * 65)
        lines.append("")
        
        # 检测结果
        anomaly_hosts = [i for i, s in enumerate(exp['anomaly_score']) if s > 0.5]
        if anomaly_hosts:
            lines.append(f"检测结果: 🔴 异常 (置信度: {max(exp['anomaly_score']):.1%})")
        else:
            lines.append(f"检测结果: 🟢 正常 (最高异常分数: {max(exp['anomaly_score']):.1%})")
        lines.append("")
        
        # Level 1: 主机分析
        lines.append("-" * 65)
        lines.append("【主机分析】")
        for h in exp['host_ranking']:
            icon = "🔴" if h['is_anomaly'] else "🟢"
            lines.append(f"  {icon} {h['host']}: 贡献 {h['contribution']:.1%}")
        lines.append("")
        
        # Level 2: 模态分析
        lines.append("-" * 65)
        lines.append("【模态分析】")
        for m in exp['modality_ranking']:
            bar_len = int(m['contribution'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {m['modality']:8s}: [{bar}] {m['contribution']:.1%}")
        lines.append("")
        
        # Level 3: 时间分析
        lines.append("-" * 65)
        lines.append("【时间分析】")
        ta = exp['temporal_analysis']
        if ta['anomaly_start'] is not None:
            lines.append(f"  异常开始: t={ta['anomaly_start']}")
        lines.append(f"  异常峰值: t={ta['anomaly_peak']}")
        lines.append(f"  时间模式: {ta['pattern']}")
        lines.append("")
        
        # Level 4: 特征分析
        lines.append("-" * 65)
        lines.append("【特征分析】(Top 5)")
        lines.append("")
        
        lines.append("  Metrics:")
        for i, f in enumerate(exp['feature_importance']['top_metrics'][:5], 1):
            lines.append(f"    {i}. {f['host']}/{f['feature']}: {f['percentage']:.1f}%")
        
        lines.append("")
        lines.append("  Logs (模板ID):")
        for i, f in enumerate(exp['feature_importance']['top_logs'][:5], 1):
            lines.append(f"    {i}. {f['host']}/template_{f['template_id']}: {f['percentage']:.1f}%")
        
        lines.append("")
        lines.append("  Traces (调用边):")
        for i, f in enumerate(exp['feature_importance']['top_traces'][:5], 1):
            lines.append(f"    {i}. {f['source']} → {f['target']}: {f['percentage']:.1f}%")
        
        lines.append("")
        lines.append("=" * 65)
        
        return "\n".join(lines)
