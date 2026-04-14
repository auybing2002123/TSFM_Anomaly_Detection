"""
根因定位分析器

基于 Attention 权重分析异常的根本原因
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class RootCauseAnalyzer:
    """
    根因定位分析器
    
    通过分析注意力权重来定位异常的根本原因
    """
    
    def __init__(self, modality_names: List[str] = None):
        """
        Args:
            modality_names: 模态名称列表，默认 ['Metrics', 'Logs', 'Traces']
        """
        self.modality_names = modality_names or ['Metrics', 'Logs', 'Traces']
    
    def analyze_attention(
        self,
        attn_weights: torch.Tensor,
        modality_ids: torch.Tensor,
        anomaly_indices: List[int]
    ) -> List[Dict]:
        """
        基于 Attention 权重分析根因
        
        Args:
            attn_weights: (T, T) 注意力权重矩阵
            modality_ids: (T,) 每个位置的模态 ID
            anomaly_indices: 异常时间点的索引列表
        
        Returns:
            每个异常时间点的根因分析结果列表
        """
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.detach().cpu().numpy()
        if isinstance(modality_ids, torch.Tensor):
            modality_ids = modality_ids.detach().cpu().numpy()
        
        results = []
        
        for idx in anomaly_indices:
            if idx >= len(attn_weights):
                continue
                
            # 获取该时间点的 attention 分布
            attn = attn_weights[idx]  # (T,)
            
            # 按模态聚合注意力权重
            modality_scores = {}
            for m_id, m_name in enumerate(self.modality_names):
                mask = modality_ids == m_id
                if mask.any():
                    modality_scores[m_name] = float(attn[mask].sum())
                else:
                    modality_scores[m_name] = 0.0
            
            # 归一化
            total = sum(modality_scores.values())
            if total > 0:
                modality_scores = {k: v / total for k, v in modality_scores.items()}
            
            # 按贡献度排序
            sorted_causes = sorted(
                modality_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            results.append({
                'timestamp_idx': idx,
                'root_causes': sorted_causes,
                'top_cause': sorted_causes[0] if sorted_causes else None
            })
        
        return results
    
    def analyze_causal_matrix(
        self,
        causal_matrix: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict:
        """
        分析学到的因果矩阵
        
        Args:
            causal_matrix: (n_modalities, n_modalities) 因果矩阵
            threshold: 因果关系阈值
        
        Returns:
            因果关系分析结果
        """
        if isinstance(causal_matrix, torch.Tensor):
            causal_matrix = causal_matrix.detach().cpu().numpy()
        
        relationships = []
        influence_scores = {m: 0.0 for m in self.modality_names}
        
        for i, src in enumerate(self.modality_names):
            for j, tgt in enumerate(self.modality_names):
                strength = float(causal_matrix[i, j])
                
                if i != j and strength > threshold:
                    relationships.append({
                        'source': src,
                        'target': tgt,
                        'strength': strength
                    })
                
                # 累计影响力（作为源的总权重）
                if i != j:
                    influence_scores[src] += strength
        
        # 归一化影响力分数
        total_influence = sum(influence_scores.values())
        if total_influence > 0:
            influence_scores = {k: v / total_influence for k, v in influence_scores.items()}
        
        # 按影响力排序
        ranked_modalities = sorted(
            influence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'relationships': relationships,
            'influence_scores': influence_scores,
            'ranked_modalities': ranked_modalities,
            'most_influential': ranked_modalities[0] if ranked_modalities else None
        }
    
    def format_report(self, analysis_results: List[Dict]) -> str:
        """
        格式化根因分析报告
        
        Args:
            analysis_results: analyze_attention() 的返回结果
        
        Returns:
            格式化的报告字符串
        """
        lines = ["=" * 50, "根因定位分析报告", "=" * 50, ""]
        
        for result in analysis_results:
            idx = result['timestamp_idx']
            lines.append(f"时间点 {idx}:")
            
            for i, (modality, score) in enumerate(result['root_causes'], 1):
                bar = '█' * int(score * 20)
                lines.append(f"  {i}. [{modality}] {bar} {score:.1%}")
            
            lines.append("")
        
        lines.append("=" * 50)
        return '\n'.join(lines)
    
    def format_causal_matrix_report(self, analysis_result: Dict) -> str:
        """
        格式化因果矩阵分析报告
        
        Args:
            analysis_result: analyze_causal_matrix() 的返回结果
        
        Returns:
            格式化的报告字符串
        """
        lines = ["=" * 50, "因果矩阵分析报告", "=" * 50, ""]
        
        # 因果关系
        lines.append("检测到的因果关系:")
        relationships = analysis_result.get('relationships', [])
        if relationships:
            for rel in relationships:
                lines.append(
                    f"  {rel['source']} → {rel['target']} "
                    f"(强度: {rel['strength']:.2f})"
                )
        else:
            lines.append("  无显著因果关系")
        lines.append("")
        
        # 影响力排名
        lines.append("模态影响力排名:")
        ranked = analysis_result.get('ranked_modalities', [])
        for i, (modality, score) in enumerate(ranked, 1):
            bar = '█' * int(score * 20)
            lines.append(f"  {i}. [{modality}] {bar} {score:.1%}")
        
        lines.append("")
        lines.append("=" * 50)
        return '\n'.join(lines)
    
    def get_root_cause_summary(
        self,
        attention_results: List[Dict],
        causal_matrix_result: Dict
    ) -> str:
        """
        生成综合的根因分析摘要
        
        Args:
            attention_results: analyze_attention() 的结果
            causal_matrix_result: analyze_causal_matrix() 的结果
        
        Returns:
            自然语言摘要
        """
        summaries = []
        
        # 从因果矩阵分析
        most_influential = causal_matrix_result.get('most_influential')
        if most_influential:
            summaries.append(
                f"根据学到的因果结构，{most_influential[0]} 是最具影响力的模态 "
                f"(影响力: {most_influential[1]:.1%})"
            )
        
        # 从注意力分析
        if attention_results:
            # 统计各模态作为 top cause 的次数
            top_cause_counts = {}
            for result in attention_results:
                if result.get('top_cause'):
                    modality = result['top_cause'][0]
                    top_cause_counts[modality] = top_cause_counts.get(modality, 0) + 1
            
            if top_cause_counts:
                most_common = max(top_cause_counts.items(), key=lambda x: x[1])
                summaries.append(
                    f"在 {len(attention_results)} 个异常时间点中，"
                    f"{most_common[0]} 最常被识别为主要原因 ({most_common[1]} 次)"
                )
        
        return "；".join(summaries) + "。" if summaries else "无法确定根因。"
