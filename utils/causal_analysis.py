"""
时序因果分析工具

基于时间先后推断因果链（保底方案）
核心思想：原因在前，结果在后
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class TemporalCausalAnalyzer:
    """
    基于时序的因果分析器
    
    通过分析各模态异常出现的时间顺序来推断因果关系
    """
    
    def __init__(self, modality_names: List[str] = None):
        """
        Args:
            modality_names: 模态名称列表，默认 ['Metrics', 'Logs', 'Traces']
        """
        self.modality_names = modality_names or ['Metrics', 'Logs', 'Traces']
    
    def analyze(
        self,
        anomaly_scores: Dict[str, np.ndarray],
        threshold: float = 0.5
    ) -> Dict:
        """
        分析各模态异常出现的时间顺序
        
        Args:
            anomaly_scores: {modality_name: (T,) scores} 各模态的异常分数
            threshold: 异常阈值
        
        Returns:
            {
                'causal_chain': ['Metrics', 'Logs', 'Traces'],
                'first_anomaly_times': {'Metrics': 10, 'Logs': 15, 'Traces': 20},
                'anomaly_counts': {'Metrics': 5, 'Logs': 3, 'Traces': 2}
            }
        """
        first_anomaly_times = {}
        anomaly_counts = {}
        
        for modality, scores in anomaly_scores.items():
            scores = np.asarray(scores)
            anomaly_mask = scores > threshold
            anomaly_counts[modality] = int(anomaly_mask.sum())
            
            if anomaly_mask.any():
                first_anomaly_times[modality] = int(np.argmax(anomaly_mask))
            else:
                first_anomaly_times[modality] = float('inf')
        
        # 按首次异常时间排序，得到因果链
        causal_chain = sorted(
            [m for m in first_anomaly_times.keys() if first_anomaly_times[m] != float('inf')],
            key=lambda x: first_anomaly_times[x]
        )
        
        return {
            'causal_chain': causal_chain,
            'first_anomaly_times': first_anomaly_times,
            'anomaly_counts': anomaly_counts
        }
    
    def analyze_window(
        self,
        anomaly_scores: Dict[str, np.ndarray],
        window_start: int,
        window_end: int,
        threshold: float = 0.5
    ) -> Dict:
        """
        分析指定时间窗口内的因果关系
        
        Args:
            anomaly_scores: 各模态的异常分数
            window_start: 窗口起始索引
            window_end: 窗口结束索引
            threshold: 异常阈值
        
        Returns:
            窗口内的因果分析结果
        """
        windowed_scores = {
            m: scores[window_start:window_end] 
            for m, scores in anomaly_scores.items()
        }
        return self.analyze(windowed_scores, threshold)
    
    def format_causal_chain(self, causal_chain: List[str]) -> str:
        """格式化因果链为可读字符串"""
        if not causal_chain:
            return "无明显因果链"
        return ' → '.join(causal_chain)
    
    def format_report(self, analysis_result: Dict) -> str:
        """
        格式化完整的因果分析报告
        
        Args:
            analysis_result: analyze() 的返回结果
        
        Returns:
            格式化的报告字符串
        """
        lines = ["=" * 50, "时序因果分析报告", "=" * 50, ""]
        
        # 因果链
        chain = analysis_result.get('causal_chain', [])
        lines.append(f"推断的因果链: {self.format_causal_chain(chain)}")
        lines.append("")
        
        # 首次异常时间
        lines.append("各模态首次异常时间:")
        first_times = analysis_result.get('first_anomaly_times', {})
        for modality in self.modality_names:
            time = first_times.get(modality, float('inf'))
            if time == float('inf'):
                lines.append(f"  - {modality}: 无异常")
            else:
                lines.append(f"  - {modality}: 时间步 {time}")
        lines.append("")
        
        # 异常计数
        lines.append("各模态异常数量:")
        counts = analysis_result.get('anomaly_counts', {})
        for modality in self.modality_names:
            count = counts.get(modality, 0)
            lines.append(f"  - {modality}: {count}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return '\n'.join(lines)
    
    def get_causal_explanation(self, analysis_result: Dict) -> str:
        """
        生成因果关系的自然语言解释
        
        Args:
            analysis_result: analyze() 的返回结果
        
        Returns:
            自然语言解释
        """
        chain = analysis_result.get('causal_chain', [])
        first_times = analysis_result.get('first_anomaly_times', {})
        
        if len(chain) == 0:
            return "未检测到明显的异常模式。"
        
        if len(chain) == 1:
            return f"仅在 {chain[0]} 模态检测到异常。"
        
        explanations = []
        for i in range(len(chain) - 1):
            src, tgt = chain[i], chain[i + 1]
            src_time = first_times.get(src, 0)
            tgt_time = first_times.get(tgt, 0)
            delay = tgt_time - src_time
            explanations.append(
                f"{src} 异常（时间步 {src_time}）可能导致了 "
                f"{tgt} 异常（时间步 {tgt_time}，延迟 {delay} 步）"
            )
        
        return "；".join(explanations) + "。"
