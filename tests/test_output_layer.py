"""
输出层单元测试（Phase 3）
测试因果分析、根因定位、可视化工具
"""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.causal_analysis import TemporalCausalAnalyzer
from utils.root_cause import RootCauseAnalyzer


class TestTemporalCausalAnalyzer:
    """时序因果分析器测试"""
    
    def test_analyze_basic(self):
        """测试基本因果分析"""
        analyzer = TemporalCausalAnalyzer()
        
        # 模拟异常分数：Metrics 先异常，然后 Logs，最后 Traces
        anomaly_scores = {
            'Metrics': np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3]),  # 第2步开始异常
            'Logs': np.array([0.1, 0.1, 0.2, 0.6, 0.8, 0.7]),     # 第3步开始异常
            'Traces': np.array([0.1, 0.1, 0.1, 0.2, 0.6, 0.9])    # 第4步开始异常
        }
        
        result = analyzer.analyze(anomaly_scores, threshold=0.5)
        
        assert 'causal_chain' in result
        assert 'first_anomaly_times' in result
        assert 'anomaly_counts' in result
        
        # 因果链应该是 Metrics -> Logs -> Traces
        assert result['causal_chain'] == ['Metrics', 'Logs', 'Traces']
        assert result['first_anomaly_times']['Metrics'] == 2
        assert result['first_anomaly_times']['Logs'] == 3
        assert result['first_anomaly_times']['Traces'] == 4
    
    def test_analyze_no_anomaly(self):
        """测试无异常情况"""
        analyzer = TemporalCausalAnalyzer()
        
        anomaly_scores = {
            'Metrics': np.array([0.1, 0.2, 0.3]),
            'Logs': np.array([0.1, 0.2, 0.3]),
            'Traces': np.array([0.1, 0.2, 0.3])
        }
        
        result = analyzer.analyze(anomaly_scores, threshold=0.5)
        
        assert result['causal_chain'] == []
        assert all(t == float('inf') for t in result['first_anomaly_times'].values())
    
    def test_format_causal_chain(self):
        """测试因果链格式化"""
        analyzer = TemporalCausalAnalyzer()
        
        chain = ['Metrics', 'Logs', 'Traces']
        formatted = analyzer.format_causal_chain(chain)
        
        assert formatted == 'Metrics → Logs → Traces'
        
        # 空链
        assert analyzer.format_causal_chain([]) == '无明显因果链'
    
    def test_format_report(self):
        """测试报告格式化"""
        analyzer = TemporalCausalAnalyzer()
        
        result = {
            'causal_chain': ['Metrics', 'Logs'],
            'first_anomaly_times': {'Metrics': 5, 'Logs': 10, 'Traces': float('inf')},
            'anomaly_counts': {'Metrics': 3, 'Logs': 2, 'Traces': 0}
        }
        
        report = analyzer.format_report(result)
        
        assert '时序因果分析报告' in report
        assert 'Metrics → Logs' in report
        assert '时间步 5' in report
    
    def test_get_causal_explanation(self):
        """测试因果解释生成"""
        analyzer = TemporalCausalAnalyzer()
        
        result = {
            'causal_chain': ['Metrics', 'Logs', 'Traces'],
            'first_anomaly_times': {'Metrics': 10, 'Logs': 15, 'Traces': 20}
        }
        
        explanation = analyzer.get_causal_explanation(result)
        
        assert 'Metrics' in explanation
        assert 'Logs' in explanation
        assert '延迟' in explanation


class TestRootCauseAnalyzer:
    """根因定位分析器测试"""
    
    def test_analyze_attention(self):
        """测试注意力分析"""
        analyzer = RootCauseAnalyzer()
        
        # 模拟注意力权重 (6, 6) - 2个时间步 x 3个模态
        attn_weights = torch.randn(6, 6).softmax(dim=-1)
        modality_ids = torch.tensor([0, 1, 2, 0, 1, 2])  # Metrics, Logs, Traces, ...
        anomaly_indices = [3, 4]  # 第2个时间步的 Metrics 和 Logs
        
        results = analyzer.analyze_attention(attn_weights, modality_ids, anomaly_indices)
        
        assert len(results) == 2
        for result in results:
            assert 'timestamp_idx' in result
            assert 'root_causes' in result
            assert 'top_cause' in result
            
            # 检查根因排序
            causes = result['root_causes']
            assert len(causes) == 3
            assert sum(score for _, score in causes) == pytest.approx(1.0, rel=0.01)
    
    def test_analyze_causal_matrix(self):
        """测试因果矩阵分析"""
        analyzer = RootCauseAnalyzer()
        
        # 模拟因果矩阵：Metrics 对其他模态有强影响
        causal_matrix = torch.tensor([
            [0.5, 0.8, 0.7],  # Metrics -> others
            [0.2, 0.5, 0.3],  # Logs -> others
            [0.1, 0.2, 0.5]   # Traces -> others
        ])
        
        result = analyzer.analyze_causal_matrix(causal_matrix, threshold=0.5)
        
        assert 'relationships' in result
        assert 'influence_scores' in result
        assert 'ranked_modalities' in result
        assert 'most_influential' in result
        
        # Metrics 应该是最有影响力的
        assert result['most_influential'][0] == 'Metrics'
    
    def test_format_report(self):
        """测试报告格式化"""
        analyzer = RootCauseAnalyzer()
        
        results = [
            {
                'timestamp_idx': 5,
                'root_causes': [('Metrics', 0.5), ('Logs', 0.3), ('Traces', 0.2)],
                'top_cause': ('Metrics', 0.5)
            }
        ]
        
        report = analyzer.format_report(results)
        
        assert '根因定位分析报告' in report
        assert '时间点 5' in report
        assert 'Metrics' in report
    
    def test_get_root_cause_summary(self):
        """测试根因摘要生成"""
        analyzer = RootCauseAnalyzer()
        
        attention_results = [
            {'timestamp_idx': 1, 'top_cause': ('Metrics', 0.6)},
            {'timestamp_idx': 2, 'top_cause': ('Metrics', 0.5)},
            {'timestamp_idx': 3, 'top_cause': ('Logs', 0.4)}
        ]
        
        causal_matrix_result = {
            'most_influential': ('Metrics', 0.45),
            'ranked_modalities': [('Metrics', 0.45), ('Logs', 0.35), ('Traces', 0.2)]
        }
        
        summary = analyzer.get_root_cause_summary(attention_results, causal_matrix_result)
        
        assert 'Metrics' in summary
        assert '影响力' in summary or '主要原因' in summary


class TestCausalVisualizer:
    """可视化工具测试（不实际绘图，只测试逻辑）"""
    
    def test_import(self):
        """测试可视化模块可以导入"""
        from utils.visualization import CausalVisualizer
        visualizer = CausalVisualizer()
        assert visualizer is not None
    
    @pytest.mark.skipif(
        True,  # 跳过实际绘图测试
        reason="Visualization tests require display"
    )
    def test_plot_causal_matrix(self):
        """测试因果矩阵绘图"""
        from utils.visualization import CausalVisualizer
        
        visualizer = CausalVisualizer()
        causal_matrix = np.random.rand(3, 3)
        
        fig = visualizer.plot_causal_matrix(causal_matrix)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
