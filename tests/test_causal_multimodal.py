"""
因果多模态模型单元测试
"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 直接导入，避免 models/__init__.py 的问题
from models.encoders.metric_encoder import MetricEncoder
from models.encoders.log_encoder import LogEncoder
from models.encoders.trace_encoder import TraceEncoder
from models.causal_attention import CausalMultiModalAttention


class TestEncoders:
    """编码器测试"""
    
    def test_metric_encoder_shape(self):
        """测试 MetricEncoder 输出形状"""
        encoder = MetricEncoder(n_metrics=50, hidden_dim=768)
        x = torch.randn(2, 10, 50)  # (B, T, n_metrics)
        out = encoder(x)
        assert out.shape == (2, 10, 768)
    
    def test_log_encoder_shape(self):
        """测试 LogEncoder 输出形状"""
        encoder = LogEncoder(log_features=4, hidden_dim=768)
        x = torch.randn(2, 10, 4)  # (B, T, log_features)
        out = encoder(x)
        assert out.shape == (2, 10, 768)
    
    def test_trace_encoder_shape(self):
        """测试 TraceEncoder 输出形状"""
        encoder = TraceEncoder(trace_features=5, hidden_dim=768)
        x = torch.randn(2, 10, 5)  # (B, T, trace_features)
        out = encoder(x)
        assert out.shape == (2, 10, 768)
    
    def test_encoder_gradient_flow(self):
        """测试编码器梯度流"""
        encoder = MetricEncoder(n_metrics=10, hidden_dim=64)
        x = torch.randn(2, 5, 10, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestCausalAttention:
    """因果注意力测试"""
    
    def test_causal_attention_shape(self):
        """测试因果注意力输出形状"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        B, T = 2, 15  # T = 5 * 3 modalities
        x = torch.randn(B, T, 64)
        modality_ids = torch.arange(3).repeat(5).unsqueeze(0).expand(B, -1)
        
        output, attn_weights, causal_matrix = attn(x, modality_ids)
        
        assert output.shape == (B, T, 64)
        assert attn_weights.shape == (B, T, T)
        assert causal_matrix.shape == (3, 3)
    
    def test_causal_matrix_range(self):
        """测试因果矩阵值在 [0, 1] 范围内"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        causal_matrix = attn.get_causal_matrix()
        
        assert causal_matrix.min() >= 0
        assert causal_matrix.max() <= 1
    
    def test_temporal_mask(self):
        """测试时序因果 mask"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        mask = attn.get_temporal_mask(5, torch.device('cpu'))
        
        # 应该是下三角矩阵
        assert mask.shape == (5, 5)
        assert mask[0, 0] == True   # 可以看自己
        assert mask[0, 1] == False  # 不能看未来
        assert mask[4, 0] == True   # 可以看过去
    
    def test_modality_mask(self):
        """测试模态因果 mask"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        B, T = 2, 6
        modality_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        
        mask = attn.get_modality_mask(modality_ids)
        
        assert mask.shape == (B, T, T)
        assert mask.min() >= 0
        assert mask.max() <= 1
    
    def test_causal_interpretation(self):
        """测试因果矩阵解释"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        # 手动设置因果矩阵使某些关系明显
        with torch.no_grad():
            attn.causal_matrix[0, 1] = 2.0  # Metrics -> Logs 强因果
            attn.causal_matrix[1, 2] = 2.0  # Logs -> Traces 强因果
        
        interpretation = attn.get_causal_interpretation()
        
        assert 'matrix' in interpretation
        assert 'relationships' in interpretation
        assert len(interpretation['relationships']) >= 2
    
    def test_gradient_flow(self):
        """测试因果注意力梯度流"""
        attn = CausalMultiModalAttention(hidden_dim=64, n_modalities=3, n_heads=4)
        
        B, T = 2, 9
        x = torch.randn(B, T, 64, requires_grad=True)
        modality_ids = torch.arange(3).repeat(3).unsqueeze(0).expand(B, -1)
        
        output, _, _ = attn(x, modality_ids)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attn.causal_matrix.grad is not None


class TestCausalMultiModalGPT2:
    """主模型测试（需要 GPT-2 缓存）"""
    
    @pytest.fixture
    def model_config(self):
        return {
            'n_metrics': 50,
            'log_features': 4,
            'trace_features': 5,
            'hidden_dim': 768,
            'use_causal_attention': True
        }
    
    def test_model_import(self):
        """测试模型可以导入"""
        from models.causal_multimodal_gpt2 import CausalMultiModalGPT2
        assert CausalMultiModalGPT2 is not None
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent.parent / "cache" / "models--gpt2").exists(),
        reason="GPT-2 cache not available"
    )
    def test_model_forward(self, model_config):
        """测试模型前向传播"""
        from models.causal_multimodal_gpt2 import CausalMultiModalGPT2
        
        model = CausalMultiModalGPT2(**model_config)
        
        B, T = 2, 10
        metrics = torch.randn(B, T, model_config['n_metrics'])
        logs = torch.randn(B, T, model_config['log_features'])
        traces = torch.randn(B, T, model_config['trace_features'])
        
        outputs = model(metrics, logs, traces)
        
        assert 'anomaly_scores' in outputs
        assert outputs['anomaly_scores'].shape == (B, T)
        assert outputs['causal_matrix'] is not None
        assert outputs['causal_matrix'].shape == (3, 3)
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent.parent / "cache" / "models--gpt2").exists(),
        reason="GPT-2 cache not available"
    )
    def test_model_without_causal_attention(self, model_config):
        """测试无因果注意力的消融模型"""
        from models.causal_multimodal_gpt2 import CausalMultiModalGPT2
        
        model_config['use_causal_attention'] = False
        model = CausalMultiModalGPT2(**model_config)
        
        B, T = 2, 10
        metrics = torch.randn(B, T, model_config['n_metrics'])
        logs = torch.randn(B, T, model_config['log_features'])
        traces = torch.randn(B, T, model_config['trace_features'])
        
        outputs = model(metrics, logs, traces)
        
        assert outputs['anomaly_scores'].shape == (B, T)
        assert outputs['causal_matrix'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
