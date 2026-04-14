"""
V4.1 日志编码器：TinyBERT

替换原始的模板计数方式，使用预训练语言模型编码日志语义。

TinyBERT 特点：
- 模型：huawei-noah/TinyBERT_General_4L_312D
- 参数量：14.5M（比 BERT-base 小 7.5 倍）
- 隐藏维度：312
- 层数：4
- 推理速度：比 BERT-base 快 9.4 倍

使用方式：
1. 离线预计算：将原始日志文本编码为嵌入向量，保存到文件
2. 在线加载：训练时直接加载预计算的嵌入，无需运行 BERT

这样做的好处：
- 训练时不需要 GPU 内存存放 BERT 模型
- 训练速度更快（无需前向传播 BERT）
- 可以复用嵌入（多次实验不需要重复计算）
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np


class TinyBERTLogEncoder(nn.Module):
    """
    TinyBERT 日志编码器
    
    支持两种模式：
    1. 在线模式：实时编码日志文本（需要 transformers）
    2. 离线模式：加载预计算的嵌入向量
    
    Args:
        model_name: HuggingFace 模型名称
        hidden_dim: BERT 隐藏维度（TinyBERT 为 312）
        output_dim: 输出维度（投影到模型的 embedding_dim）
        freeze: 是否冻结 BERT 参数
        pooling: 池化方式 ('cls' 或 'mean')
        cache_dir: 模型缓存目录
        precomputed_path: 预计算嵌入路径（如果提供，则使用离线模式）
    """
    
    def __init__(
        self,
        model_name: str = 'huawei-noah/TinyBERT_General_4L_312D',
        hidden_dim: int = 312,
        output_dim: int = 64,
        freeze: bool = True,
        pooling: str = 'cls',
        cache_dir: Optional[str] = None,
        precomputed_path: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze = freeze
        self.pooling = pooling
        self.precomputed_path = precomputed_path
        
        # 投影层：将 BERT 维度投影到模型维度
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # 位置编码（用于时序建模）
        self.max_len = 100
        pe = torch.zeros(1, self.max_len, output_dim)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, output_dim, 2).float() * 
            (-np.log(10000.0) / output_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 离线模式：不加载 BERT
        if precomputed_path is not None:
            self.bert = None
            self.tokenizer = None
            self._mode = 'offline'
            print(f"TinyBERT 日志编码器初始化（离线模式）")
            print(f"  预计算嵌入路径: {precomputed_path}")
        else:
            # 在线模式：加载 BERT
            self._mode = 'online'
            self._load_bert(cache_dir)
    
    def _load_bert(self, cache_dir: Optional[str] = None):
        """加载 TinyBERT 模型"""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        
        # 设置缓存目录
        if cache_dir is None:
            # 默认使用工作区根目录的 cache
            cache_dir = Path(__file__).parent.parent.parent.parent / "cache"
        cache_dir = Path(cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        
        print(f"TinyBERT 日志编码器初始化（在线模式）")
        print(f"  模型: {self.model_name}")
        print(f"  缓存目录: {cache_dir}")
        
        # 尝试离线加载
        try:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir),
                local_files_only=True
            )
            self.bert = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir),
                local_files_only=True
            )
            print(f"  ✓ 离线加载成功")
        except Exception as e:
            print(f"  ⚠️ 离线加载失败，尝试在线下载: {e}")
            os.environ['HF_HUB_OFFLINE'] = '0'
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir)
            )
            self.bert = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir)
            )
            print(f"  ✓ 在线下载成功")
        
        # 冻结参数
        if self.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"  ✓ BERT 参数已冻结")
        
        # 统计参数
        bert_params = sum(p.numel() for p in self.bert.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        print(f"  BERT 参数: {bert_params:,}")
        print(f"  投影层参数: {proj_params:,}")
    
    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        编码文本列表（用于预计算）
        
        Args:
            texts: 日志文本列表
            max_length: 最大序列长度
            batch_size: 批次大小
        
        Returns:
            embeddings: (N, hidden_dim) 嵌入向量
        """
        if self._mode == 'offline':
            raise RuntimeError("在线编码需要加载 BERT 模型")
        
        self.bert.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                device = next(self.bert.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.bert(**inputs)
                
                # 池化
                if self.pooling == 'cls':
                    batch_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                else:
                    # Mean pooling
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    batch_emb = (outputs.last_hidden_state * attention_mask).sum(1)
                    batch_emb = batch_emb / attention_mask.sum(1)
                
                embeddings.append(batch_emb.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def forward(
        self,
        log_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            log_embeddings: 预计算的日志嵌入
                - 离线模式: (B, T, N, hidden_dim) 预计算嵌入
                - 在线模式: 同上（需要先调用 encode_texts）
        
        Returns:
            output: (B, T, N, output_dim) 投影后的嵌入 + PE
        """
        B, T, N, D = log_embeddings.shape
        
        # 投影到模型维度
        # (B, T, N, hidden_dim) -> (B, T, N, output_dim)
        output = self.projection(log_embeddings)
        
        # 添加位置编码
        output = output + self.pe[:, :T, :].unsqueeze(2)  # (1, T, 1, D)
        
        return output


class TemplateLogEncoder(nn.Module):
    """
    原始模板计数编码器（用于对比）
    
    与 V2/V3 的 ModalityEmbedding 相同，保持兼容性。
    """
    
    def __init__(
        self,
        log_dim: int = 256,
        output_dim: int = 64,
        max_len: int = 100
    ):
        super().__init__()
        
        self.linear = nn.Linear(log_dim, output_dim)
        
        # 位置编码
        pe = torch.zeros(1, max_len, output_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, output_dim, 2).float() * 
            (-np.log(10000.0) / output_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, log_dim) 模板计数
        
        Returns:
            output: (B, T, N, output_dim) 嵌入 + PE
        """
        T = x.shape[1]
        out = self.linear(x)
        out = out + self.pe[:, :T, :].unsqueeze(2)
        return out


class ClusterCountLogEncoder(nn.Module):
    """
    语义聚类计数编码器
    
    使用 BERT 嵌入的 K-Means 聚类结果，统计每个聚类的出现次数。
    结合了 BERT 的语义能力和模板计数的频率信息。
    
    输入格式和模板计数相同：(B, T, N, num_clusters)
    
    Args:
        num_clusters: 聚类数量（默认 64）
        output_dim: 输出维度（默认 64）
        max_len: 最大序列长度
    """
    
    def __init__(
        self,
        num_clusters: int = 64,
        output_dim: int = 64,
        max_len: int = 100
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.output_dim = output_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(num_clusters, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # 位置编码
        pe = torch.zeros(1, max_len, output_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, output_dim, 2).float() * 
            (-np.log(10000.0) / output_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        print(f"语义聚类计数编码器初始化")
        print(f"  聚类数量: {num_clusters}")
        print(f"  输出维度: {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, num_clusters) 聚类计数
        
        Returns:
            output: (B, T, N, output_dim) 嵌入 + PE
        """
        T = x.shape[1]
        out = self.projection(x)
        out = out + self.pe[:, :T, :].unsqueeze(2)
        return out


class AttentionPoolingLogEncoder(nn.Module):
    """
    Attention Pooling 日志编码器
    
    使用可学习的 attention 机制聚合每个时间点每个主机的多条日志嵌入，
    而不是简单的 mean pooling。
    
    优点：
    - 模型可以学习哪条日志对异常检测更重要
    - 保留了日志数量信息（通过 mask）
    - 比 mean pooling 更有表达力
    
    Args:
        hidden_dim: BERT 隐藏维度（312）
        output_dim: 输出维度（64）
        num_heads: attention heads 数量
        max_logs: 每主机每时间点最大日志数
        use_count_feature: 是否添加日志数量特征
        log_proj_dim: 日志投影维度（None 表示使用 output_dim）
    """
    
    def __init__(
        self,
        hidden_dim: int = 312,
        output_dim: int = 64,
        num_heads: int = 4,
        max_logs: int = 10,
        max_len: int = 100,
        use_count_feature: bool = False,
        log_proj_dim: int = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_logs = max_logs
        self.use_count_feature = use_count_feature
        
        # 日志投影维度（可以比 output_dim 大，用于保留更多信息）
        self.log_proj_dim = log_proj_dim if log_proj_dim is not None else output_dim
        
        # 可学习的 query 向量（用于 attention pooling）
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.query)
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, self.log_proj_dim),
            nn.LayerNorm(self.log_proj_dim),
            nn.GELU()
        )
        
        # 位置编码（使用 output_dim，因为最终输出需要和其他模态对齐）
        pe = torch.zeros(1, max_len, output_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, output_dim, 2).float() * 
            (-np.log(10000.0) / output_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 如果 log_proj_dim != output_dim，需要额外的对齐层
        if self.log_proj_dim != output_dim:
            self.align_layer = nn.Linear(self.log_proj_dim, output_dim)
        else:
            self.align_layer = None
        
        print(f"Attention Pooling 日志编码器初始化")
        print(f"  BERT 隐藏维度: {hidden_dim}")
        print(f"  日志投影维度: {self.log_proj_dim}")
        print(f"  输出维度: {output_dim}")
        print(f"  Attention heads: {num_heads}")
        print(f"  最大日志数/主机: {max_logs}")
    
    def forward(
        self,
        log_embeddings: torch.Tensor,
        log_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            log_embeddings: (B, T, N, K, D) 每条日志的嵌入
                B: batch size
                T: 时间步数
                N: 主机数
                K: 每主机最大日志数
                D: BERT 隐藏维度
            log_masks: (B, T, N, K) 日志 mask（1=有效，0=padding）
        
        Returns:
            output: (B, T, N, output_dim) 聚合后的嵌入 + PE
        """
        B, T, N, K, D = log_embeddings.shape
        
        # 重塑为 (B*T*N, K, D) 以便批量处理
        log_emb_flat = log_embeddings.view(B * T * N, K, D)
        
        # 处理 mask
        if log_masks is not None:
            # (B, T, N, K) -> (B*T*N, K)
            mask_flat = log_masks.view(B * T * N, K)
            
            # 检测全零 mask（某主机某时间点无日志）
            # 对于这些情况，我们需要特殊处理，否则 attention 会产生 NaN
            all_masked = (mask_flat.sum(dim=-1) == 0)  # (B*T*N,)
            
            # 对于全零 mask，临时设置第一个位置为有效，避免 NaN
            # 后面会用零向量替换这些位置的输出
            mask_flat_safe = mask_flat.clone()
            mask_flat_safe[all_masked, 0] = 1
            
            # 转换为 attention mask（True = 忽略）
            key_padding_mask = (mask_flat_safe == 0)
        else:
            key_padding_mask = None
            all_masked = None
        
        # 扩展 query 到 batch 大小
        query = self.query.expand(B * T * N, -1, -1)  # (B*T*N, 1, D)
        
        # Attention pooling
        # query: (B*T*N, 1, D), key/value: (B*T*N, K, D)
        pooled, attn_weights = self.attention(
            query=query,
            key=log_emb_flat,
            value=log_emb_flat,
            key_padding_mask=key_padding_mask
        )
        # pooled: (B*T*N, 1, D)
        
        pooled = pooled.squeeze(1)  # (B*T*N, D)
        
        # 对于全零 mask 的位置，用零向量替换（表示无日志）
        if all_masked is not None and all_masked.any():
            pooled = pooled.clone()
            pooled[all_masked] = 0.0
        
        # 投影
        if self.use_count_feature:
            # 语义特征
            semantic = self.semantic_projection(pooled)  # (B*T*N, output_dim - 8)
            
            # 数量特征：从 mask 计算每个位置的日志数量
            if log_masks is not None:
                # mask_flat 已经是 (B*T*N, K)
                log_count = mask_flat.sum(dim=-1, keepdim=True).float()  # (B*T*N, 1)
                # 归一化到 [0, 1] 范围
                log_count = log_count / self.max_logs
            else:
                log_count = torch.ones(B * T * N, 1, device=pooled.device)
            
            count_feat = self.count_projection(log_count)  # (B*T*N, 8)
            
            # 拼接语义和数量特征
            output = torch.cat([semantic, count_feat], dim=-1)  # (B*T*N, output_dim)
        else:
            output = self.projection(pooled)  # (B*T*N, output_dim)
        
        # 重塑回 (B, T, N, output_dim)
        output = output.view(B, T, N, self.output_dim)
        
        # 添加位置编码
        output = output + self.pe[:, :T, :].unsqueeze(2)
        
        return output


def create_log_encoder(config) -> nn.Module:
    """
    工厂函数：根据配置创建日志编码器
    
    Args:
        config: V4Config 配置对象
    
    Returns:
        log_encoder: 日志编码器模块
    """
    if config.log_encoder_type == 'bert':
        return TinyBERTLogEncoder(
            model_name=config.bert_model_name,
            hidden_dim=config.bert_hidden_dim,
            output_dim=config.embedding_dim,
            freeze=config.bert_freeze,
            pooling=config.bert_pooling,
            precomputed_path=config.precomputed_log_embeddings
        )
    elif config.log_encoder_type == 'bert_attn':
        return AttentionPoolingLogEncoder(
            hidden_dim=config.bert_hidden_dim,
            output_dim=config.embedding_dim,
            num_heads=config.attn_pool_heads,
            max_logs=config.max_logs_per_host,
            use_count_feature=getattr(config, 'use_count_feature', False)
        )
    elif config.log_encoder_type == 'bert_cluster':
        return ClusterCountLogEncoder(
            num_clusters=getattr(config, 'num_clusters', 64),
            output_dim=config.embedding_dim
        )
    else:
        return TemplateLogEncoder(
            log_dim=config.log_dim,
            output_dim=config.embedding_dim
        )
