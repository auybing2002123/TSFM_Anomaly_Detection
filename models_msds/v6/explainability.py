"""
V6 可解释性模块：根因定位 + 多层次归因 + 预测偏差可视化

完全独立的模块，不修改任何已有文件。
只读取训练好的 V6 checkpoint，对异常样本进行解释。

三层解释：
  1. 根因定位：按异常概率排序（不需要因果矩阵）
  2. 模态归因：Integrated Gradients 分解三模态贡献
  3. 特征归因：具体哪个指标/日志模板/调用边异常

V6 特有：预测偏差可视化（GPT-2 预测 vs 实际值）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RootCauseResult:
    """根因定位结果"""
    # 服务排名（按异常概率降序）
    ranking: List[Dict]  # [{'service': str, 'idx': int, 'score': float, 'is_anomaly': bool}, ...]
    # 异常服务列表
    anomaly_services: List[str]
    # 根因服务（异常概率最高的）
    root_cause: Optional[str]
    root_cause_idx: Optional[int]
    root_cause_score: float


@dataclass
class ModalityAttribution:
    """模态归因结果"""
    # 各模态贡献比例
    metrics_pct: float
    logs_pct: float
    traces_pct: float
    # 原始归因值
    metrics_raw: float
    logs_raw: float
    traces_raw: float


@dataclass
class FeatureAttribution:
    """特征级归因"""
    top_metrics: List[Dict]   # [{'service': str, 'feature': str, 'contribution': float}, ...]
    top_logs: List[Dict]      # [{'service': str, 'template_id': int, 'contribution': float}, ...]
    top_traces: List[Dict]    # [{'source': str, 'target': str, 'contribution': float}, ...]


@dataclass
class DeviationInfo:
    """预测偏差信息（V6 特有）"""
    # 每个服务的偏差 L2 范数（最后一步）
    deviation_norms: List[float]  # (N,)
    # 偏差最大的服务
    max_deviation_service: str
    max_deviation_idx: int


@dataclass
class TemporalAnalysis:
    """
    时序分析结果（V6 特有）
    
    利用 GPT-2 逐步预测的特性，计算每个时间步的预测偏差。
    偏差突然飙升的时间步 = 异常开始时间。
    """
    # 每个时间步的偏差 (根因服务)
    deviation_per_step: List[float]  # (T-1,) t=1..T-1 的偏差
    # 异常开始时间（偏差开始飙升的时间步）
    anomaly_onset: Optional[int]
    # 偏差峰值时间
    peak_step: int
    # ASCII 可视化
    pattern: str
    # IG 时间归因（可选）
    ig_temporal: Optional[List[float]]  # (T,) 每个时间步的 IG 贡献


@dataclass
class ShapleyModalityAttribution:
    """Shapley 模态归因结果（精确值，3 player）"""
    metrics_pct: float
    logs_pct: float
    traces_pct: float
    # 原始 Shapley 值
    metrics_shapley: float
    logs_shapley: float
    traces_shapley: float
    # 值函数表（用于分析交互效应）
    coalition_values: Optional[Dict] = None  # {'empty': v, 'M': v, 'L': v, ...}
    # Shapley Interaction Values（模态间交互效应）
    interaction_values: Optional[Dict] = None  # {'M×L': float, 'M×T': float, 'L×T': float}


@dataclass
class ExplanationResult:
    """完整的解释结果"""
    sample_idx: int
    root_cause: RootCauseResult
    modality_attribution: Optional[ModalityAttribution]
    feature_attribution: Optional[FeatureAttribution]
    deviation_info: DeviationInfo
    temporal_analysis: Optional[TemporalAnalysis]
    anomaly_scores: np.ndarray  # (N,)
    shapley_modality: Optional[ShapleyModalityAttribution] = None


# ============================================================
# 根因定位器（纯推理，不需要因果矩阵）
# ============================================================

class V6RootCauseLocator:
    """
    V6 根因定位器
    
    核心逻辑：直接按异常概率排序。
    V6 没有因果模块，不需要因果推理。
    """
    
    def __init__(self, service_names: List[str], threshold: float = 0.5):
        self.service_names = service_names
        self.threshold = threshold
    
    def locate(self, anomaly_scores: np.ndarray) -> RootCauseResult:
        """
        根因定位
        
        Args:
            anomaly_scores: (N,) 每个服务的异常概率
        
        Returns:
            RootCauseResult
        """
        N = len(anomaly_scores)
        
        # 按异常概率降序排序
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        
        ranking = []
        anomaly_services = []
        
        for idx in sorted_indices:
            score = float(anomaly_scores[idx])
            name = self.service_names[idx] if idx < len(self.service_names) else f'service_{idx}'
            is_anomaly = score > self.threshold
            
            ranking.append({
                'service': name,
                'idx': int(idx),
                'score': score,
                'is_anomaly': is_anomaly
            })
            
            if is_anomaly:
                anomaly_services.append(name)
        
        # 根因 = 异常概率最高的服务
        root_cause = anomaly_services[0] if anomaly_services else None
        root_cause_idx = int(sorted_indices[0]) if anomaly_services else None
        root_cause_score = float(anomaly_scores[sorted_indices[0]]) if anomaly_services else 0.0
        
        return RootCauseResult(
            ranking=ranking,
            anomaly_services=anomaly_services,
            root_cause=root_cause,
            root_cause_idx=root_cause_idx,
            root_cause_score=root_cause_score
        )
    
    def batch_locate(self, anomaly_scores: np.ndarray) -> List[RootCauseResult]:
        """批量根因定位"""
        B = anomaly_scores.shape[0]
        return [self.locate(anomaly_scores[i]) for i in range(B)]


# ============================================================
# V6 Integrated Gradients（适配 V6 的 forward 接口）
# ============================================================

class V6IntegratedGradients:
    """
    V6 专用 Integrated Gradients
    
    与旧版 IG 的区别：
    - 适配 V6 的 forward 接口（evaluate=True 返回 cls_probs）
    - 不依赖因果矩阵
    - 支持提取预测偏差信息
    """
    
    def __init__(self, model: nn.Module, steps: int = 50, device: str = 'cuda'):
        self.model = model
        self.steps = steps
        self.device = device
    
    def _compute_gradients(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_service: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算梯度"""
        data_node = data_node.clone().requires_grad_(True)
        data_log = data_log.clone().requires_grad_(True)
        data_edge = data_edge.clone().requires_grad_(True)
        
        B, T, N = data_node.shape[:3]
        dummy_label = torch.zeros(B, N, 3, device=self.device)
        dummy_label[:, :, 0] = 1
        
        self.model.eval()
        cls_probs, _ = self.model(
            data_node, data_log, data_edge,
            dummy_label, evaluate=True
        )
        
        anomaly_score = cls_probs[:, :, 1]  # (B, N)
        
        if target_service is not None:
            target = anomaly_score[:, target_service].sum()
        else:
            target = anomaly_score.sum()
        
        target.backward()
        
        return (
            data_node.grad.clone(),
            data_log.grad.clone(),
            data_edge.grad.clone(),
            anomaly_score.detach()
        )
    
    def attribute(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_service: Optional[int] = None
    ) -> Dict:
        """
        计算 Integrated Gradients
        
        Returns:
            {
                'ig_metric': (B, T, N, metric_dim),
                'ig_log': (B, T, N, log_dim),
                'ig_trace': (B, T, N, N, trace_dim),
                'anomaly_scores': (B, N)
            }
        """
        data_node = data_node.to(self.device)
        data_log = data_log.to(self.device)
        data_edge = data_edge.to(self.device)
        
        # Baseline = 零
        baseline_node = torch.zeros_like(data_node)
        baseline_log = torch.zeros_like(data_log)
        baseline_edge = torch.zeros_like(data_edge)
        
        # 累积梯度
        acc_node = torch.zeros_like(data_node)
        acc_log = torch.zeros_like(data_log)
        acc_edge = torch.zeros_like(data_edge)
        
        for step in range(self.steps + 1):
            alpha = step / self.steps
            
            interp_node = baseline_node + alpha * (data_node - baseline_node)
            interp_log = baseline_log + alpha * (data_log - baseline_log)
            interp_edge = baseline_edge + alpha * (data_edge - baseline_edge)
            
            with torch.enable_grad():
                g_node, g_log, g_edge, _ = self._compute_gradients(
                    interp_node, interp_log, interp_edge, target_service
                )
            
            acc_node += g_node
            acc_log += g_log
            acc_edge += g_edge
        
        # IG = (input - baseline) × avg_gradient
        avg_node = acc_node / (self.steps + 1)
        avg_log = acc_log / (self.steps + 1)
        avg_edge = acc_edge / (self.steps + 1)
        
        ig_node = (data_node - baseline_node) * avg_node
        ig_log = (data_log - baseline_log) * avg_log
        ig_edge = (data_edge - baseline_edge) * avg_edge
        
        # 最终异常分数
        with torch.enable_grad():
            _, _, _, anomaly_scores = self._compute_gradients(
                data_node, data_log, data_edge, target_service
            )
        
        return {
            'ig_metric': ig_node.detach(),
            'ig_log': ig_log.detach(),
            'ig_trace': ig_edge.detach(),
            'anomaly_scores': anomaly_scores.detach()
        }


# ============================================================
# V6 预测偏差提取器
# ============================================================

class V6DeviationExtractor:
    """
    提取 V6 的预测偏差信息
    
    V6 的异常检测核心：GPT-2 预测下一步 → 和实际值比较 → 偏差大就是异常。
    
    两种偏差：
    1. 最后一步偏差 (N,)：每个服务的异常程度
    2. 逐步偏差 (N, T-1)：每个时间步的偏差变化，用于定位异常开始时间
    """
    
    def __init__(self, model: nn.Module, service_names: List[str], device: str = 'cuda'):
        self.model = model
        self.service_names = service_names
        self.device = device
    
    @torch.no_grad()
    def extract(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor
    ) -> Tuple[List[DeviationInfo], List[TemporalAnalysis]]:
        """
        提取预测偏差（包含逐步偏差）
        
        Returns:
            deviation_infos: 每个样本的偏差信息
            temporal_analyses: 每个样本的时序分析（针对偏差最大的服务）
        """
        self.model.eval()
        data_node = data_node.to(self.device)
        data_log = data_log.to(self.device)
        data_edge = data_edge.to(self.device)
        
        B, T, N = data_node.shape[:3]
        config = self.model.config
        
        # Step 1: 模态编码
        metric_feat = self.model.metric_encoder(data_node)
        log_feat = self.model.log_encoder(data_log)
        trace_feat = self.model.trace_encoder(data_edge, self.model.adj)
        
        # Step 2: 融合
        fused = torch.cat([metric_feat, log_feat, trace_feat], dim=-1)
        fused = self.model.fusion_proj(fused)
        
        # Step 3: GPT-2
        gpt_input = fused.permute(0, 2, 1, 3).reshape(B * N, T, config.gpt2_dim)
        gpt_output = self.model.gpt2(inputs_embeds=gpt_input).last_hidden_state
        # gpt_output: (B*N, T, 768)
        
        # Step 4a: 最后一步偏差（与 forward 一致）
        pred_last = self.model.pred_head(gpt_output[:, -2, :])
        actual_last = gpt_input[:, -1, :]
        deviation_last = pred_last - actual_last
        deviation_last = deviation_last.reshape(B, N, config.gpt2_dim)
        deviation_norms = torch.norm(deviation_last, dim=-1).cpu().numpy()  # (B, N)
        
        # Step 4b: 逐步偏差
        # GPT-2 在位置 t 的输出预测位置 t+1 的输入
        # 所以 deviation[t] = pred_head(gpt_output[:, t, :]) - gpt_input[:, t+1, :]
        # 有效范围: t = 0, 1, ..., T-2 (共 T-1 步)
        per_step_deviations = []
        for t in range(T - 1):
            pred_t = self.model.pred_head(gpt_output[:, t, :])  # (B*N, 768)
            actual_t1 = gpt_input[:, t + 1, :]  # (B*N, 768)
            dev_t = torch.norm(pred_t - actual_t1, dim=-1)  # (B*N,)
            dev_t = dev_t.reshape(B, N)  # (B, N)
            per_step_deviations.append(dev_t.cpu().numpy())
        
        # per_step_deviations: list of (B, N), length T-1
        per_step_arr = np.stack(per_step_deviations, axis=1)  # (B, T-1, N)
        
        # 构建结果
        deviation_infos = []
        temporal_analyses = []
        
        for b in range(B):
            norms = deviation_norms[b]
            max_idx = int(np.argmax(norms))
            max_name = self.service_names[max_idx] if max_idx < len(self.service_names) else f'service_{max_idx}'
            
            deviation_infos.append(DeviationInfo(
                deviation_norms=norms.tolist(),
                max_deviation_service=max_name,
                max_deviation_idx=max_idx
            ))
            
            # 时序分析：针对偏差最大的服务
            dev_series = per_step_arr[b, :, max_idx]  # (T-1,)
            temporal_analyses.append(self._analyze_temporal(dev_series))
        
        return deviation_infos, temporal_analyses
    
    def _analyze_temporal(self, dev_series: np.ndarray) -> TemporalAnalysis:
        """分析单个服务的偏差时序"""
        T_minus_1 = len(dev_series)
        
        # 异常开始时间：偏差超过 均值+标准差 的第一个时间步
        mean_dev = dev_series.mean()
        std_dev = dev_series.std()
        threshold = mean_dev + std_dev
        
        anomaly_onset = None
        for t in range(T_minus_1):
            if dev_series[t] > threshold:
                anomaly_onset = t + 1  # +1 因为 dev_series[0] 对应 t=0→t=1 的预测
                break
        
        # 峰值
        peak_step = int(np.argmax(dev_series)) + 1
        
        # ASCII 可视化
        pattern = self._make_ascii_pattern(dev_series)
        
        return TemporalAnalysis(
            deviation_per_step=dev_series.tolist(),
            anomaly_onset=anomaly_onset,
            peak_step=peak_step,
            pattern=pattern,
            ig_temporal=None  # 后续由 IG 填充
        )
    
    @staticmethod
    def _make_ascii_pattern(series: np.ndarray) -> str:
        """生成 ASCII 时序可视化"""
        if series.max() == series.min():
            return '[' + '▄' * len(series) + ']'
        normalized = (series - series.min()) / (series.max() - series.min())
        chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        levels = np.clip((normalized * 7).astype(int), 0, 7)
        return '[' + ''.join(chars[l] for l in levels) + ']'


# ============================================================
# Shapley 模态归因（精确计算，3 player = 8 子集）
# ============================================================

class V6ShapleyModality:
    """
    精确 Shapley 值计算模态归因（3 模态 = 3 player）
    
    对 2^3 = 8 个子集各做一次前向传播，计算值函数后
    用 Shapley 公式精确分配。缺失模态用零向量替代。
    
    优势：能揭示模态间交互效应（如 M+L 远大于 M 和 L 之和）
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def attribute(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        target_service: int
    ) -> ShapleyModalityAttribution:
        """
        计算 3 模态的精确 Shapley 值
        
        Args:
            data_node: (1, T, N, metric_dim)
            data_log: (1, T, N, log_dim)
            data_edge: (1, T, N, N, trace_dim)
            target_service: 根因服务索引
        """
        self.model.eval()
        
        # 零基线
        zero_node = torch.zeros_like(data_node)
        zero_log = torch.zeros_like(data_log)
        zero_edge = torch.zeros_like(data_edge)
        
        # 枚举 8 个子集，计算值函数 v(S) = P(异常 | 根因服务)
        # 编码: bit0=Metrics, bit1=Logs, bit2=Traces
        coalition_values = {}
        labels = ['empty', 'M', 'L', 'ML', 'T', 'MT', 'LT', 'MLT']
        
        with torch.no_grad():
            B, T, N = data_node.shape[:3]
            dummy_label = torch.zeros(1, N, 3, device=self.device)
            dummy_label[:, :, 0] = 1
            
            for mask in range(8):
                node_in = data_node if (mask & 1) else zero_node
                log_in = data_log if (mask & 2) else zero_log
                edge_in = data_edge if (mask & 4) else zero_edge
                
                cls_probs, _ = self.model(
                    node_in.to(self.device),
                    log_in.to(self.device),
                    edge_in.to(self.device),
                    dummy_label, evaluate=True
                )
                # 值函数 = 根因服务的异常概率
                v = cls_probs[0, target_service, 1].item()
                coalition_values[labels[mask]] = v
        
        # Shapley 公式: φ_i = Σ_{S⊆N\{i}} |S|!(n-|S|-1)!/n! * [v(S∪{i}) - v(S)]
        # n=3, 手动展开
        import math
        n = 3
        
        def shapley_value(player_bit, player_label):
            """计算单个 player 的 Shapley 值"""
            phi = 0.0
            # 遍历不含 player 的所有子集
            other_bits = [b for b in range(3) if b != player_bit]
            for subset_mask_of_others in range(4):  # 2^2 = 4
                # 构建不含 player 的子集 S
                s_mask = 0
                s_size = 0
                for i, bit in enumerate(other_bits):
                    if subset_mask_of_others & (1 << i):
                        s_mask |= (1 << bit)
                        s_size += 1
                
                # v(S ∪ {player}) - v(S)
                s_with_player = s_mask | (1 << player_bit)
                marginal = coalition_values[labels[s_with_player]] - coalition_values[labels[s_mask]]
                
                # 权重: |S|!(n-|S|-1)!/n!
                weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / math.factorial(n)
                phi += weight * marginal
            
            return phi
        
        phi_m = shapley_value(0, 'M')   # Metrics = bit 0
        phi_l = shapley_value(1, 'L')   # Logs = bit 1
        phi_t = shapley_value(2, 'T')   # Traces = bit 2
        
        # 归一化为百分比
        total = abs(phi_m) + abs(phi_l) + abs(phi_t) + 1e-8
        
        # Shapley Interaction Values (Grabisch 1999)
        # I(i,j) = Σ_{S⊆N\{i,j}} |S|!(n-|S|-2)!/(n-1)! × [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]
        # 对 3 player, N\{i,j} 只有 1 个元素，所以只有 2 个子集
        cv = coalition_values
        player_bits = {'M': 0, 'L': 1, 'T': 2}
        interaction_values = {}
        for (p1, p2) in [('M', 'L'), ('M', 'T'), ('L', 'T')]:
            b1, b2 = player_bits[p1], player_bits[p2]
            other_bit = [b for b in range(3) if b != b1 and b != b2][0]
            I_val = 0.0
            for include_other in [False, True]:
                s_mask = (1 << other_bit) if include_other else 0
                s_size = 1 if include_other else 0
                s_ij = s_mask | (1 << b1) | (1 << b2)
                s_i = s_mask | (1 << b1)
                s_j = s_mask | (1 << b2)
                delta = cv[labels[s_ij]] - cv[labels[s_i]] - cv[labels[s_j]] + cv[labels[s_mask]]
                weight = math.factorial(s_size) * math.factorial(n - s_size - 2) / math.factorial(n - 1)
                I_val += weight * delta
            interaction_values[f'{p1}×{p2}'] = I_val
        
        return ShapleyModalityAttribution(
            metrics_pct=abs(phi_m) / total * 100,
            logs_pct=abs(phi_l) / total * 100,
            traces_pct=abs(phi_t) / total * 100,
            metrics_shapley=phi_m,
            logs_shapley=phi_l,
            traces_shapley=phi_t,
            coalition_values=coalition_values,
            interaction_values=interaction_values
        )


# ============================================================
# V6 解释器（整合所有组件）
# ============================================================

class V6Explainer:
    """
    V6 模型解释器
    
    整合根因定位、模态归因、特征归因、预测偏差。
    """
    
    def __init__(
        self,
        model: nn.Module,
        service_names: List[str],
        metric_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        ig_steps: int = 50,
        device: str = 'cuda'
    ):
        self.model = model
        self.service_names = service_names
        self.metric_names = metric_names or [f'metric_{i}' for i in range(model.config.metric_dim)]
        self.device = device
        
        self.locator = V6RootCauseLocator(service_names, threshold)
        self.ig = V6IntegratedGradients(model, steps=ig_steps, device=device)
        self.deviation_extractor = V6DeviationExtractor(model, service_names, device)
        self.shapley = V6ShapleyModality(model, device=device)
    
    def explain_sample(
        self,
        data_node: torch.Tensor,
        data_log: torch.Tensor,
        data_edge: torch.Tensor,
        sample_idx: int = 0,
        compute_ig: bool = True,
        compute_shapley: bool = True,
        top_k: int = 5
    ) -> ExplanationResult:
        """
        解释单个样本

        Args:
            data_node: (1, T, N, metric_dim)
            data_log: (1, T, N, log_dim)
            data_edge: (1, T, N, N, trace_dim)
            sample_idx: 样本编号（用于标识）
            compute_ig: 是否计算 IG（较慢，可跳过）
            compute_shapley: 是否计算 Shapley 模态归因（8 次前向，较快）
            top_k: 返回 Top-K 特征
        """
        # 1. 获取异常分数
        self.model.eval()
        with torch.no_grad():
            B, T, N = data_node.shape[:3]
            dummy_label = torch.zeros(1, N, 3, device=self.device)
            dummy_label[:, :, 0] = 1

            cls_probs, _ = self.model(
                data_node.to(self.device),
                data_log.to(self.device),
                data_edge.to(self.device),
                dummy_label, evaluate=True
            )
            anomaly_scores = cls_probs[0, :, 1].cpu().numpy()  # (N,)

        # 2. 根因定位
        root_cause = self.locator.locate(anomaly_scores)

        # 3. 预测偏差 + 时序分析
        deviation_infos, temporal_analyses = self.deviation_extractor.extract(
            data_node, data_log, data_edge
        )
        deviation_info = deviation_infos[0]
        temporal_analysis = temporal_analyses[0]

        # 4. IG 归因（可选）
        modality_attr = None
        feature_attr = None

        if compute_ig and root_cause.root_cause_idx is not None:
            ig_result = self.ig.attribute(
                data_node, data_log, data_edge,
                target_service=root_cause.root_cause_idx
            )
            modality_attr = self._compute_modality_attribution(ig_result)
            feature_attr = self._compute_feature_attribution(ig_result, top_k)

            # IG 时间归因：聚合每个时间步的 IG 贡献
            target = root_cause.root_cause_idx
            ig_m = ig_result['ig_metric'].cpu().numpy()   # (1, T, N, metric_dim)
            ig_l = ig_result['ig_log'].cpu().numpy()       # (1, T, N, log_dim)
            ig_t = ig_result['ig_trace'].cpu().numpy()     # (1, T, N, N, trace_dim)

            ig_per_step = []
            for t in range(T):
                # 根因服务在时间步 t 的总 IG 贡献
                m = np.abs(ig_m[0, t, target, :]).sum()
                l = np.abs(ig_l[0, t, target, :]).sum()
                tr = np.abs(ig_t[0, t, target, :, :]).sum() + \
                     np.abs(ig_t[0, t, :, target, :]).sum()
                ig_per_step.append(float(m + l + tr))

            temporal_analysis.ig_temporal = ig_per_step

        # 5. Shapley 模态归因（可选，8 次前向传播）
        shapley_attr = None
        if compute_shapley and root_cause.root_cause_idx is not None:
            shapley_attr = self.shapley.attribute(
                data_node, data_log, data_edge,
                target_service=root_cause.root_cause_idx
            )

        return ExplanationResult(
            sample_idx=sample_idx,
            root_cause=root_cause,
            modality_attribution=modality_attr,
            feature_attribution=feature_attr,
            deviation_info=deviation_info,
            temporal_analysis=temporal_analysis,
            anomaly_scores=anomaly_scores,
            shapley_modality=shapley_attr
        )

    
    def _compute_modality_attribution(self, ig_result: Dict) -> ModalityAttribution:
        """从 IG 结果计算模态归因"""
        m_raw = ig_result['ig_metric'].abs().sum().item()
        l_raw = ig_result['ig_log'].abs().sum().item()
        t_raw = ig_result['ig_trace'].abs().sum().item()
        total = m_raw + l_raw + t_raw + 1e-8
        
        return ModalityAttribution(
            metrics_pct=m_raw / total * 100,
            logs_pct=l_raw / total * 100,
            traces_pct=t_raw / total * 100,
            metrics_raw=m_raw,
            logs_raw=l_raw,
            traces_raw=t_raw
        )
    
    def _compute_feature_attribution(self, ig_result: Dict, top_k: int) -> FeatureAttribution:
        """从 IG 结果计算特征级归因"""
        # Metrics: (1, T, N, metric_dim) → 对 batch 和时间求和 → (N, metric_dim)
        metric_imp = ig_result['ig_metric'][0].abs().sum(dim=0).cpu().numpy()
        N, D_m = metric_imp.shape
        
        flat_m = metric_imp.flatten()
        top_m_idx = np.argsort(flat_m)[::-1][:top_k]
        
        top_metrics = []
        total_m = flat_m.sum() + 1e-8
        for idx in top_m_idx:
            s_idx, f_idx = divmod(int(idx), D_m)
            s_name = self.service_names[s_idx] if s_idx < len(self.service_names) else f'service_{s_idx}'
            f_name = self.metric_names[f_idx] if f_idx < len(self.metric_names) else f'metric_{f_idx}'
            top_metrics.append({
                'service': s_name,
                'feature': f_name,
                'contribution': float(flat_m[idx] / total_m * 100)
            })
        
        # Logs: (1, T, N, log_dim) → (N, log_dim)
        log_imp = ig_result['ig_log'][0].abs().sum(dim=0).cpu().numpy()
        N, D_l = log_imp.shape
        
        flat_l = log_imp.flatten()
        top_l_idx = np.argsort(flat_l)[::-1][:top_k]
        
        top_logs = []
        total_l = flat_l.sum() + 1e-8
        for idx in top_l_idx:
            s_idx, t_idx = divmod(int(idx), D_l)
            s_name = self.service_names[s_idx] if s_idx < len(self.service_names) else f'service_{s_idx}'
            top_logs.append({
                'service': s_name,
                'template_id': int(t_idx),
                'contribution': float(flat_l[idx] / total_l * 100)
            })
        
        # Traces: (1, T, N, N, trace_dim) → 对 batch, 时间, trace_dim 求和 → (N, N)
        trace_imp = ig_result['ig_trace'][0].abs().sum(dim=(0, 3)).cpu().numpy()
        N2 = trace_imp.shape[0]
        
        flat_t = trace_imp.flatten()
        top_t_idx = np.argsort(flat_t)[::-1][:top_k]
        
        top_traces = []
        total_t = flat_t.sum() + 1e-8
        for idx in top_t_idx:
            src, dst = divmod(int(idx), N2)
            src_name = self.service_names[src] if src < len(self.service_names) else f'service_{src}'
            dst_name = self.service_names[dst] if dst < len(self.service_names) else f'service_{dst}'
            if flat_t[idx] > 0:
                top_traces.append({
                    'source': src_name,
                    'target': dst_name,
                    'contribution': float(flat_t[idx] / total_t * 100)
                })
        
        return FeatureAttribution(
            top_metrics=top_metrics,
            top_logs=top_logs,
            top_traces=top_traces
        )


# ============================================================
# 报告生成器
# ============================================================

def generate_report(exp: ExplanationResult) -> str:
    """生成人类可读的解释报告"""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  V6 异常检测解释报告  (样本 #{exp.sample_idx})")
    lines.append("=" * 65)
    lines.append("")
    
    # 检测结果
    rc = exp.root_cause
    if rc.anomaly_services:
        lines.append(f"检测结果: 🔴 异常 ({len(rc.anomaly_services)} 个服务)")
    else:
        lines.append(f"检测结果: 🟢 正常 (最高分: {max(exp.anomaly_scores):.4f})")
    lines.append("")
    
    # 根因定位
    lines.append("-" * 65)
    lines.append("【根因定位】(按异常概率排序)")
    for i, r in enumerate(rc.ranking[:10]):  # 最多显示 10 个
        icon = "🔴" if r['is_anomaly'] else "🟢"
        marker = " ← 根因" if i == 0 and r['is_anomaly'] else ""
        lines.append(f"  {i+1:2d}. {icon} {r['service']:20s}  P(异常)={r['score']:.4f}{marker}")
    if len(rc.ranking) > 10:
        lines.append(f"  ... (共 {len(rc.ranking)} 个服务)")
    lines.append("")
    
    # 预测偏差
    lines.append("-" * 65)
    lines.append("【预测偏差】(GPT-2 预测 vs 实际值)")
    dev = exp.deviation_info
    norms = dev.deviation_norms
    sorted_dev = sorted(enumerate(norms), key=lambda x: x[1], reverse=True)
    for i, (idx, norm) in enumerate(sorted_dev[:5]):
        name = exp.root_cause.ranking[0]['service'] if idx == dev.max_deviation_idx else ''
        svc = rc.ranking[0]['service']  # 用 ranking 里的名字
        # 找到对应的服务名
        for r in rc.ranking:
            if r['idx'] == idx:
                svc = r['service']
                break
        marker = " ← 偏差最大" if idx == dev.max_deviation_idx else ""
        lines.append(f"  {svc:20s}  偏差={norm:.4f}{marker}")
    lines.append("")
    
    # 模态归因
    if exp.modality_attribution:
        lines.append("-" * 65)
        lines.append("【模态归因】(对根因服务的 IG 分析)")
        ma = exp.modality_attribution
        
        def bar(pct):
            n = int(pct / 5)
            return "█" * n + "░" * (20 - n)
        
        lines.append(f"  Metrics : [{bar(ma.metrics_pct)}] {ma.metrics_pct:.1f}%")
        lines.append(f"  Logs    : [{bar(ma.logs_pct)}] {ma.logs_pct:.1f}%")
        lines.append(f"  Traces  : [{bar(ma.traces_pct)}] {ma.traces_pct:.1f}%")
        lines.append("")
    
    # Shapley 模态归因
    if exp.shapley_modality:
        lines.append("-" * 65)
        lines.append("【模态归因 - Shapley】(精确 Shapley 值，揭示交互效应)")
        sm = exp.shapley_modality
        
        def bar_s(pct):
            n = int(pct / 5)
            return "█" * n + "░" * (20 - n)
        
        lines.append(f"  Metrics : [{bar_s(sm.metrics_pct)}] {sm.metrics_pct:.1f}%  (φ={sm.metrics_shapley:.4f})")
        lines.append(f"  Logs    : [{bar_s(sm.logs_pct)}] {sm.logs_pct:.1f}%  (φ={sm.logs_shapley:.4f})")
        lines.append(f"  Traces  : [{bar_s(sm.traces_pct)}] {sm.traces_pct:.1f}%  (φ={sm.traces_shapley:.4f})")
        
        # 显示交互效应（如果值函数表可用）
        if sm.coalition_values:
            cv = sm.coalition_values
            lines.append("")
            lines.append("  值函数表 (交互效应分析):")
            lines.append(f"    v(∅)={cv.get('empty',0):.4f}  v(M)={cv.get('M',0):.4f}  v(L)={cv.get('L',0):.4f}  v(T)={cv.get('T',0):.4f}")
            lines.append(f"    v(M,L)={cv.get('ML',0):.4f}  v(M,T)={cv.get('MT',0):.4f}  v(L,T)={cv.get('LT',0):.4f}  v(M,L,T)={cv.get('MLT',0):.4f}")
            
            # 检测交互效应: v(M,L) vs v(M)+v(L)
            interaction_ml = cv.get('ML', 0) - cv.get('M', 0) - cv.get('L', 0) + cv.get('empty', 0)
            if abs(interaction_ml) > 0.1:
                lines.append(f"    ⚡ Metrics×Logs 交互效应: {interaction_ml:.4f} ({'协同' if interaction_ml > 0 else '冗余'})")
        
        # Shapley Interaction Values（交互矩阵）
        if sm.interaction_values:
            iv = sm.interaction_values
            lines.append("")
            lines.append("  交互矩阵 (Shapley Interaction Index):")
            lines.append(f"    M×L = {iv.get('M×L', 0):+.4f}  {'⚡协同' if iv.get('M×L', 0) > 0.1 else ''}")
            lines.append(f"    M×T = {iv.get('M×T', 0):+.4f}")
            lines.append(f"    L×T = {iv.get('L×T', 0):+.4f}")
        lines.append("")
    
    # 特征归因
    if exp.feature_attribution:
        lines.append("-" * 65)
        lines.append("【特征归因】(Top 5)")
        fa = exp.feature_attribution
        
        lines.append("  Metrics:")
        for i, f in enumerate(fa.top_metrics[:5], 1):
            lines.append(f"    {i}. {f['service']}/{f['feature']}: {f['contribution']:.1f}%")
        
        lines.append("  Logs:")
        for i, f in enumerate(fa.top_logs[:5], 1):
            lines.append(f"    {i}. {f['service']}/template_{f['template_id']}: {f['contribution']:.1f}%")
        
        lines.append("  Traces:")
        for i, f in enumerate(fa.top_traces[:5], 1):
            lines.append(f"    {i}. {f['source']} → {f['target']}: {f['contribution']:.1f}%")
        lines.append("")
    
    # 时序分析
    if exp.temporal_analysis:
        lines.append("-" * 65)
        lines.append("【时序分析】(GPT-2 逐步预测偏差 → 异常发生时间)")
        ta = exp.temporal_analysis
        
        # 异常起始和峰值
        if ta.anomaly_onset is not None:
            lines.append(f"  异常起始: 第 {ta.anomaly_onset} 步 (偏差超过 μ+σ)")
        else:
            lines.append(f"  异常起始: 未检测到明显突变")
        lines.append(f"  偏差峰值: 第 {ta.peak_step} 步")
        
        total_steps = len(ta.deviation_per_step)
        lines.append(f"  时间跨度: {total_steps} 步")
        lines.append("")
        
        # ASCII 偏差曲线
        lines.append(f"  偏差曲线 ({exp.deviation_info.max_deviation_service}):")
        lines.append(f"  {ta.pattern}")
        lines.append("")
        
        # IG 时间归因（如果有）
        if ta.ig_temporal:
            ig = ta.ig_temporal
            total_ig = sum(ig)
            if total_ig > 0:
                lines.append("  IG 时间归因 (各时间步对异常判定的贡献):")
                # 找 top-5 时间步
                ig_sorted = sorted(enumerate(ig), key=lambda x: x[1], reverse=True)
                for rank, (t, val) in enumerate(ig_sorted[:5], 1):
                    pct = val / total_ig * 100
                    lines.append(f"    {rank}. 第 {t} 步: {pct:.1f}%")
                lines.append("")
    
    lines.append("=" * 65)
    return "\n".join(lines)
