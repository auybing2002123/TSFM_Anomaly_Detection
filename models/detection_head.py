"""
Detection Head for Anomaly Detection.

This module contains the detection head components:
- EmbeddingPooler: Pool multi-variate embeddings
- ReconDecoder: Reconstruction branch
- PredPredictor: Prediction branch
- ScoreFusion: Fuse anomaly scores
- DetectionHead: Combined detection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EmbeddingPooler(nn.Module):
    """
    Pool multi-variate embeddings into fixed-size representation.
    
    Supports multiple pooling strategies:
    - mean: Simple mean pooling over features
    - attention: Learnable attention pooling
    - flatten: Flatten and project
    
    Args:
        input_dim: Embedding dimension from backbone (d_model).
        n_features: Number of input features/variates.
        output_dim: Output hidden dimension.
        pooling: Pooling strategy ('mean', 'attention', 'flatten').
    """
    
    def __init__(
        self,
        input_dim: int,
        n_features: int,
        output_dim: int,
        pooling: str = 'attention',
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_features = n_features
        self.output_dim = output_dim
        self.pooling = pooling
        
        if pooling == 'mean':
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif pooling == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif pooling == 'flatten':
            self.proj = nn.Sequential(
                nn.Linear(input_dim * n_features, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool embeddings.
        
        Args:
            x: (batch, n_features, d_model)
            
        Returns:
            pooled: (batch, output_dim)
        """
        if self.pooling == 'mean':
            # Mean over features: (batch, d_model)
            pooled = x.mean(dim=1)
            return self.proj(pooled)
        
        elif self.pooling == 'attention':
            # Self-attention over features
            attn_out, _ = self.attention(x, x, x)
            # Mean over features after attention
            pooled = attn_out.mean(dim=1)
            return self.proj(pooled)
        
        elif self.pooling == 'flatten':
            # Flatten: (batch, n_features * d_model)
            flat = x.flatten(start_dim=1)
            return self.proj(flat)


class ReconDecoder(nn.Module):
    """
    Reconstruction decoder for anomaly detection.
    
    Reconstructs the input window from the pooled embedding.
    
    Args:
        input_dim: Hidden dimension from pooler.
        window_size: Size of input window.
        n_features: Number of output features.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        n_features: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.n_features = n_features
        self.output_size = window_size * n_features
        
        layers = []
        current_dim = input_dim
        
        # Gradually expand to target dimension
        for i in range(num_layers - 1):
            next_dim = min(current_dim * 2, self.output_size)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim
        
        # Final layer to output size
        layers.append(nn.Linear(current_dim, self.output_size))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input window.
        
        Args:
            x: (batch, hidden_dim)
            
        Returns:
            recon: (batch, window_size, n_features)
        """
        out = self.decoder(x)
        return out.view(-1, self.window_size, self.n_features)


class PredPredictor(nn.Module):
    """
    Prediction head for next-step forecasting.
    
    Predicts the next time step from the pooled embedding.
    
    Args:
        input_dim: Hidden dimension from pooler.
        n_features: Number of output features.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_features: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_features = n_features
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, current_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(current_dim, n_features))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict next time step.
        
        Args:
            x: (batch, hidden_dim)
            
        Returns:
            pred: (batch, n_features)
        """
        return self.predictor(x)


class ScoreFusion(nn.Module):
    """
    Fuse reconstruction and prediction anomaly scores.
    
    S = α·S_recon + β·S_pred
    where α, β are learnable parameters (normalized via softmax).
    
    Args:
        learnable: Whether weights are learnable.
        init_alpha: Initial value for alpha.
        init_beta: Initial value for beta.
    """
    
    def __init__(
        self,
        learnable: bool = True,
        init_alpha: float = 0.5,
        init_beta: float = 0.5,
    ):
        super().__init__()
        
        self.learnable = learnable
        
        if learnable:
            # Use raw logits, apply softmax during forward
            self.alpha_logit = nn.Parameter(torch.tensor(init_alpha))
            self.beta_logit = nn.Parameter(torch.tensor(init_beta))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
            self.register_buffer('beta', torch.tensor(init_beta))
    
    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized weights."""
        if self.learnable:
            weights = F.softmax(
                torch.stack([self.alpha_logit, self.beta_logit]),
                dim=0
            )
            return weights[0], weights[1]
        else:
            total = self.alpha + self.beta
            return self.alpha / total, self.beta / total
    
    def forward(
        self,
        recon_score: torch.Tensor,
        pred_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse anomaly scores.
        
        Args:
            recon_score: (batch,) reconstruction error.
            pred_score: (batch,) prediction error.
            
        Returns:
            fused_score: (batch,) combined anomaly score.
        """
        alpha, beta = self.get_weights()
        return alpha * recon_score + beta * pred_score


class DetectionHead(nn.Module):
    """
    Combined detection head with reconstruction and prediction branches.
    
    This module combines:
    - EmbeddingPooler
    - ReconDecoder
    - PredPredictor
    - ScoreFusion
    
    Args:
        d_model: Embedding dimension from backbone.
        n_features: Number of input features.
        window_size: Size of input window.
        hidden_dim: Hidden dimension for detection head.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
        pooling: Pooling strategy.
        fusion_learnable: Whether fusion weights are learnable.
    """
    
    def __init__(
        self,
        d_model: int,
        n_features: int,
        window_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'attention',
        fusion_learnable: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # Pooler
        self.pooler = EmbeddingPooler(
            input_dim=d_model,
            n_features=n_features,
            output_dim=hidden_dim,
            pooling=pooling,
            dropout=dropout,
        )
        
        # Reconstruction branch
        self.recon_decoder = ReconDecoder(
            input_dim=hidden_dim,
            window_size=window_size,
            n_features=n_features,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Prediction branch
        self.pred_predictor = PredPredictor(
            input_dim=hidden_dim,
            n_features=n_features,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Score fusion
        self.score_fusion = ScoreFusion(learnable=fusion_learnable)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        x_input: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through detection head.
        
        Args:
            embeddings: (batch, n_features, d_model) from backbone.
            x_input: (batch, window_size, n_features) original input for loss.
            x_target: (batch, n_features) target for prediction, defaults to x_input[:, -1, :].
            
        Returns:
            dict with:
            - 'pooled': Pooled embeddings
            - 'recon': Reconstructed input
            - 'pred': Predicted next step
            - 'recon_score': Reconstruction anomaly score (if x_input provided)
            - 'pred_score': Prediction anomaly score (if x_input provided)
            - 'anomaly_score': Fused anomaly score (if x_input provided)
        """
        # Pool embeddings
        pooled = self.pooler(embeddings)  # (batch, hidden_dim)
        
        # Detection branches
        recon = self.recon_decoder(pooled)  # (batch, window_size, n_features)
        pred = self.pred_predictor(pooled)  # (batch, n_features)
        
        output = {
            'pooled': pooled,
            'recon': recon,
            'pred': pred,
        }
        
        # Compute anomaly scores if input provided
        if x_input is not None:
            # Reconstruction score: MSE per sample
            recon_score = ((recon - x_input) ** 2).mean(dim=(1, 2))
            
            # Prediction target
            if x_target is None:
                x_target = x_input[:, -1, :]
            
            # Prediction score: MSE per sample
            pred_score = ((pred - x_target) ** 2).mean(dim=1)
            
            # Fused score
            anomaly_score = self.score_fusion(recon_score, pred_score)
            
            output.update({
                'recon_score': recon_score,
                'pred_score': pred_score,
                'anomaly_score': anomaly_score,
            })
        
        return output
