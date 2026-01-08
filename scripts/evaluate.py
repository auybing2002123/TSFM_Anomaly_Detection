#!/usr/bin/env python
"""
Evaluation script for TSFM-AD model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset SMD
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import AnomalyDataset
from data.data_loader import TSFMADDataLoader
from models.tsfm_ad import TSFMADModel
from utils.metrics import compute_metrics, point_adjusted_f1, best_threshold_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TSFMADEvaluator:
    """Evaluator for TSFM-AD model."""
    
    def __init__(
        self,
        model: TSFMADModel,
        test_loader: DataLoader,
        device: str = 'cuda',
        point_adjust: bool = True,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.point_adjust = point_adjust
    
    @torch.no_grad()
    def get_anomaly_scores(self) -> tuple:
        """
        Compute anomaly scores for all test samples.
        
        Returns:
            scores: Array of anomaly scores
            labels: Array of ground truth labels
        """
        all_scores = []
        all_labels = []
        
        for batch in tqdm(self.test_loader, desc="Computing scores"):
            x = batch['data'].to(self.device)
            labels = batch['label'].numpy()
            
            # Get anomaly scores
            scores = self.model.get_anomaly_scores(x)
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels)
        
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        return scores, labels
    
    def evaluate(
        self, 
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Run full evaluation.
        
        Args:
            threshold: Anomaly detection threshold.
                      - 如果提供：使用该阈值（推荐从 checkpoint 获取）
                      - 如果为 None：在测试集上搜索最佳阈值
                      
        Note:
            在测试集上搜索阈值在学术界有争议，但很多 baseline 论文这样做。
            建议主要关注 AUC-ROC 和 AUC-PR（阈值无关指标）。
        
        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing anomaly scores...")
        scores, labels = self.get_anomaly_scores()
        
        logger.info("Computing metrics...")
        
        # 确定阈值来源
        if threshold is not None:
            threshold_source = 'checkpoint'
        else:
            threshold_source = 'test_set_search'
            logger.info(
                "No threshold provided, searching on test set. "
                "Note: Many baseline papers do this, but it's controversial. "
                "Consider using AUC-ROC and AUC-PR as primary metrics."
            )
        
        metrics = compute_metrics(
            y_true=labels,
            scores=scores,
            threshold=threshold,
            point_adjust=self.point_adjust,
        )
        
        # Add additional statistics
        metrics['num_samples'] = len(labels)
        metrics['num_anomalies'] = int(labels.sum())
        metrics['anomaly_ratio'] = float(labels.mean())
        metrics['score_mean'] = float(scores.mean())
        metrics['score_std'] = float(scores.std())
        metrics['threshold_source'] = threshold_source
        
        return metrics
    
    def evaluate_per_entity(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate per entity (for datasets with multiple machines/entities).
        
        Args:
            entity_ids: List of entity identifiers
            
        Returns:
            Dictionary mapping entity_id to metrics
        """
        # This would require entity information in the dataset
        # For now, return overall metrics
        return {'overall': self.evaluate()}


def load_model(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """Load model from checkpoint.
    
    Returns:
        Tuple of (model, checkpoint_info) where checkpoint_info contains threshold etc.
    """
    logger.info(f"Loading model from {checkpoint_path}")
    model, checkpoint_info = TSFMADModel.load_checkpoint(checkpoint_path, device=device)
    return model, checkpoint_info


def main():
    parser = argparse.ArgumentParser(description='Evaluate TSFM-AD model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides checkpoint config)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--no_point_adjust', action='store_true',
                        help='Disable point-adjusted metrics')
    parser.add_argument('--ignore_checkpoint_threshold', action='store_true',
                        help='Ignore threshold from checkpoint and search on test set (not recommended)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load model
    model, checkpoint_info = load_model(args.checkpoint, device=device)
    config = model.config
    
    # Get threshold from checkpoint
    checkpoint_threshold = checkpoint_info.get('threshold')
    if checkpoint_threshold is not None and not args.ignore_checkpoint_threshold:
        logger.info(f"Using threshold from checkpoint: {checkpoint_threshold:.4f}")
        use_threshold = checkpoint_threshold
    else:
        if checkpoint_threshold is None:
            logger.info(
                "No threshold found in checkpoint, will search on test set. "
                "Note: This is common in baseline papers but controversial."
            )
        else:
            logger.info("Ignoring checkpoint threshold as requested")
        use_threshold = None
    
    # Override dataset if specified
    if args.dataset:
        config['data']['dataset'] = args.dataset
    
    # Load test data
    data_cfg = config.get('data', {})
    dataset_name = data_cfg.get('dataset', 'SMD')
    
    logger.info(f"Loading test dataset: {dataset_name}")
    
    try:
        # Use TSFMADDataLoader to load test data
        data_loader = TSFMADDataLoader(
            config={
                'window_size': data_cfg.get('window_size', 100),
                'stride': 1,  # Use stride=1 for evaluation
                'normalize': data_cfg.get('normalize', True),
                'val_ratio': 0.0,  # No validation split for test
            }
        )
        
        datasets = data_loader.load_dataset(dataset_name)
        test_dataset = datasets['test_dataset']
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create evaluator and run evaluation
    evaluator = TSFMADEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        point_adjust=not args.no_point_adjust,
    )
    
    metrics = evaluator.evaluate(threshold=use_threshold)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Anomalies: {metrics['num_anomalies']} ({metrics['anomaly_ratio']:.2%})")
    print(f"Threshold Source: {metrics['threshold_source']}")
    print("-" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print("=" * 50)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return metrics


if __name__ == '__main__':
    main()
