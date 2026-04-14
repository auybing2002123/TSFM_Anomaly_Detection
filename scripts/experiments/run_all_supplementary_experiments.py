#!/usr/bin/env python
"""
Run all supplementary experiments for the paper.

Experiments:
1. DA-LoRA: SMD→MSL (MMD), SMD→MSL (CORAL), SMD→SMAP (CORAL)
2. More cross-domain pairs: MSL→PSM, PSM→MSL, SMAP→PSM
3. Zero-shot for new pairs

Usage:
    python scripts/run_all_supplementary_experiments.py
"""

import subprocess
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.resolve()

def run_experiment(cmd: str, description: str):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*70)
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(project_dir),
    )
    
    if result.returncode != 0:
        print(f"⚠️ Experiment failed: {description}")
    else:
        print(f"✅ Completed: {description}")
    
    return result.returncode == 0


def main():
    experiments = [
        # DA-LoRA experiments
        ("python scripts/run_cross_domain_experiments.py --mode da_lora --source SMD --target MSL --domain_method mmd --epochs 5",
         "DA-LoRA: SMD → MSL (MMD)"),
        
        ("python scripts/run_cross_domain_experiments.py --mode da_lora --source SMD --target MSL --domain_method coral --epochs 5",
         "DA-LoRA: SMD → MSL (CORAL)"),
        
        ("python scripts/run_cross_domain_experiments.py --mode da_lora --source SMD --target SMAP --domain_method coral --epochs 5",
         "DA-LoRA: SMD → SMAP (CORAL)"),
        
        # New cross-domain zero-shot pairs
        ("python scripts/run_cross_domain_experiments.py --mode cross_domain --source MSL --target PSM --epochs 3",
         "Zero-shot: MSL → PSM"),
        
        ("python scripts/run_cross_domain_experiments.py --mode cross_domain --source PSM --target MSL --epochs 3",
         "Zero-shot: PSM → MSL"),
        
        ("python scripts/run_cross_domain_experiments.py --mode cross_domain --source SMAP --target PSM --epochs 3",
         "Zero-shot: SMAP → PSM"),
    ]
    
    print("="*70)
    print("SUPPLEMENTARY EXPERIMENTS")
    print(f"Total: {len(experiments)} experiments")
    print("="*70)
    
    results = []
    for cmd, desc in experiments:
        success = run_experiment(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for desc, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {desc}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\nCompleted: {success_count}/{len(results)}")


if __name__ == '__main__':
    main()
