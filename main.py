#!/usr/bin/env python3
"""
NSGA-Net: Multi-Objective Neural Architecture Search
Main entry point and demo
"""

import random
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nsga_net import (
    DatasetConfig,
    SearchConfig,
    NSGANet,
    UniversalModelBuilder,
    ResultsAnalyzer
)


def demo_image():
    """Demo for image classification (CIFAR-10 style)"""
    print("\n" + "="*70)
    print("Image Classification Demo (CIFAR-10)")
    print("="*70)
    
    dc = DatasetConfig('image', (3, 32, 32), 10)
    sc = SearchConfig(
        population_size=10,
        generations=5,
        use_zero_cost_proxy=True,
        use_bayesian_guidance=True
    )
    
    nsga = NSGANet(dc, sc)
    pareto = nsga.search(verbose=False)
    
    if pareto:
        best = nsga.get_best_architecture(pareto, 'balanced')
        print(f"✓ Best: Acc={best.accuracy:.4f}, FLOPs={best.flops:.1f}M, "
              f"Params={best.params:.2f}M, Memory={best.memory_mb:.1f}MB")
        return best
    return None


def demo_text():
    """Demo for text classification"""
    print("\n" + "="*70)
    print("Text Classification Demo")
    print("="*70)
    
    dc = DatasetConfig('text', 256, 5, vocab_size=30000)
    sc = SearchConfig(
        population_size=10,
        generations=5,
        use_zero_cost_proxy=True,
        use_bayesian_guidance=True
    )
    
    nsga = NSGANet(dc, sc)
    pareto = nsga.search(verbose=False)
    
    if pareto:
        best = nsga.get_best_architecture(pareto, 'balanced')
        print(f"✓ Best: Acc={best.accuracy:.4f}, FLOPs={best.flops:.1f}M, "
              f"Params={best.params:.2f}M, Memory={best.memory_mb:.1f}MB")
        return best
    return None


def demo_sequence():
    """Demo for time series prediction"""
    print("\n" + "="*70)
    print("Time Series Demo")
    print("="*70)
    
    dc = DatasetConfig('sequence', (100, 10), 1, task_type='regression')
    sc = SearchConfig(
        population_size=10,
        generations=5,
        use_zero_cost_proxy=True,
        use_bayesian_guidance=True
    )
    
    nsga = NSGANet(dc, sc)
    pareto = nsga.search(verbose=False)
    
    if pareto:
        best = nsga.get_best_architecture(pareto, 'balanced')
        print(f"✓ Best: Acc={best.accuracy:.4f}, FLOPs={best.flops:.1f}M, "
              f"Params={best.params:.2f}M, Memory={best.memory_mb:.1f}MB")
        return best
    return None


def demo_tabular():
    """Demo for tabular data classification"""
    print("\n" + "="*70)
    print("Tabular Data Demo")
    print("="*70)
    
    dc = DatasetConfig('tabular', 50, 10)
    sc = SearchConfig(
        population_size=10,
        generations=5,
        use_zero_cost_proxy=True,
        use_bayesian_guidance=True
    )
    
    nsga = NSGANet(dc, sc)
    pareto = nsga.search(verbose=False)
    
    if pareto:
        best = nsga.get_best_architecture(pareto, 'balanced')
        print(f"✓ Best: Acc={best.accuracy:.4f}, FLOPs={best.flops:.1f}M, "
              f"Params={best.params:.2f}M, Memory={best.memory_mb:.1f}MB")
        return best
    return None


def main():
    """Run all demos"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           NSGA-Net: Neural Architecture Search                  ║
║     Multi-Objective Optimization for Deep Learning              ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ Multi-modal: Image, Text, Sequence, Tabular                 ║
║  ✓ Multi-objective: Accuracy, FLOPs, Params, Latency           ║
║  ✓ Zero-cost proxies for fast evaluation                       ║
║  ✓ Bayesian guidance for search efficiency                     ║
║  ✓ Weight sharing for reduced training cost                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run demos
    for name, demo_fn in [
        ('Image', demo_image),
        ('Text', demo_text),
        ('Sequence', demo_sequence),
        ('Tabular', demo_tabular)
    ]:
        try:
            best = demo_fn()
            if best:
                results[name] = {
                    'accuracy': best.accuracy,
                    'flops': best.flops,
                    'params': best.params
                }
            else:
                results[name] = None
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = None
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for name, res in results.items():
        if res:
            print(f"{name:<20} Acc={res['accuracy']:.4f}, FLOPs={res['flops']:.1f}M")
        else:
            print(f"{name:<20} FAILED")
    print("="*70)
    
    print("\n✓ All demos completed!")


if __name__ == "__main__":
    main()
