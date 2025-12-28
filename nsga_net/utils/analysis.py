"""
Results Analysis and Export
"""

import json
from pathlib import Path
import numpy as np


class ResultsAnalyzer:
    """Analyze and export search results"""
    
    @staticmethod
    def print_pareto_front(pareto, dtype: str):
        print(f"\n{'='*95}\nPareto Front - {dtype.upper()}\n{'='*95}")
        print(f"{'ID':<15} {'Acc':<10} {'FLOPs(M)':<12} {'Params(M)':<12} {'Latency(ms)':<15} {'Memory(MB)':<12}")
        print("-"*95)
        for a in sorted(pareto, key=lambda x: -x.accuracy):
            lat = f"{a.latency:.2f}" if a.latency > 0 else "N/A"
            print(f"{a.config.id:<15} {a.accuracy:<10.4f} {a.flops:<12.1f} {a.params:<12.2f} {lat:<15} {a.memory_mb:<12.1f}")
        print("="*95)
    
    @staticmethod
    def compute_statistics(pareto):
        if not pareto: return
        accs, flops, params = [a.accuracy for a in pareto], [a.flops for a in pareto], [a.params for a in pareto]
        print(f"\n{'='*60}\nStatistics\n{'='*60}")
        print(f"Solutions: {len(pareto)}")
        print(f"Accuracy: Min={min(accs):.4f}, Max={max(accs):.4f}, Mean={np.mean(accs):.4f}")
        print(f"FLOPs(M): Min={min(flops):.1f}, Max={max(flops):.1f}, Mean={np.mean(flops):.1f}")
        print(f"Params(M): Min={min(params):.2f}, Max={max(params):.2f}, Mean={np.mean(params):.2f}")
        print("="*60)
    
    @staticmethod
    def export_architecture(arch, filepath: str):
        data = {
            'id': arch.config.id, 'type': arch.config.data_type, 'task': arch.config.task_type,
            'layers': [{'operation': l.operation, 'params': l.params, 'input_shape': l.input_shape, 'output_shape': l.output_shape} for l in arch.config.layers],
            'performance': {'accuracy': float(arch.accuracy), 'flops': float(arch.flops), 'params': float(arch.params), 'latency': float(arch.latency), 'memory_mb': float(arch.memory_mb)}
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f: json.dump(data, f, indent=2)
        print(f"✓ Architecture exported to {filepath}")
    
    @staticmethod
    def save_search_results(nsga_net, pareto, filepath: str):
        results = {
            'dataset': {'type': nsga_net.dc.data_type, 'shape': nsga_net.dc.input_shape, 'classes': nsga_net.dc.num_classes},
            'search_config': {'population': nsga_net.sc.population_size, 'generations': nsga_net.sc.generations},
            'history': nsga_net.history,
            'pareto_front': [{'id': a.config.id, 'accuracy': float(a.accuracy), 'flops': float(a.flops), 'params': float(a.params)} for a in pareto]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        print(f"✓ Results saved to {filepath}")
