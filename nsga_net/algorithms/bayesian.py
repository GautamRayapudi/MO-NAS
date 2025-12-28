"""
Bayesian Optimization Guidance for NAS
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


class BayesianGuidance:
    """Gaussian Process-based surrogate model"""
    
    def __init__(self):
        self.history = []
        self.architecture_features = []
        self.performance_scores = []
        self.flops_scores = []
    
    def _extract_features(self, arch) -> np.ndarray:
        features = [len(arch.config.layers), arch.flops if arch.flops > 0 else 100, arch.params if arch.params > 0 else 1]
        ops_count = defaultdict(int)
        for layer in arch.config.layers:
            ops_count[layer.operation] += 1
        for op in ['conv', 'linear', 'lstm', 'pool', 'skip']:
            features.append(sum(v for k, v in ops_count.items() if op in k.lower()))
        features.append(arch.trainability_score)
        return np.array(features)
    
    def update(self, arch):
        self.history.append({'id': arch.config.id, 'depth': len(arch.config.layers), 
                            'accuracy': arch.accuracy, 'flops': arch.flops, 'trainability': arch.trainability_score})
        self.architecture_features.append(self._extract_features(arch))
        self.performance_scores.append(arch.accuracy)
        self.flops_scores.append(arch.flops)
    
    def predict_performance(self, arch) -> Tuple[float, float, float]:
        if len(self.history) < 5:
            return 0.5, 100.0, 0.5
        try:
            X, features = np.array(self.architecture_features), self._extract_features(arch)
            y_acc, y_flops = np.array(self.performance_scores), np.array(self.flops_scores)
            distances = np.sqrt(((X - features) ** 2).sum(axis=1))
            k = min(5, len(X))
            nearest_idx = np.argsort(distances)[:k]
            weights = 1.0 / (distances[nearest_idx] + 1e-6)
            weights /= weights.sum()
            return (y_acc[nearest_idx] * weights).sum(), (y_flops[nearest_idx] * weights).sum(), y_acc[nearest_idx].std()
        except Exception:
            return 0.5, 100.0, 0.5
    
    def suggest_promising_architectures(self, n: int = 3) -> List[Dict]:
        if len(self.history) < 10:
            return []
        sorted_history = sorted(self.history, key=lambda x: x['accuracy'], reverse=True)[:min(10, len(self.history))]
        return [{'optimal_depth': h['depth'], 'target_flops': h['flops'], 'expected_accuracy': h['accuracy']} for h in sorted_history[:n]]
