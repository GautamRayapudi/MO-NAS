"""
Architecture Training and Evaluation
"""

import time
import warnings
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core import SearchConfig, UniversalModelBuilder
from .proxies import ZeroCostProxy


class Trainer:
    """Enhanced trainer with zero-cost proxies and weight sharing"""
    
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.weight_dict = {}
        self.calibration_data = []
        self.warmup_epochs = 1
    
    def _get_layer_signature(self, layer_config) -> str:
        op = layer_config.operation
        params_str = "_".join(f"{k}:{v}" for k, v in sorted(layer_config.params.items()) if isinstance(v, (int, float)))
        return f"{op}_{params_str}_in{layer_config.input_shape}_out{layer_config.output_shape}"
    
    def _apply_weight_sharing(self, model, arch):
        if not self.cfg.use_weight_sharing or not self.weight_dict:
            return
        try:
            for layer, layer_cfg in zip(model.layers, arch.config.layers):
                sig = self._get_layer_signature(layer_cfg)
                if sig in self.weight_dict:
                    try: layer.load_state_dict(self.weight_dict[sig])
                    except: pass
        except: pass
    
    def _save_weights_to_dict(self, model, arch):
        if not self.cfg.use_weight_sharing:
            return
        try:
            for layer, layer_cfg in zip(model.layers, arch.config.layers):
                sig = self._get_layer_signature(layer_cfg)
                if sig not in self.weight_dict:
                    self.weight_dict[sig] = layer.state_dict()
        except: pass
    
    def train_architecture(self, arch, train_loader=None, val_loader=None, epochs=None):
        if not arch.is_valid:
            return self._heuristic_evaluation(arch)
        if self.cfg.use_zero_cost_proxy and train_loader is not None:
            return self._zero_cost_evaluation(arch, train_loader)
        if not TORCH_AVAILABLE or train_loader is None:
            return self._heuristic_evaluation(arch)
        
        epochs = epochs or self.cfg.search_epochs
        try:
            model = UniversalModelBuilder.build_model(arch)
            if model is None:
                return self._heuristic_evaluation(arch)
            if arch.memory_mb > self.cfg.max_memory_mb:
                arch.accuracy = 0.0
                return 0.0
            
            model = model.to(self.device)
            self._apply_weight_sharing(model, arch)
            
            if self.cfg.latency_weight > 0:
                arch.latency = self._measure_latency(model, arch.dataset_config)
            
            criterion = nn.CrossEntropyLoss() if arch.dataset_config.task_type == 'classification' else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - self.warmup_epochs))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [self.warmup_epochs])
            
            best_acc, patience = 0.0, 0
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        loss = criterion(model(data), target)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        if batch_idx >= 50 and epochs == self.cfg.search_epochs:
                            break
                    except: continue
                
                val_acc = self._evaluate(model, val_loader, criterion)
                if val_acc > best_acc:
                    best_acc, patience = val_acc, 0
                else:
                    patience += 1
                scheduler.step()
                if patience >= self.cfg.early_stopping_patience:
                    break
            
            self._save_weights_to_dict(model, arch)
            arch.accuracy, arch.trained = best_acc, True
            return best_acc
        except Exception as e:
            warnings.warn(f"Training failed: {e}")
            return self._heuristic_evaluation(arch)
    
    def _zero_cost_evaluation(self, arch, train_loader):
        try:
            data, _ = next(iter(train_loader))
            score = ZeroCostProxy.evaluate_architecture(arch, data, self.device)
            if len(self.calibration_data) >= 5:
                scores, accs = [d['proxy'] for d in self.calibration_data], [d['accuracy'] for d in self.calibration_data]
                min_s, max_s, min_a, max_a = min(scores), max(scores), min(accs), max(accs)
                estimated_acc = min_a + (score - min_s) / (max_s - min_s + 1e-6) * (max_a - min_a) if max_s > min_s else np.mean(accs)
            else:
                estimated_acc = 0.3 + 0.4 * min(score, 1.0)
            arch.accuracy, arch.trained = max(0.0, min(0.95, estimated_acc)), True
            self.calibration_data.append({'id': arch.config.id, 'proxy': score, 'accuracy': arch.accuracy})
            return arch.accuracy
        except:
            return self._heuristic_evaluation(arch)
    
    def _heuristic_evaluation(self, arch):
        score = 0.5
        targets = {'image': 15, 'text': 8, 'sequence': 6, 'tabular': 5}
        score -= abs(len(arch.config.layers) - targets.get(arch.dataset_config.data_type, 10)) * 0.015
        arch.calculate_complexity()
        if arch.dataset_config.data_type == 'image' and 200 < arch.flops < 3000:
            score += 0.15
        if arch.memory_mb > self.cfg.max_memory_mb:
            score -= 0.3
        score += min(len(set(l.operation for l in arch.config.layers)) * 0.015, 0.1)
        if arch.trainability_score > 0:
            score = 0.6 * score + 0.4 * arch.trainability_score
        arch.accuracy = max(0.0, min(0.95, score + np.random.normal(0, 0.04)))
        return arch.accuracy
    
    def _evaluate(self, model, val_loader, criterion):
        if val_loader is None:
            return 0.0
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    _, predicted = model(data).max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    if batch_idx >= 20:
                        break
                except: continue
        return correct / total if total > 0 else 0.0
    
    def _measure_latency(self, model, dc) -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        try:
            model.eval()
            if dc.data_type == 'image':
                dummy = torch.randn(1, dc.channels, dc.height, dc.width).to(self.device)
            elif dc.data_type == 'text':
                dummy = torch.randint(0, dc.vocab_size, (1, dc.max_seq_length)).to(self.device)
            elif dc.data_type == 'sequence':
                dummy = torch.randn(1, dc.sequence_length, dc.feature_dim).to(self.device)
            else:
                dummy = torch.randn(1, dc.numerical_features).to(self.device)
            for _ in range(10): _ = model(dummy)
            if self.device == 'cuda': torch.cuda.synchronize()
            start = time.time()
            for _ in range(100): _ = model(dummy)
            if self.device == 'cuda': torch.cuda.synchronize()
            return (time.time() - start) * 10
        except:
            return 0.0
