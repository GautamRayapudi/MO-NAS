"""
Architecture Configuration and Representation
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

from .config import DatasetConfig


@dataclass
class LayerConfig:
    """Layer configuration"""
    operation: str
    params: Dict[str, Union[int, float, str]]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    def __post_init__(self):
        self.input_shape = tuple(self.input_shape)
        self.output_shape = tuple(self.output_shape)


@dataclass
class ArchitectureConfig:
    """Architecture configuration"""
    id: str
    data_type: str
    task_type: str
    layers: List[LayerConfig] = field(default_factory=list)
    depth: int = 0
    
    def add_layer(self, operation: str, params: Dict, input_shape: Tuple, output_shape: Tuple):
        layer = LayerConfig(operation, params, tuple(input_shape), tuple(output_shape))
        self.layers.append(layer)
        self.depth += 1
    
    def validate(self) -> Tuple[bool, str]:
        if not self.layers or self.depth < 2:
            return False, "Too few layers"
        
        for i in range(len(self.layers) - 1):
            curr_out = self.layers[i].output_shape
            next_in = self.layers[i+1].input_shape
            
            curr_op = self.layers[i].operation.lower()
            next_op = self.layers[i+1].operation.lower()
            skip_ops = ('global_avg_pool', 'global_max_pool', 'flatten', 'adaptive', 'linear')
            if any(op in curr_op for op in skip_ops) or any(op in next_op for op in skip_ops):
                continue
            
            if len(curr_out) == 3 and len(next_in) == 3:
                if curr_out[0] != next_in[0]:
                    return False, f"Channel mismatch at layer {i}: {curr_out[0]} -> {next_in[0]}"
            elif len(curr_out) == len(next_in):
                if curr_out != next_in:
                    return False, f"Shape mismatch at layer {i}: {curr_out} -> {next_in}"
        
        has_trainable = any('conv' in l.operation.lower() or 'linear' in l.operation.lower() 
                           or 'lstm' in l.operation.lower() for l in self.layers)
        if not has_trainable:
            return False, "No trainable layers"
        return True, "Valid"


class Architecture:
    """Architecture with enhanced validation and metrics"""
    
    def __init__(self, config: ArchitectureConfig, dataset_config: DatasetConfig):
        self.config = config
        self.dataset_config = dataset_config
        valid, msg = self.config.validate()
        self.is_valid = valid
        self.validation_message = msg
        
        self.accuracy = 0.0
        self.loss = float('inf')
        self.flops = 0.0
        self.params = 0.0
        self.latency = 0.0
        self.memory_mb = 0.0
        self.trainability_score = 0.0
        self.objectives = {}
        self.rank = 0
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions = []
        self.trained = False
    
    def calculate_complexity(self):
        self.flops = self._calculate_flops()
        self.params = self._calculate_params()
        self.memory_mb = self._estimate_memory()
        return self.flops, self.params, self.memory_mb
    
    def _calculate_flops(self) -> float:
        total = 0
        for layer in self.config.layers:
            op = layer.operation.lower()
            p = layer.params
            in_s = layer.input_shape
            out_s = layer.output_shape
            
            try:
                if 'conv' in op and '1d' not in op:
                    k = p.get('kernel_size', 3)
                    ic = in_s[0] if len(in_s) >= 3 else 1
                    oc = out_s[0] if len(out_s) >= 3 else 1
                    spatial = np.prod(out_s[1:]) if len(out_s) > 1 else 1
                    groups = p.get('groups', 1)
                    if 'depthwise' in op:
                        total += k * k * ic * spatial + ic * oc * spatial
                    else:
                        total += k * k * ic * oc * spatial / groups
                    total += 2 * oc * spatial + oc * spatial
                elif 'linear' in op or 'dense' in op:
                    total += 2 * int(np.prod(in_s)) * int(np.prod(out_s))
                elif 'lstm' in op:
                    seq = in_s[0] if len(in_s) > 1 else 1
                    inp_d = in_s[-1] if len(in_s) > 1 else in_s[0]
                    hid_d = p.get('hidden_dim', out_s[-1])
                    bi = 2 if p.get('bidirectional', False) else 1
                    total += seq * bi * 4 * (inp_d * hid_d + hid_d * hid_d + hid_d)
                elif 'gru' in op:
                    seq = in_s[0] if len(in_s) > 1 else 1
                    inp_d = in_s[-1] if len(in_s) > 1 else in_s[0]
                    hid_d = p.get('hidden_dim', out_s[-1])
                    bi = 2 if p.get('bidirectional', False) else 1
                    total += seq * bi * 3 * (inp_d * hid_d + hid_d * hid_d)
                elif 'conv1d' in op:
                    k = p.get('kernel_size', 3)
                    ic = in_s[-1] if len(in_s) >= 2 else in_s[0]
                    oc = out_s[-1] if len(out_s) >= 2 else out_s[0]
                    seq_len = out_s[0] if len(out_s) >= 2 else 1
                    total += k * ic * oc * seq_len + 2 * oc * seq_len
            except Exception as e:
                warnings.warn(f"FLOPs calculation error: {e}")
        return total / 1e6
    
    def _calculate_params(self) -> float:
        total = 0
        for layer in self.config.layers:
            op = layer.operation.lower()
            p = layer.params
            in_s = layer.input_shape
            out_s = layer.output_shape
            
            try:
                if 'conv' in op and '1d' not in op:
                    k = p.get('kernel_size', 3)
                    ic = in_s[0] if len(in_s) >= 3 else 1
                    oc = out_s[0] if len(out_s) >= 3 else 1
                    if 'depthwise' in op:
                        total += k * k * ic + ic * oc
                    else:
                        total += k * k * ic * oc / p.get('groups', 1)
                    total += 2 * oc
                elif 'linear' in op or 'dense' in op:
                    total += int(np.prod(in_s)) * int(np.prod(out_s)) + int(np.prod(out_s))
                elif 'lstm' in op:
                    inp_d = in_s[-1] if len(in_s) > 1 else in_s[0]
                    hid_d = p.get('hidden_dim', out_s[-1])
                    bi = 2 if p.get('bidirectional', False) else 1
                    total += bi * 4 * (inp_d * hid_d + hid_d * hid_d + 2 * hid_d)
                elif 'gru' in op:
                    inp_d = in_s[-1] if len(in_s) > 1 else in_s[0]
                    hid_d = p.get('hidden_dim', out_s[-1])
                    bi = 2 if p.get('bidirectional', False) else 1
                    total += bi * 3 * (inp_d * hid_d + hid_d * hid_d + 2 * hid_d)
            except Exception as e:
                warnings.warn(f"Params calculation error: {e}")
        return total / 1e6
    
    def _estimate_memory(self) -> float:
        param_memory = self.params * 4
        max_activation = max((np.prod(l.output_shape) * 4 / 1e6 for l in self.config.layers), default=0)
        return param_memory + max_activation + 3 * param_memory
    
    def evaluate_objectives(self):
        self.calculate_complexity()
        self.objectives = {
            'accuracy': self.accuracy,
            'flops': self.flops,
            'params': self.params,
            'latency': self.latency,
            'memory': self.memory_mb,
            'error_rate': 1.0 - self.accuracy
        }
        return self.objectives
