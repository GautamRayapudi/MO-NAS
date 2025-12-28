"""
NSGA-Net Configuration Classes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Universal dataset configuration with validation"""
    data_type: str
    input_shape: Union[Tuple[int, ...], List[int]]
    num_classes: int
    task_type: str = 'classification'
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    vocab_size: Optional[int] = None
    max_seq_length: Optional[int] = None
    sequence_length: Optional[int] = None
    feature_dim: Optional[int] = None
    numerical_features: Optional[int] = None
    categorical_features: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.data_type not in ['image', 'text', 'sequence', 'tabular']:
            raise ValueError(f"Invalid data_type: {self.data_type}")
        
        if self.data_type == 'image':
            if isinstance(self.input_shape, (list, tuple)) and len(self.input_shape) == 3:
                self.channels, self.height, self.width = self.input_shape
        elif self.data_type == 'text':
            if isinstance(self.input_shape, int):
                self.max_seq_length = self.input_shape
            if self.vocab_size is None:
                self.vocab_size = 30000
        elif self.data_type == 'sequence':
            if isinstance(self.input_shape, (list, tuple)) and len(self.input_shape) == 2:
                self.sequence_length, self.feature_dim = self.input_shape
        elif self.data_type == 'tabular':
            if isinstance(self.input_shape, int):
                self.numerical_features = self.input_shape


@dataclass
class SearchConfig:
    """Enhanced search configuration"""
    population_size: int = 20
    generations: int = 30
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    search_epochs: int = 5
    final_epochs: int = 600
    batch_size: int = 128
    learning_rate: float = 0.025
    weight_decay: float = 3e-4
    accuracy_weight: float = 1.0
    flops_weight: float = 1.0
    params_weight: float = 0.5
    latency_weight: float = 0.0
    max_flops: Optional[float] = None
    max_params: Optional[float] = None
    max_latency: Optional[float] = None
    max_memory_mb: float = 8000
    use_weight_sharing: bool = True
    use_zero_cost_proxy: bool = True
    use_bayesian_guidance: bool = True
    early_stopping_patience: int = 10
    device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
