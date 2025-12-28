"""
Search Space Definitions
"""

import random
from abc import ABC, abstractmethod

from ..core import DatasetConfig, ArchitectureConfig


class SearchSpace(ABC):
    @abstractmethod
    def sample_architecture(self): pass
    
    @abstractmethod
    def get_operations(self): pass


class ImageSearchSpace(SearchSpace):
    def __init__(self, dc: DatasetConfig):
        self.dc = dc
        self.ops = ['conv_3x3', 'conv_5x5', 'depthwise_conv_3x3', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect']
        self.channels = [16, 32, 64, 128, 256]
        self.depth_range = (8, 20)
    
    def sample_architecture(self):
        cfg = ArchitectureConfig(id=f"img_{random.randint(0,999999):06d}", data_type='image', task_type=self.dc.task_type)
        depth = random.randint(*self.depth_range)
        ch, h, w = self.dc.channels, self.dc.height, self.dc.width
        stem_ch = random.choice([16, 32])
        cfg.add_layer('conv_3x3', {'in_channels': ch, 'out_channels': stem_ch, 'kernel_size': 3, 'stride': 1}, (ch, h, w), (stem_ch, h, w))
        ch = stem_ch
        for i in range(1, depth):
            if i in [depth//3, 2*depth//3] and h > 8:
                stride, out_ch, op = 2, min(ch * 2, 512), 'conv_3x3'
                h, w = h // 2, w // 2
            else:
                stride, op = 1, random.choice(self.ops)
                valid = [c for c in self.channels if c >= ch//2 and c <= ch*2] or [ch]
                out_ch = random.choice(valid)
            if 'pool' in op or 'skip' in op: out_ch = ch
            cfg.add_layer(op, {'in_channels': ch, 'out_channels': out_ch, 'stride': stride, 'kernel_size': 3}, (ch, h, w), (out_ch, h, w))
            ch = out_ch
        cfg.add_layer('global_avg_pool', {}, (ch, h, w), (ch,))
        cfg.add_layer('linear', {'in_features': ch, 'out_features': self.dc.num_classes, 'activation': False}, (ch,), (self.dc.num_classes,))
        return cfg
    
    def get_operations(self): return self.ops


class TextSearchSpace(SearchSpace):
    def __init__(self, dc: DatasetConfig):
        self.dc = dc
        self.ops = ['lstm', 'gru', 'bilstm', 'bigru', 'transformer_encoder', 'conv1d_3']
        self.embed_dims, self.hidden_dims = [128, 256, 512], [128, 256, 512]
        self.depth_range = (4, 12)
    
    def sample_architecture(self):
        cfg = ArchitectureConfig(id=f"txt_{random.randint(0,999999):06d}", data_type='text', task_type=self.dc.task_type)
        depth, ed = random.randint(*self.depth_range), random.choice(self.embed_dims)
        cfg.add_layer('embedding', {'vocab_size': self.dc.vocab_size, 'embed_dim': ed}, (self.dc.max_seq_length,), (self.dc.max_seq_length, ed))
        dim, seq = ed, self.dc.max_seq_length
        for _ in range(depth):
            op = random.choice(self.ops)
            if 'lstm' in op or 'gru' in op:
                hd, bi = random.choice(self.hidden_dims), 'bi' in op
                cfg.add_layer(op, {'input_dim': dim, 'hidden_dim': hd, 'bidirectional': bi}, (seq, dim), (seq, hd * (2 if bi else 1)))
                dim = hd * (2 if bi else 1)
            elif 'transformer' in op:
                nh = random.choice([h for h in [4, 8] if ed % h == 0])
                cfg.add_layer(op, {'embed_dim': dim, 'num_heads': nh, 'ffn_dim': dim*4}, (seq, dim), (seq, dim))
            else:
                cfg.add_layer(op, {'in_channels': dim, 'out_channels': dim, 'kernel_size': 3}, (seq, dim), (seq, dim))
        cfg.add_layer('global_max_pool', {}, (seq, dim), (dim,))
        cfg.add_layer('linear', {'in_features': dim, 'out_features': self.dc.num_classes, 'activation': False}, (dim,), (self.dc.num_classes,))
        return cfg
    
    def get_operations(self): return self.ops


class SequenceSearchSpace(SearchSpace):
    def __init__(self, dc: DatasetConfig):
        self.dc = dc
        self.ops, self.hidden_dims = ['lstm', 'gru', 'dense'], [32, 64, 128, 256]
        self.depth_range = (3, 10)
    
    def sample_architecture(self):
        cfg = ArchitectureConfig(id=f"seq_{random.randint(0,999999):06d}", data_type='sequence', task_type=self.dc.task_type)
        depth, dim, seq = random.randint(*self.depth_range), self.dc.feature_dim, self.dc.sequence_length
        for _ in range(depth):
            op, hd = random.choice(self.ops), random.choice(self.hidden_dims)
            if op in ['lstm', 'gru']:
                cfg.add_layer(op, {'input_dim': dim, 'hidden_dim': hd}, (seq, dim), (seq, hd))
            else:
                cfg.add_layer(op, {'in_features': dim, 'out_features': hd}, (dim,), (hd,))
            dim = hd
        out_d = self.dc.num_classes if self.dc.task_type == 'classification' else 1
        cfg.add_layer('linear', {'in_features': dim, 'out_features': out_d, 'activation': False}, (dim,), (out_d,))
        return cfg
    
    def get_operations(self): return self.ops


class TabularSearchSpace(SearchSpace):
    def __init__(self, dc: DatasetConfig):
        self.dc = dc
        self.ops, self.hidden_dims = ['dense', 'dense_bn', 'dense_dropout'], [32, 64, 128, 256, 512]
        self.depth_range = (2, 8)
    
    def sample_architecture(self):
        cfg = ArchitectureConfig(id=f"tab_{random.randint(0,999999):06d}", data_type='tabular', task_type=self.dc.task_type)
        depth, dim = random.randint(*self.depth_range), self.dc.numerical_features
        for _ in range(depth):
            op, hd = random.choice(self.ops), random.choice([d for d in self.hidden_dims if d <= dim * 2])
            params = {'in_features': dim, 'out_features': hd}
            if 'dropout' in op: params['dropout'] = random.choice([0.1, 0.2, 0.3])
            cfg.add_layer(op, params, (dim,), (hd,))
            dim = hd
        out_d = self.dc.num_classes if self.dc.task_type == 'classification' else 1
        cfg.add_layer('linear', {'in_features': dim, 'out_features': out_d, 'activation': False}, (dim,), (out_d,))
        return cfg
    
    def get_operations(self): return self.ops


def create_search_space(dc: DatasetConfig) -> SearchSpace:
    spaces = {'image': ImageSearchSpace, 'text': TextSearchSpace, 'sequence': SequenceSearchSpace, 'tabular': TabularSearchSpace}
    if dc.data_type not in spaces:
        raise ValueError(f"Unknown data type: {dc.data_type}")
    return spaces[dc.data_type](dc)
