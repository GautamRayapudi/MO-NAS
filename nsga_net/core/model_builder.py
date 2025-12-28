"""
PyTorch Model Builder
"""

import warnings
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .architecture import Architecture, LayerConfig


class UniversalModelBuilder:
    """Enhanced model builder with modern operations"""
    
    @staticmethod
    def build_model(architecture: Architecture) -> Optional['nn.Module']:
        if not TORCH_AVAILABLE or not architecture.is_valid:
            return None
        
        try:
            dt = architecture.dataset_config.data_type
            if dt == 'image':
                return UniversalModelBuilder._build_image_model(architecture.config, architecture.dataset_config)
            elif dt == 'text':
                return UniversalModelBuilder._build_text_model(architecture.config, architecture.dataset_config)
            elif dt == 'sequence':
                return UniversalModelBuilder._build_sequence_model(architecture.config, architecture.dataset_config)
            elif dt == 'tabular':
                return UniversalModelBuilder._build_tabular_model(architecture.config, architecture.dataset_config)
        except Exception as e:
            warnings.warn(f"Error building model: {e}")
            return None
    
    @staticmethod
    def _build_image_model(config, dataset_config):
        class ImageModel(nn.Module):
            def __init__(self, cfg, dcfg):
                super().__init__()
                self.layers = nn.ModuleList()
                for lc in cfg.layers:
                    layer = UniversalModelBuilder._create_layer(lc, dcfg)
                    if layer:
                        self.layers.append(layer)
            
            def forward(self, x):
                for layer in self.layers:
                    if isinstance(layer, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                        x = layer(x)
                        x = x.view(x.size(0), -1)
                    else:
                        x = layer(x)
                return x
        return ImageModel(config, dataset_config)
    
    @staticmethod
    def _build_text_model(config, dataset_config):
        class TextModel(nn.Module):
            def __init__(self, cfg, dcfg):
                super().__init__()
                self.layers = nn.ModuleList()
                self.configs = cfg.layers
                for lc in cfg.layers:
                    layer = UniversalModelBuilder._create_layer(lc, dcfg)
                    if layer:
                        self.layers.append(layer)
            
            def forward(self, x):
                for layer, cfg in zip(self.layers, self.configs):
                    if isinstance(layer, nn.Embedding):
                        x = layer(x)
                    elif isinstance(layer, (nn.LSTM, nn.GRU)):
                        x, _ = layer(x)
                    elif 'pool' in cfg.operation.lower() and len(x.shape) == 3:
                        x = x.max(dim=1)[0] if 'max' in cfg.operation else x.mean(dim=1)
                    elif isinstance(layer, nn.Linear) and len(x.shape) > 2:
                        x = x.view(x.size(0), -1)
                        x = layer(x)
                    else:
                        x = layer(x)
                return x
        return TextModel(config, dataset_config)
    
    @staticmethod
    def _build_sequence_model(config, dataset_config):
        class SequenceModel(nn.Module):
            def __init__(self, cfg, dcfg):
                super().__init__()
                self.layers = nn.ModuleList()
                for lc in cfg.layers:
                    layer = UniversalModelBuilder._create_layer(lc, dcfg)
                    if layer:
                        self.layers.append(layer)
            
            def forward(self, x):
                for layer in self.layers:
                    if isinstance(layer, (nn.LSTM, nn.GRU)):
                        x, _ = layer(x)
                    elif isinstance(layer, nn.Linear) and len(x.shape) == 3:
                        x = x[:, -1, :]
                        x = layer(x)
                    else:
                        x = layer(x)
                return x
        return SequenceModel(config, dataset_config)
    
    @staticmethod
    def _build_tabular_model(config, dataset_config):
        class TabularModel(nn.Module):
            def __init__(self, cfg, dcfg):
                super().__init__()
                self.layers = nn.ModuleList()
                for lc in cfg.layers:
                    layer = UniversalModelBuilder._create_layer(lc, dcfg)
                    if layer:
                        self.layers.append(layer)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        return TabularModel(config, dataset_config)
    
    @staticmethod
    def _create_layer(lc: LayerConfig, dc):
        op = lc.operation.lower()
        p = lc.params
        
        try:
            if 'conv' in op and '1d' not in op and 'depthwise' not in op:
                return nn.Sequential(
                    nn.Conv2d(p.get('in_channels', 3), p.get('out_channels', 64),
                             p.get('kernel_size', 3), stride=p.get('stride', 1),
                             padding=p.get('kernel_size', 3)//2, bias=False),
                    nn.BatchNorm2d(p.get('out_channels', 64)),
                    nn.ReLU(inplace=True)
                )
            elif 'depthwise' in op:
                ic, oc, k = p.get('in_channels', 3), p.get('out_channels', 64), p.get('kernel_size', 3)
                return nn.Sequential(
                    nn.Conv2d(ic, ic, k, stride=p.get('stride', 1), padding=k//2, groups=ic, bias=False),
                    nn.BatchNorm2d(ic), nn.ReLU(inplace=True),
                    nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)
                )
            elif 'conv1d' in op:
                return nn.Sequential(
                    nn.Conv1d(p.get('in_channels', 128), p.get('out_channels', 128),
                             p.get('kernel_size', 3), padding=p.get('kernel_size', 3)//2, bias=False),
                    nn.BatchNorm1d(p.get('out_channels', 128)), nn.ReLU(inplace=True)
                )
            elif 'global_avg_pool' in op:
                return nn.AdaptiveAvgPool2d(1)
            elif 'global_max_pool' in op:
                return nn.AdaptiveMaxPool1d(1)
            elif 'linear' in op or ('dense' in op and 'dropout' not in op and 'bn' not in op):
                layers = [nn.Linear(p.get('in_features', 128), p.get('out_features', 10))]
                if p.get('activation', True):
                    layers.append(nn.ReLU(inplace=True))
                return nn.Sequential(*layers)
            elif 'dense_bn' in op:
                of = p.get('out_features', 128)
                return nn.Sequential(nn.Linear(p.get('in_features', 128), of), nn.BatchNorm1d(of), nn.ReLU(inplace=True))
            elif 'dense_dropout' in op:
                of = p.get('out_features', 128)
                return nn.Sequential(nn.Linear(p.get('in_features', 128), of), nn.Dropout(p.get('dropout', 0.2)), nn.ReLU(inplace=True))
            elif op in ['lstm', 'bilstm']:
                return nn.LSTM(p.get('input_dim', 128), p.get('hidden_dim', 256), batch_first=True, bidirectional='bi' in op)
            elif op in ['gru', 'bigru']:
                return nn.GRU(p.get('input_dim', 128), p.get('hidden_dim', 256), batch_first=True, bidirectional='bi' in op)
            elif 'embedding' in op:
                return nn.Embedding(p.get('vocab_size', dc.vocab_size or 30000), p.get('embed_dim', 512))
            elif 'transformer' in op:
                ed, nh = p.get('embed_dim', 512), p.get('num_heads', 8)
                if ed % nh != 0:
                    nh = max([h for h in [4, 8, 12, 16] if ed % h == 0], default=4)
                return nn.TransformerEncoderLayer(d_model=ed, nhead=nh, dim_feedforward=p.get('ffn_dim', ed*4), dropout=0.1, batch_first=True)
            else:
                return nn.Identity()
        except Exception as e:
            warnings.warn(f"Error creating layer {op}: {e}")
            return nn.Identity()
