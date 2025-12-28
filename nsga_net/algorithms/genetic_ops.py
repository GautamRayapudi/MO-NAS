"""
Genetic Operations for Neural Architecture Search
"""

import copy
import random
import warnings

from ..core import Architecture, ArchitectureConfig, LayerConfig


class GeneticOperations:
    """Genetic operations for architecture evolution"""
    
    @staticmethod
    def crossover(p1: Architecture, p2: Architecture, rate: float = 0.8):
        if random.random() > rate or len(p1.config.layers) < 4:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        
        try:
            c1_cfg, c2_cfg = copy.deepcopy(p1.config), copy.deepcopy(p2.config)
            min_d = min(len(c1_cfg.layers), len(c2_cfg.layers))
            if min_d < 4:
                return copy.deepcopy(p1), copy.deepcopy(p2)
            
            valid_points = []
            for sp in range(2, min_d - 2):
                p1_seg_out = p1.config.layers[sp-1].output_shape
                p2_next_in = p2.config.layers[sp].input_shape
                p2_seg_out = p2.config.layers[sp-1].output_shape
                p1_next_in = p1.config.layers[sp].input_shape
                
                if p1_seg_out == p2_next_in and p2_seg_out == p1_next_in:
                    valid_points.append(sp)
                elif (len(p1_seg_out) >= 1 and len(p2_next_in) >= 1 and
                      p1_seg_out[0] == p2_next_in[0] and p2_seg_out[0] == p1_next_in[0]):
                    valid_points.append(sp)
            
            if not valid_points:
                return copy.deepcopy(p1), copy.deepcopy(p2)
            
            sp = random.choice(valid_points)
            c1_cfg.layers = p1.config.layers[:sp] + p2.config.layers[sp:]
            c2_cfg.layers = p2.config.layers[:sp] + p1.config.layers[sp:]
            c1_cfg.depth, c2_cfg.depth = len(c1_cfg.layers), len(c2_cfg.layers)
            
            ch1, ch2 = Architecture(c1_cfg, p1.dataset_config), Architecture(c2_cfg, p2.dataset_config)
            if not ch1.is_valid or not ch2.is_valid:
                return copy.deepcopy(p1), copy.deepcopy(p2)
            return ch1, ch2
        except Exception as e:
            warnings.warn(f"Crossover failed: {e}")
            return copy.deepcopy(p1), copy.deepcopy(p2)
    
    @staticmethod
    def mutate(arch: Architecture, rate: float = 0.1):
        try:
            m = copy.deepcopy(arch)
            cfg = m.config
            if len(cfg.layers) < 3:
                return m
            
            mutation_type = random.choice(['operation', 'parameter', 'depth'])
            
            if mutation_type == 'operation':
                from ..search.search_spaces import create_search_space
                for i in range(1, len(cfg.layers) - 1):
                    if random.random() < rate:
                        ss = create_search_space(arch.dataset_config)
                        cfg.layers[i].operation = random.choice(ss.get_operations())
            
            elif mutation_type == 'parameter':
                for i in range(1, len(cfg.layers) - 1):
                    if random.random() < rate:
                        layer = cfg.layers[i]
                        if 'out_channels' in layer.params:
                            current = layer.params['out_channels']
                            layer.params['out_channels'] = max(8, min(512, random.choice([current // 2, current, current * 2])))
                        elif 'hidden_dim' in layer.params:
                            current = layer.params['hidden_dim']
                            layer.params['hidden_dim'] = max(32, min(512, random.choice([current // 2, current, current * 2])))
            
            elif mutation_type == 'depth' and len(cfg.layers) > 4:
                if random.random() < 0.5 and len(cfg.layers) < 25:
                    pos = random.randint(1, len(cfg.layers) - 2)
                    existing = cfg.layers[pos]
                    cfg.layers.insert(pos, LayerConfig('skip_connect', {}, existing.input_shape, existing.output_shape))
                elif len(cfg.layers) > 5:
                    cfg.layers.pop(random.randint(1, len(cfg.layers) - 3))
                cfg.depth = len(cfg.layers)
            
            m.config = cfg
            valid, _ = cfg.validate()
            if not valid:
                return copy.deepcopy(arch)
            m.is_valid = valid
            return m
        except Exception as e:
            warnings.warn(f"Mutation failed: {e}")
            return copy.deepcopy(arch)
