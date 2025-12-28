"""
NSGANet - Main Neural Architecture Search Class
"""

import random
import warnings
import numpy as np
from typing import Optional, List

from ..core import DatasetConfig, SearchConfig, Architecture
from ..algorithms import NSGAII, GeneticOperations, BayesianGuidance
from ..evaluation import Trainer
from .search_spaces import create_search_space


class NSGANet:
    """Enhanced NSGA-Net for Neural Architecture Search"""
    
    def __init__(self, dataset_config: DatasetConfig, search_config: SearchConfig):
        self.dc = dataset_config
        self.sc = search_config
        self.ss = create_search_space(dataset_config)
        self.genetic = GeneticOperations()
        self.bayesian = BayesianGuidance()
        self.trainer = Trainer(search_config)
        self.generation = 0
        self.history = []
        self.best_arch = None
    
    def initialize_population(self) -> List[Architecture]:
        pop, attempts = [], 0
        print(f"Initializing population of {self.sc.population_size}...")
        while len(pop) < self.sc.population_size and attempts < self.sc.population_size * 3:
            try:
                cfg = self.ss.sample_architecture()
                arch = Architecture(cfg, self.dc)
                if arch.is_valid:
                    arch.calculate_complexity()
                    if (not self.sc.max_flops or arch.flops <= self.sc.max_flops) and \
                       (not self.sc.max_params or arch.params <= self.sc.max_params) and \
                       arch.memory_mb <= self.sc.max_memory_mb:
                        pop.append(arch)
            except: pass
            attempts += 1
        print(f"✓ Initialized {len(pop)} valid architectures")
        return pop
    
    def evaluate_population(self, pop, train_loader=None, val_loader=None):
        for i, arch in enumerate(pop):
            if not arch.trained:
                try:
                    self.trainer.train_architecture(arch, train_loader, val_loader)
                    arch.evaluate_objectives()
                    self.bayesian.update(arch)
                except Exception as e:
                    warnings.warn(f"Evaluation failed: {e}")
                    arch.accuracy = 0.0
                    arch.evaluate_objectives()
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i+1}/{len(pop)} architectures")
        return pop
    
    def generate_offspring(self, pop) -> List[Architecture]:
        offspring, attempts = [], 0
        while len(offspring) < self.sc.population_size and attempts < self.sc.population_size * 3:
            try:
                p1, p2 = self._tournament_select(pop), self._tournament_select(pop)
                c1, c2 = self.genetic.crossover(p1, p2, self.sc.crossover_rate)
                c1, c2 = self.genetic.mutate(c1, self.sc.mutation_rate), self.genetic.mutate(c2, self.sc.mutation_rate)
                for child in [c1, c2]:
                    if child.is_valid:
                        child.calculate_complexity()
                        if (not self.sc.max_flops or child.flops <= self.sc.max_flops) and child.memory_mb <= self.sc.max_memory_mb:
                            offspring.append(child)
                            if len(offspring) >= self.sc.population_size: break
            except: pass
            attempts += 1
        return offspring[:self.sc.population_size]
    
    def _tournament_select(self, pop, size: int = 3):
        tournament = random.sample(pop, min(size, len(pop)))
        return min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
    
    def bayesian_guidance(self, offspring):
        if not self.sc.use_bayesian_guidance or len(self.bayesian.history) < 10:
            return offspring
        guided = []
        for arch in offspring:
            try:
                pred_acc, pred_flops, _ = self.bayesian.predict_performance(arch)
                if pred_acc < 0.35 or (self.sc.max_flops and pred_flops > self.sc.max_flops * 1.2):
                    new_arch = Architecture(self.ss.sample_architecture(), self.dc)
                    guided.append(new_arch if new_arch.is_valid else arch)
                else:
                    guided.append(arch)
            except: guided.append(arch)
        return guided
    
    def search(self, train_loader=None, val_loader=None, verbose: bool = True):
        print(f"\n{'='*70}\nNSGA-Net Enhanced Search\n{'='*70}")
        print(f"Dataset: {self.dc.data_type.upper()} | Task: {self.dc.task_type}")
        print(f"Input: {self.dc.input_shape} | Classes: {self.dc.num_classes}")
        print(f"Population: {self.sc.population_size} | Generations: {self.sc.generations}")
        print(f"Zero-cost proxy: {self.sc.use_zero_cost_proxy}\nBayesian guidance: {self.sc.use_bayesian_guidance}")
        print(f"{'='*70}\n")
        
        pop = self.evaluate_population(self.initialize_population(), train_loader, val_loader)
        pop = [a for a in pop if a.accuracy > 0]
        if not pop:
            raise RuntimeError("No valid architectures in initial population")
        
        best_acc, best_eff = max(a.accuracy for a in pop), min(a.flops for a in pop)
        objs = ['accuracy', 'flops'] + (['params'] if self.sc.params_weight > 0 else []) + (['latency'] if self.sc.latency_weight > 0 else [])
        
        for gen in range(self.sc.generations):
            self.generation = gen + 1
            if verbose: print(f"\n{'='*70}\nGeneration {self.generation}/{self.sc.generations}\n{'='*70}")
            
            offspring = self.evaluate_population(self.generate_offspring(pop), train_loader, val_loader)
            guided = self.evaluate_population(self.bayesian_guidance(offspring), train_loader, val_loader)
            combined = pop + [a for a in offspring if a.is_valid and a.accuracy > 0] + [a for a in guided if a.is_valid and a.accuracy > 0]
            pop = NSGAII.select_population(combined, self.sc.population_size, objs)
            
            curr_best = max(a.accuracy for a in pop)
            if curr_best > best_acc:
                best_acc, self.best_arch = curr_best, max(pop, key=lambda x: x.accuracy)
            best_eff = min(best_eff, min(a.flops for a in pop))
            
            fronts = NSGAII.non_dominated_sort(pop, objs)
            self.history.append({'generation': self.generation, 'best_accuracy': best_acc, 'best_efficiency': best_eff,
                                'avg_accuracy': np.mean([a.accuracy for a in pop]), 'pareto_size': len(fronts[0]) if fronts else 0})
        
        pareto = NSGAII.non_dominated_sort(pop, objs)[0] if pop else []
        print(f"\n{'='*70}\nSearch Complete!\n{'='*70}")
        print(f"Pareto Front: {len(pareto)} architectures\nBest Accuracy: {best_acc:.4f}\nBest Efficiency: {best_eff:.1f}M FLOPs\n{'='*70}\n")
        return pareto
    
    def get_best_architecture(self, pareto, preference: str = 'balanced'):
        if not pareto: return self.best_arch
        if preference == 'accuracy': return max(pareto, key=lambda x: x.accuracy)
        if preference == 'efficiency': return min(pareto, key=lambda x: x.flops)
        if preference == 'params': return min(pareto, key=lambda x: x.params)
        # balanced
        accs, flops = [a.accuracy for a in pareto], [a.flops for a in pareto]
        acc_range, flop_range = max(accs) - min(accs) + 1e-6, max(flops) - min(flops) + 1e-6
        scores = [(a.accuracy - min(accs))/acc_range + (max(flops) - a.flops)/flop_range for a in pareto]
        return pareto[np.argmax(scores)]
