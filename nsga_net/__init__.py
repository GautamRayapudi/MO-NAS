"""
NSGA-Net: Multi-Objective Neural Architecture Search

A production-ready framework for automatically discovering optimal neural network
architectures using NSGA-II multi-objective optimization.
"""

# Core classes
from .core import (
    DatasetConfig,
    SearchConfig,
    LayerConfig,
    ArchitectureConfig,
    Architecture,
    UniversalModelBuilder,
)

# Algorithms
from .algorithms import (
    NSGAII,
    GeneticOperations,
    BayesianGuidance,
)

# Evaluation
from .evaluation import (
    ZeroCostProxy,
    Trainer,
)

# Search
from .search import (
    SearchSpace,
    ImageSearchSpace,
    TextSearchSpace,
    SequenceSearchSpace,
    TabularSearchSpace,
    create_search_space,
    NSGANet,
)

# Utils
from .utils import (
    ResultsAnalyzer,
)

__version__ = "1.0.0"
__author__ = "NSGA-Net Team"

__all__ = [
    # Core
    'DatasetConfig', 'SearchConfig', 'LayerConfig', 'ArchitectureConfig', 
    'Architecture', 'UniversalModelBuilder',
    # Algorithms
    'NSGAII', 'GeneticOperations', 'BayesianGuidance',
    # Evaluation
    'ZeroCostProxy', 'Trainer',
    # Search
    'SearchSpace', 'ImageSearchSpace', 'TextSearchSpace', 'SequenceSearchSpace',
    'TabularSearchSpace', 'create_search_space', 'NSGANet',
    # Utils
    'ResultsAnalyzer',
]
