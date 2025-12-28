"""
Evaluation module - Proxies and Training
"""

from .proxies import ZeroCostProxy
from .trainer import Trainer

__all__ = [
    'ZeroCostProxy',
    'Trainer',
]
