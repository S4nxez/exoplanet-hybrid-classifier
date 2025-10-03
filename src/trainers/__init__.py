"""
TRAINERS MODULE
===============
Módulo de entrenadores para diferentes tipos de modelos
"""

from .base_trainer import BaseTrainer
from .total_trainer import TotalModelTrainer
from .partial_trainer import PartialModelTrainer
from .hybrid_trainer import HybridModelTrainer

__all__ = [
    'BaseTrainer',
    'TotalModelTrainer', 
    'PartialModelTrainer',
    'HybridModelTrainer'
]