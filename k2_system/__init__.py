#!/usr/bin/env python3
"""
__init__.py - K2 Final System Package
"""

__version__ = "1.0.0"
__author__ = "K2 Research Team"
__description__ = "Sistema ensemble para detección de exoplanetas K2"

# Importaciones principales
from .models.k2_ensemble import K2EnsembleSystem
from .utils.data_utils import K2DataLoader, load_k2_sample_data
from .utils.visualization_utils import K2Visualizer, K2Reporter
from .config.k2_config import K2Config, validate_config

__all__ = [
    'K2EnsembleSystem',
    'K2DataLoader',
    'K2Visualizer',
    'K2Reporter',
    'K2Config',
    'validate_config',
    'load_k2_sample_data'
]