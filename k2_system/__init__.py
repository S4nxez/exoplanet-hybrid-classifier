#!/usr/bin/env python3
"""Public exports for the K2 package."""

__version__ = "1.0.0"
__author__ = "K2 Research Team"
__description__ = "Sistema ensemble para deteccion de exoplanetas K2"

from .models.k2_ensemble import K2EnsembleSystem
from .utils.data_utils import K2DataLoader, load_k2_sample_data
from .config.k2_config import K2Config, validate_config

try:
    from .utils.visualization_utils import K2Visualizer, K2Reporter
except Exception:  # Optional dependency (plotly) may be missing in minimal envs.
    K2Visualizer = None
    K2Reporter = None

__all__ = [
    "K2EnsembleSystem",
    "K2DataLoader",
    "K2Config",
    "validate_config",
    "load_k2_sample_data",
]

if K2Visualizer is not None:
	__all__.append("K2Visualizer")
if K2Reporter is not None:
	__all__.append("K2Reporter")