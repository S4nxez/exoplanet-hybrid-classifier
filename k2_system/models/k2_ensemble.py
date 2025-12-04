#!/usr/bin/env python3
"""Lightweight runtime that wires the K2 director with its experts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .k2_director import K2Director, K2_FEATURE_COLUMNS
from ..config.k2_config import LogConfig

logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Sequence[float]]
FrameLike = Union[pd.Series, pd.DataFrame]
SampleLike = Union[ArrayLike, FrameLike, dict]


class K2EnsembleSystem:
    """Runtime orchestrator that uses the gating director plus RF/TF experts."""

    def __init__(self, model_path: Union[str, Path, None] = None):
        base_dir = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else base_dir / "saved_models"

        self.director = K2Director()
        self.rf_model = None
        self.rf_scaler = None
        self.tf_model = None
        self.tf_scaler = None
        self.feature_columns = K2_FEATURE_COLUMNS
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, *relative_paths: str) -> Path:
        for rel in relative_paths:
            candidate = self.model_path / rel
            if candidate.exists():
                return candidate
        joined = ", ".join(relative_paths)
        raise FileNotFoundError(f"No se encontro el artefacto K2 requerido en [{joined}]")

    def _load_joblib(self, *relative_paths: str):
        path = self._resolve_path(*relative_paths)
        return joblib.load(path)

    def load_system(self):
        """Load persisted RF/TF experts and configure the director."""
        logger.info("Loading K2 hybrid models from %s", self.model_path)

        self.rf_model = self._load_joblib(
            "k2_rf_model.pkl",
            "models/randomforest/k2_randomforest_model.pkl",
        )
        self.rf_scaler = self._load_joblib(
            "k2_rf_scaler.pkl",
            "models/randomforest/k2_rf_scaler.pkl",
        )

        tf_model_path = self._resolve_path(
            "k2_tf_model.h5",
            "models/tensorflow/k2_tensorflow_model.h5",
        )
        self.tf_model = tf.keras.models.load_model(tf_model_path)
        self.tf_scaler = self._load_joblib(
            "k2_tf_scaler.pkl",
            "models/tensorflow/k2_tf_scaler.pkl",
        )

        self.director.configure(
            rf_model=self.rf_model,
            rf_scaler=self.rf_scaler,
            tf_model=self.tf_model,
            tf_scaler=self.tf_scaler,
        )
        self.is_loaded = True
        logger.info("K2 director configured with RF + TF experts")

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load_system()

    def _format_input(self, features: SampleLike) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            df = features.copy()
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features])
        elif isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            array = np.asarray(features)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            df = pd.DataFrame(array, columns=self.feature_columns)

        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas para K2: {missing}")

        aligned = df[self.feature_columns]
        return aligned.to_numpy(dtype=float, copy=True)

    def predict(self, X: SampleLike, return_model_choice: bool = False):
        self._ensure_loaded()
        matrix = self._format_input(X)
        details = self.director.predict_with_details(matrix)
        predictions = details['predictions']

        if return_model_choice:
            models = details['model_used']
            if len(models) == 1:
                return predictions, models[0]
            return predictions, models.tolist()
        return predictions

    def predict_with_details(self, X: SampleLike):
        self._ensure_loaded()
        matrix = self._format_input(X)
        return self.director.predict_with_details(matrix)


__all__ = ["K2EnsembleSystem"]
