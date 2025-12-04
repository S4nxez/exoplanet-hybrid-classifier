"""Dynamic expert selector for the TOI mission."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

from ..utils.config import TOIFeatures, TOIGatingThresholds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Thresholds:
    confidence_gap: float
    minimum_confidence: float
    complexity_threshold: float


class TOIDirector:
    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.tf_model = None
        self.tf_scaler = None
        self.is_trained = False
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        self.feature_index = {name: idx for idx, name in enumerate(TOIFeatures.CORE)}
        cfg = TOIGatingThresholds()
        self.thresholds = _Thresholds(cfg.confidence_gap, cfg.minimum_confidence, cfg.complexity_threshold)

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _complexity_score(self, X: np.ndarray) -> np.ndarray:
        period = np.clip(X[:, self.feature_index['pl_orbper']], 0, 1000)
        duration = np.clip(X[:, self.feature_index['pl_trandurh']], 0, 20)
        depth = np.clip(X[:, self.feature_index['pl_trandep']], 0, 2000)
        magnitude = np.clip(X[:, self.feature_index['st_tmag']], 5, 18)
        teff = np.clip(X[:, self.feature_index['st_teff']], 2500, 9000)

        period_score = period / 1000
        duration_score = duration / 20
        depth_score = 1 - depth / 2000
        mag_score = (magnitude - 5) / 13
        teff_score = (teff - 2500) / (9000 - 2500)

        return 0.3 * period_score + 0.2 * duration_score + 0.2 * depth_score + 0.15 * mag_score + 0.15 * teff_score

    def _model_confidences(self, X: np.ndarray):
        X_rf = self.rf_scaler.transform(X)
        X_tf = self.tf_scaler.transform(X)

        rf_probs = self.rf_model.predict_proba(X_rf)[:, 1]
        tf_probs = self.tf_model.predict(X_tf, verbose=0).flatten()

        rf_conf = np.abs(rf_probs - 0.5)
        tf_conf = np.abs(tf_probs - 0.5)
        return rf_probs, tf_probs, rf_conf, tf_conf

    def select_expert(self, X):
        X = self._ensure_2d(X)
        complexity = self._complexity_score(X)
        _, _, rf_conf, tf_conf = self._model_confidences(X)

        selections = []
        for comp, rf_c, tf_c in zip(complexity, rf_conf, tf_conf):
            max_conf = max(rf_c, tf_c)
            if max_conf >= self.thresholds.minimum_confidence:
                diff = rf_c - tf_c
                if diff > self.thresholds.confidence_gap:
                    selections.append('RandomForest')
                    continue
                if -diff > self.thresholds.confidence_gap:
                    selections.append('TensorFlow')
                    continue

            # fallback on complexity heuristics
            if comp <= self.thresholds.complexity_threshold:
                selections.append('RandomForest')
            else:
                selections.append('TensorFlow')

        if len(selections) == 1:
            return selections[0]
        return np.array(selections)

    def predict(self, X, return_model_choice=False):
        if not self.is_trained:
            raise ValueError("Director not configured")

        X = self._ensure_2d(X)
        selections = self.select_expert(X)
        if isinstance(selections, str):
            selections = np.array([selections])

        preds = np.zeros(X.shape[0], dtype=int)
        rf_mask = selections == 'RandomForest'
        tf_mask = selections == 'TensorFlow'

        if np.any(rf_mask):
            X_rf = self.rf_scaler.transform(X[rf_mask])
            preds[rf_mask] = self.rf_model.predict(X_rf)

        if np.any(tf_mask):
            X_tf = self.tf_scaler.transform(X[tf_mask])
            tf_preds = self.tf_model.predict(X_tf, verbose=0).flatten()
            preds[tf_mask] = (tf_preds > 0.5).astype(int)

        for choice in selections:
            self.decision_stats[choice] += 1

        if return_model_choice:
            selection_output = selections[0] if len(selections) == 1 else selections.tolist()
            return preds, selection_output
        return preds

    def configure(self, rf_model, rf_scaler, tf_model, tf_scaler):
        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.tf_model = tf_model
        self.tf_scaler = tf_scaler
        self.is_trained = True
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        logger.info("TOI Director configured with dynamic RF/TF gating")

    def get_stats(self):
        total = sum(self.decision_stats.values())
        if total == 0:
            return {"RandomForest": 0, "TensorFlow": 0, "total": 0}

        return {
            "RandomForest": self.decision_stats['RandomForest'],
            "TensorFlow": self.decision_stats['TensorFlow'],
            "total": total,
            "rf_percentage": (self.decision_stats['RandomForest'] / total) * 100,
            "tf_percentage": (self.decision_stats['TensorFlow'] / total) * 100,
        }

    def reset_stats(self):
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
