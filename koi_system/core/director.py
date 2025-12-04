import numpy as np
import logging
from dataclasses import dataclass

from koi_system.config.koi_config import KOIConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatingThresholds:
    confidence_gap: float = 0.15
    minimum_confidence: float = 0.05
    complexity_threshold: float = 0.45


class KOIDirector:
    """Dynamic selector between RandomForest and TensorFlow for the KOI mission."""

    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.tf_model = None
        self.tf_scaler = None
        self.is_trained = False
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        self.feature_index = {name: idx for idx, name in enumerate(KOIConfig.FEATURES)}
        self.thresholds = GatingThresholds()

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _complexity_score(self, X):
        period = np.clip(X[:, self.feature_index['koi_period']], 0, 200)
        duration = np.clip(X[:, self.feature_index['koi_duration']], 0, 20)
        depth = np.clip(X[:, self.feature_index['koi_depth']], 0, 1)
        teq = np.clip(X[:, self.feature_index['koi_teq']], 0, 2500)

        period_score = period / 200  # long orbital periods tend to be harder cases
        duration_score = duration / 20
        depth_score = 1 - depth / 1.0  # shallow transits => complex cases
        teq_score = teq / 2500

        return 0.35 * period_score + 0.25 * duration_score + 0.2 * depth_score + 0.2 * teq_score

    def _model_confidences(self, X):
        X_rf = self.rf_scaler.transform(X)
        X_tf = self.tf_scaler.transform(X)

        rf_probs = self.rf_model.predict_proba(X_rf)[:, 1]
        tf_probs = self.tf_model.predict(X_tf, verbose=0).flatten()

        rf_conf = np.abs(rf_probs - 0.5)
        tf_conf = np.abs(tf_probs - 0.5)

        return rf_probs, tf_probs, rf_conf, tf_conf

    def select_expert(self, X):
        """Choose which model (RF or TF) to use for each sample."""
        X = self._ensure_2d(X)
        complexity = self._complexity_score(X)
        _, _, rf_conf, tf_conf = self._model_confidences(X)

        selections = []
        for comp, rf_c, tf_c in zip(complexity, rf_conf, tf_conf):
            max_conf = max(rf_c, tf_c)
            if max_conf >= self.thresholds.minimum_confidence:
                gap = rf_c - tf_c
                if gap > self.thresholds.confidence_gap:
                    selections.append('RandomForest')
                    continue
                if -gap > self.thresholds.confidence_gap:
                    selections.append('TensorFlow')
                    continue

            # Complexity heuristic: easy cases go to RF, harder ones to TF
            if comp <= self.thresholds.complexity_threshold:
                selections.append('RandomForest')
            else:
                selections.append('TensorFlow')

        if len(selections) == 1:
            return selections[0]
        return np.array(selections)

    def predict(self, X, return_model_choice=False):
        if not self.is_trained:
            raise ValueError("Director no configurado")

        X = self._ensure_2d(X)
        selections = self.select_expert(X)
        if isinstance(selections, str):
            selections = np.array([selections])

        predictions = np.zeros(X.shape[0], dtype=int)

        rf_mask = selections == 'RandomForest'
        tf_mask = selections == 'TensorFlow'

        if np.any(rf_mask):
            X_rf = self.rf_scaler.transform(X[rf_mask])
            predictions[rf_mask] = self.rf_model.predict(X_rf)

        if np.any(tf_mask):
            X_tf = self.tf_scaler.transform(X[tf_mask])
            tf_preds = self.tf_model.predict(X_tf, verbose=0).flatten()
            predictions[tf_mask] = (tf_preds > 0.5).astype(int)

        for choice in selections:
            self.decision_stats[choice] += 1

        if return_model_choice:
            model_output = selections[0] if len(selections) == 1 else selections.tolist()
            return predictions, model_output

        return predictions

    def configure(self, rf_model, rf_scaler, tf_model, tf_scaler):
        """Configure the director with trained models."""
        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.tf_model = tf_model
        self.tf_scaler = tf_scaler
        self.is_trained = True
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        logger.info("KOI Director configured with dynamic RF/TF gating")

    def get_stats(self):
        """Return usage statistics for each model."""
        total = sum(self.decision_stats.values())
        if total == 0:
            return {"RandomForest": 0, "TensorFlow": 0, "total": 0}

        return {
            "RandomForest": self.decision_stats['RandomForest'],
            "TensorFlow": self.decision_stats['TensorFlow'],
            "total": total,
            "rf_percentage": (self.decision_stats['RandomForest'] / total) * 100,
            "tf_percentage": (self.decision_stats['TensorFlow'] / total) * 100
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
