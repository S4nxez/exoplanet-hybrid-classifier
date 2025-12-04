import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class K2GatingThresholds:
    confidence_gap: float = 0.12
    minimum_confidence: float = 0.05
    complexity_threshold: float = 0.5


K2_FEATURE_COLUMNS = [
    'pl_orbper',
    'pl_rade',
    'st_teff',
    'st_rad',
    'st_mass',
    'sy_dist',
    'pl_eqt',
    'pl_orbsmax'
]


class K2Director:
    """Dynamic gating director for the K2 mission."""

    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.tf_model = None
        self.tf_scaler = None
        self.is_trained = False
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        self.feature_index = {name: idx for idx, name in enumerate(K2_FEATURE_COLUMNS)}
        self.thresholds = K2GatingThresholds()

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _complexity_score(self, X):
        period = np.clip(X[:, self.feature_index['pl_orbper']], 0, 300)
        radius = np.clip(X[:, self.feature_index['pl_rade']], 0, 15)
        teff = np.clip(X[:, self.feature_index['st_teff']], 2000, 9000)
        distance = np.clip(X[:, self.feature_index['sy_dist']], 0, 1500)
        semi_major = np.clip(X[:, self.feature_index['pl_orbsmax']], 0, 5) if 'pl_orbsmax' in self.feature_index else np.zeros(len(period))

        period_score = period / 300
        radius_score = radius / 15
        teff_score = (teff - 2000) / (9000 - 2000)
        distance_score = distance / 1500
        sma_score = semi_major / 5

        return 0.35 * period_score + 0.2 * distance_score + 0.25 * sma_score + 0.1 * radius_score + 0.1 * teff_score

    def _model_confidences(self, X):
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
                gap = rf_c - tf_c
                if gap > self.thresholds.confidence_gap:
                    selections.append('RandomForest')
                    continue
                if -gap > self.thresholds.confidence_gap:
                    selections.append('TensorFlow')
                    continue

            selections.append('RandomForest' if comp <= self.thresholds.complexity_threshold else 'TensorFlow')

        return np.array(selections)

    def predict_with_details(self, X):
        if not self.is_trained:
            raise ValueError("Director no configurado")

        X = self._ensure_2d(X)
        selections = self.select_expert(X)

        rf_probs, tf_probs, _, _ = self._model_confidences(X)

        predictions = np.zeros(X.shape[0], dtype=int)
        probabilities = np.zeros(X.shape[0], dtype=float)
        confidences = np.zeros(X.shape[0], dtype=float)

        rf_mask = selections == 'RandomForest'
        tf_mask = selections == 'TensorFlow'

        if np.any(rf_mask):
            rf_pred = (rf_probs[rf_mask] > 0.5).astype(int)
            predictions[rf_mask] = rf_pred
            probabilities[rf_mask] = rf_probs[rf_mask]
            confidences[rf_mask] = np.maximum(rf_probs[rf_mask], 1 - rf_probs[rf_mask])

        if np.any(tf_mask):
            tf_pred = (tf_probs[tf_mask] > 0.5).astype(int)
            predictions[tf_mask] = tf_pred
            probabilities[tf_mask] = tf_probs[tf_mask]
            confidences[tf_mask] = np.maximum(tf_probs[tf_mask], 1 - tf_probs[tf_mask])

        for choice in selections:
            self.decision_stats[choice] += 1

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': selections,
            'prediction_confidences': confidences,
            'rf_cases': int(np.sum(selections == 'RandomForest')),
            'tf_cases': int(np.sum(selections == 'TensorFlow')),
        }

    def predict(self, X, return_model_choice=False):
        details = self.predict_with_details(X)
        predictions = details['predictions']
        selections = details['model_used']

        if return_model_choice:
            if len(selections) == 1:
                return predictions, selections[0]
            return predictions, selections.tolist()

        return predictions

    def configure(self, rf_model, rf_scaler, tf_model, tf_scaler):
        """Configure the director with trained models."""
        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.tf_model = tf_model
        self.tf_scaler = tf_scaler
        self.is_trained = True
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        logger.info("K2 Director configured with dynamic RF/TF gating")

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

