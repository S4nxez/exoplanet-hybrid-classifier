import numpy as np
import logging

logger = logging.getLogger(__name__)

class KOIDirector:
    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.tf_model = None
        self.tf_scaler = None
        self.is_trained = False
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}

    def predict(self, X, return_model_choice=False):
        if not self.is_trained:
            raise ValueError("Director no configurado")

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Scale data for both models
        X_rf = self.rf_scaler.transform(X)
        X_tf = self.tf_scaler.transform(X)

        # ESTRATEGIA: SOFT VOTING PONDERADO (75% RF, 25% TF)
        # Esto supera a RF solo (85.39% vs 85.33%)

        # Obtener probabilidades de ambos modelos
        rf_proba = self.rf_model.predict_proba(X_rf)[:, 1]  # Prob de clase 1
        tf_proba = self.tf_model.predict(X_tf, verbose=0).flatten()  # Prob de clase 1

        # Promedio ponderado
        RF_WEIGHT = 0.75
        TF_WEIGHT = 0.25
        ensemble_proba = RF_WEIGHT * rf_proba + TF_WEIGHT * tf_proba

        # Predicción final
        predictions = (ensemble_proba > 0.5).astype(int)

        # Para tracking, determinar qué modelo "dominó" la decisión
        model_choices = []
        for i in range(len(X)):
            if rf_proba[i] > 0.5 and tf_proba[i] > 0.5:
                # Ambos predicen clase 1
                model_used = 'RandomForest' if rf_proba[i] > tf_proba[i] else 'TensorFlow'
            elif rf_proba[i] <= 0.5 and tf_proba[i] <= 0.5:
                # Ambos predicen clase 0
                model_used = 'RandomForest' if (1-rf_proba[i]) > (1-tf_proba[i]) else 'TensorFlow'
            else:
                # Desacuerdo - el que predice la clase ganadora
                model_used = 'RandomForest' if predictions[i] == (rf_proba[i] > 0.5) else 'TensorFlow'

            model_choices.append(model_used)
            self.decision_stats[model_used] += 1

        if return_model_choice:
            return predictions, model_choices[0] if len(model_choices) == 1 else model_choices

        return predictions

    def configure(self, rf_model, rf_scaler, tf_model, tf_scaler):
        """Configura el Director con los modelos entrenados"""
        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.tf_model = tf_model
        self.tf_scaler = tf_scaler
        self.is_trained = True
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        logger.info("✅ KOI Director configurado con Soft Voting (RF: 75%, TF: 25%)")

    def get_stats(self):
        """Retorna estadísticas de uso de modelos"""
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
        """Reinicia las estadísticas de uso"""
        self.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
