#!/usr/bin/env python3
"""
🧠 K2 TENSORFLOW MODEL
======================
Modelo TensorFlow especializado para casos complejos y ambiguos.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from pathlib import Path

from config.k2_config import TFConfig, K2Config, LogConfig, EvalConfig

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class K2TensorFlow:
    """
    Modelo TensorFlow especializado para casos complejos.
    Optimizado para alta sensibilidad y detección de patrones sutiles.
    """

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # Más robusto para outliers
        self.feature_selector = SelectKBest(
            score_func=f_classif,
            k=K2Config.N_FEATURES_SELECTED
        )
        self.is_fitted = False
        self.optimal_threshold = TFConfig.PREDICTION_THRESHOLD

    def _create_model(self, input_dim):
        """Crea arquitectura del modelo TensorFlow"""
        # Configurar semilla para reproducibilidad
        tf.random.set_seed(K2Config.RANDOM_SEED)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),

            # Primera capa oculta
            tf.keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[0],
                activation=TFConfig.ACTIVATION,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Segunda capa oculta
            tf.keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[1],
                activation=TFConfig.ACTIVATION,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Tercera capa oculta
            tf.keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[2],
                activation=TFConfig.ACTIVATION,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Capa de salida
            tf.keras.layers.Dense(1, activation=TFConfig.OUTPUT_ACTIVATION)
        ])

        # Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(TFConfig.LEARNING_RATE),
            loss=TFConfig.LOSS,
            metrics=TFConfig.METRICS
        )

        return model

    def preprocess_data(self, X, y=None, fit_preprocessors=False):
        """Preprocesa datos con scaling robusto y selección de características"""

        if fit_preprocessors:
            # Ajustar preprocessadores
            X_scaled = self.scaler.fit_transform(X)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            return X_selected, y
        else:
            # Solo transformar
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            return X_selected, y

    def _create_balanced_weights(self, y):
        """Crea pesos balanceados para clases"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)

        # Calcular pesos inversamente proporcionales a la frecuencia
        class_weights = {}
        for i, count in enumerate(counts):
            class_weights[unique[i]] = total / (len(unique) * count)

        return class_weights

    def optimize_threshold(self, X_val, y_val):
        """Optimiza threshold para balance precisión-recall"""
        # Obtener probabilidades
        y_proba = self.model.predict(X_val).flatten()

        # Probar diferentes thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = TFConfig.PREDICTION_THRESHOLD

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calcular métricas
            cm = confusion_matrix(y_val, y_pred)
            if cm.size == 4:  # Asegurar matriz 2x2
                tn, fp, fn, tp = cm.ravel()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Penalizar si recall es muy bajo
                if recall >= 0.85 and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        self.optimal_threshold = best_threshold
        logger.info(f"🎯 Threshold optimizado: {self.optimal_threshold:.3f} (F1: {best_f1:.3f})")

    def fit(self, X, y, validation_split=0.2):
        """Entrena el modelo TensorFlow"""
        logger.info("🧠 Iniciando entrenamiento TensorFlow (casos complejos)")

        # División inicial
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split,
            random_state=K2Config.RANDOM_SEED,
            stratify=y
        )

        # Preprocesamiento
        X_train_processed, _ = self.preprocess_data(
            X_train, y_train, fit_preprocessors=True
        )
        X_val_processed, _ = self.preprocess_data(X_val, fit_preprocessors=False)

        logger.info(f"📊 Datos de entrenamiento: {X_train_processed.shape}")
        logger.info(f"📊 Distribución de clases: {np.bincount(y_train)}")

        # Crear modelo
        self.model = self._create_model(X_train_processed.shape[1])

        # Pesos de clase para balanceo
        class_weights = self._create_balanced_weights(y_train)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=TFConfig.PATIENCE,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_tf_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        # Entrenamiento
        history = self.model.fit(
            X_train_processed, y_train,
            validation_data=(X_val_processed, y_val),
            epochs=TFConfig.EPOCHS,
            batch_size=TFConfig.BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=LogConfig.VERBOSE
        )

        # Optimizar threshold
        self.optimize_threshold(X_val_processed, y_val)

        self.is_fitted = True

        # Evaluación
        y_pred = self.predict(X_val)
        self._log_performance(y_val, y_pred, "Validación")

        return history

    def predict(self, X):
        """Predicciones con threshold optimizado"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_processed, _ = self.preprocess_data(X, fit_preprocessors=False)
        y_proba = self.model.predict(X_processed).flatten()

        return (y_proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        """Probabilidades de predicción"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_processed, _ = self.preprocess_data(X, fit_preprocessors=False)
        y_proba = self.model.predict(X_processed).flatten()

        # Retornar formato compatible con sklearn
        return np.column_stack([1 - y_proba, y_proba])

    def get_uncertainty_score(self, X):
        """Calcula score de incertidumbre para cada predicción"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        y_proba = self.predict_proba(X)[:, 1]

        # Calcular incertidumbre como distancia del 0.5
        uncertainty = 1 - 2 * np.abs(y_proba - 0.5)

        return uncertainty

    def _log_performance(self, y_true, y_pred, dataset_name):
        """Registra métricas de rendimiento"""
        logger.info(f"📈 Rendimiento en {dataset_name}:")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info(f"🎯 Precisión: {precision:.3f}")
        logger.info(f"🎯 Recall: {recall:.3f}")
        logger.info(f"❌ Falsos Positivos: {fp}")

    def evaluate_complex_cases(self, X_test, y_test):
        """Evaluación específica para casos complejos"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        uncertainty = self.get_uncertainty_score(X_test)

        # Métricas básicas
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # AUC-ROC
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0

        # Análisis de incertidumbre
        high_uncertainty_mask = uncertainty >= 0.3
        avg_uncertainty = np.mean(uncertainty)

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'false_positives': fp,
            'true_positives': tp,
            'confusion_matrix': cm.tolist(),
            'threshold_used': self.optimal_threshold,
            'average_uncertainty': avg_uncertainty,
            'high_uncertainty_cases': np.sum(high_uncertainty_mask),
            'meets_recall_target': recall >= EvalConfig.TARGET_RECALL,
            'meets_f1_target': f1 >= EvalConfig.TARGET_F1
        }

        return results

    def save_model(self, base_path):
        """Guarda el modelo completo"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar modelo TensorFlow
        self.model.save(base_path / "k2_tensorflow_model.h5")

        # Guardar preprocessadores
        joblib.dump(self.scaler, base_path / "k2_tf_scaler.pkl")
        joblib.dump(self.feature_selector, base_path / "k2_tf_feature_selector.pkl")

        # Guardar metadatos
        metadata = {
            'optimal_threshold': self.optimal_threshold,
            'architecture': TFConfig.HIDDEN_LAYERS,
            'n_features_selected': K2Config.N_FEATURES_SELECTED
        }
        joblib.dump(metadata, base_path / "k2_tf_metadata.pkl")

        logger.info(f"✅ Modelo TensorFlow guardado en {base_path}")

    def load_model(self, base_path):
        """Carga el modelo completo"""
        base_path = Path(base_path)

        # Cargar modelo TensorFlow
        self.model = tf.keras.models.load_model(base_path / "k2_tensorflow_model.h5")

        # Cargar preprocessadores
        self.scaler = joblib.load(base_path / "k2_tf_scaler.pkl")
        self.feature_selector = joblib.load(base_path / "k2_tf_feature_selector.pkl")

        # Cargar metadatos
        metadata = joblib.load(base_path / "k2_tf_metadata.pkl")
        self.optimal_threshold = metadata['optimal_threshold']

        self.is_fitted = True
        logger.info(f"✅ Modelo TensorFlow cargado desde {base_path}")

if __name__ == "__main__":
    print("🧠 K2 TensorFlow Model")
    print("Especializado en casos complejos y ambiguos")
    print(f"Arquitectura: {TFConfig.HIDDEN_LAYERS}")
    print(f"Target: Recall ≥ {EvalConfig.TARGET_RECALL}, F1 ≥ {EvalConfig.TARGET_F1}")