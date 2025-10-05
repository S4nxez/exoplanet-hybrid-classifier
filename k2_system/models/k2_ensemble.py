#!/usr/bin/env python3
"""
🎯 K2 ENSEMBLE SYSTEM
====================
Sistema ensemble que combina Director + RandomForest + TensorFlow
para detección optimizada de exoplanetas K2.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
import time

from models.k2_director import K2Director
from models.k2_randomforest import K2RandomForest
from models.k2_tensorflow import K2TensorFlow
from config.k2_config import K2Config, EvalConfig, LogConfig

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class K2EnsembleSystem:
    """
    Sistema Ensemble K2 que combina tres modelos especializados:
    1. Director: Decide qué modelo usar para cada caso
    2. RandomForest: Alta precisión para casos seguros
    3. TensorFlow: Alta sensibilidad para casos complejos
    """

    def __init__(self):
        self.director = K2Director()
        self.rf_model = K2RandomForest()
        self.tf_model = K2TensorFlow()

        self.is_fitted = False
        self.training_stats = {}

    def fit(self, X, y, validation_split=0.2):
        """Entrena todo el sistema ensemble"""
        logger.info("🚀 Iniciando entrenamiento del Sistema K2 Ensemble")
        start_time = time.time()

        # División de datos para ensemble
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split,
            random_state=K2Config.RANDOM_SEED,
            stratify=y
        )

        logger.info(f"📊 Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"📊 Conjunto de validación: {X_val.shape[0]} muestras")
        logger.info(f"📊 Distribución de clases: {np.bincount(y_train)}")

        # 1. Entrenar Director
        logger.info("\n🧠 FASE 1: Entrenamiento del Director")
        director_start = time.time()
        self.director.fit(X_train, y_train, validation_split=0.2)
        director_time = time.time() - director_start

        # 2. Obtener asignaciones del Director para datos de entrenamiento
        director_assignments, director_confidences = self.director.predict_model_type(X_train)

        # Separar datos según asignaciones del Director
        rf_mask = director_assignments == 0
        tf_mask = director_assignments == 1

        X_train_rf = X_train[rf_mask]
        y_train_rf = y_train[rf_mask]
        X_train_tf = X_train[tf_mask]
        y_train_tf = y_train[tf_mask]

        logger.info(f"📈 Casos asignados a RandomForest: {len(X_train_rf)}")
        logger.info(f"📈 Casos asignados a TensorFlow: {len(X_train_tf)}")

        # 3. Entrenar RandomForest (si hay casos asignados)
        rf_time = 0
        if len(X_train_rf) > 0:
            logger.info("\n🌲 FASE 2: Entrenamiento RandomForest")
            rf_start = time.time()
            self.rf_model.fit(X_train_rf, y_train_rf, validation_split=0.2)
            rf_time = time.time() - rf_start
        else:
            logger.warning("⚠️ No hay casos asignados a RandomForest")

        # 4. Entrenar TensorFlow (si hay casos asignados)
        tf_time = 0
        if len(X_train_tf) > 0:
            logger.info("\n🧠 FASE 3: Entrenamiento TensorFlow")
            tf_start = time.time()
            self.tf_model.fit(X_train_tf, y_train_tf, validation_split=0.2)
            tf_time = time.time() - tf_start
        else:
            logger.warning("⚠️ No hay casos asignados a TensorFlow")

        self.is_fitted = True
        total_time = time.time() - start_time

        # Guardar estadísticas de entrenamiento
        self.training_stats = {
            'total_training_time': total_time,
            'director_time': director_time,
            'rf_time': rf_time,
            'tf_time': tf_time,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'rf_assigned_samples': len(X_train_rf),
            'tf_assigned_samples': len(X_train_tf),
            'class_distribution': np.bincount(y_train).tolist()
        }

        # Evaluación en conjunto de validación
        logger.info("\n📊 EVALUACIÓN DEL SISTEMA ENSEMBLE")
        self._evaluate_system(X_val, y_val)

        logger.info(f"⏱️ Tiempo total de entrenamiento: {total_time:.2f} segundos")

        return self

    def predict(self, X):
        """Predicciones del sistema ensemble"""
        if not self.is_fitted:
            raise ValueError("El sistema debe ser entrenado primero")

        # 1. Director decide qué modelo usar para cada caso
        model_assignments, confidences = self.director.predict_model_type(X)

        # 2. Separar casos según asignaciones
        rf_mask = model_assignments == 0
        tf_mask = model_assignments == 1

        # 3. Inicializar array de predicciones
        predictions = np.zeros(len(X), dtype=int)

        # 4. Predicciones con RandomForest
        if np.any(rf_mask) and self.rf_model.is_fitted:
            X_rf = X[rf_mask]
            predictions[rf_mask] = self.rf_model.predict(X_rf)

        # 5. Predicciones con TensorFlow
        if np.any(tf_mask) and self.tf_model.is_fitted:
            X_tf = X[tf_mask]
            predictions[tf_mask] = self.tf_model.predict(X_tf)

        return predictions

    def predict_with_details(self, X):
        """Predicciones detalladas con información del modelo usado"""
        if not self.is_fitted:
            raise ValueError("El sistema debe ser entrenado primero")

        # Director decide qué modelo usar
        model_assignments, director_confidences = self.director.predict_model_type(X)

        # Separar casos
        rf_mask = model_assignments == 0
        tf_mask = model_assignments == 1

        # Inicializar arrays
        predictions = np.zeros(len(X), dtype=int)
        probabilities = np.zeros(len(X))
        model_used = np.array(['unknown'] * len(X), dtype=object)
        prediction_confidences = np.zeros(len(X))

        # Predicciones RandomForest
        if np.any(rf_mask) and self.rf_model.is_fitted:
            X_rf = X[rf_mask]
            rf_pred = self.rf_model.predict(X_rf)
            rf_proba = self.rf_model.predict_proba(X_rf)[:, 1]

            predictions[rf_mask] = rf_pred
            probabilities[rf_mask] = rf_proba
            model_used[rf_mask] = 'RandomForest'
            prediction_confidences[rf_mask] = np.maximum(rf_proba, 1 - rf_proba)

        # Predicciones TensorFlow
        if np.any(tf_mask) and self.tf_model.is_fitted:
            X_tf = X[tf_mask]
            tf_pred = self.tf_model.predict(X_tf)
            tf_proba = self.tf_model.predict_proba(X_tf)[:, 1]

            predictions[tf_mask] = tf_pred
            probabilities[tf_mask] = tf_proba
            model_used[tf_mask] = 'TensorFlow'
            prediction_confidences[tf_mask] = np.maximum(tf_proba, 1 - tf_proba)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_used,
            'director_confidences': director_confidences,
            'prediction_confidences': prediction_confidences,
            'rf_cases': np.sum(rf_mask),
            'tf_cases': np.sum(tf_mask)
        }

    def _evaluate_system(self, X_val, y_val):
        """Evaluación completa del sistema"""
        predictions = self.predict(X_val)

        # Métricas generales
        logger.info("📈 Rendimiento General del Sistema:")
        print(classification_report(y_val, predictions))

        cm = confusion_matrix(y_val, predictions)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logger.info(f"🎯 Precisión Global: {precision:.3f}")
        logger.info(f"🎯 Recall Global: {recall:.3f}")
        logger.info(f"🎯 F1-Score Global: {f1:.3f}")
        logger.info(f"❌ Falsos Positivos: {fp}")
        logger.info(f"✅ Verdaderos Positivos: {tp}")

        # Análisis por modelo
        detailed_results = self.predict_with_details(X_val)

        rf_cases = detailed_results['rf_cases']
        tf_cases = detailed_results['tf_cases']
        total_cases = len(X_val)

        logger.info(f"\n📊 Distribución de casos:")
        logger.info(f"   RandomForest: {rf_cases} ({rf_cases/total_cases*100:.1f}%)")
        logger.info(f"   TensorFlow: {tf_cases} ({tf_cases/total_cases*100:.1f}%)")

        # Verificar objetivos
        meets_precision = precision >= EvalConfig.TARGET_PRECISION
        meets_recall = recall >= EvalConfig.TARGET_RECALL
        meets_f1 = f1 >= EvalConfig.TARGET_F1
        meets_fp = fp <= EvalConfig.MAX_FALSE_POSITIVES

        logger.info(f"\n🎯 Cumplimiento de Objetivos:")
        logger.info(f"   Precisión ≥ {EvalConfig.TARGET_PRECISION}: {'✅' if meets_precision else '❌'}")
        logger.info(f"   Recall ≥ {EvalConfig.TARGET_RECALL}: {'✅' if meets_recall else '❌'}")
        logger.info(f"   F1-Score ≥ {EvalConfig.TARGET_F1}: {'✅' if meets_f1 else '❌'}")
        logger.info(f"   FP ≤ {EvalConfig.MAX_FALSE_POSITIVES}: {'✅' if meets_fp else '❌'}")

    def get_system_summary(self):
        """Retorna resumen completo del sistema"""
        if not self.is_fitted:
            return {"error": "Sistema no entrenado"}

        summary = {
            'system_status': 'trained' if self.is_fitted else 'not_trained',
            'models': {
                'director': {
                    'fitted': self.director.is_fitted,
                    'architecture': 'TensorFlow',
                    'purpose': 'Clasificar casos en seguros vs complejos'
                },
                'randomforest': {
                    'fitted': self.rf_model.is_fitted,
                    'architecture': 'RandomForest',
                    'purpose': 'Alta precisión para casos seguros',
                    'n_estimators': self.rf_model.model.n_estimators if self.rf_model.is_fitted else None
                },
                'tensorflow': {
                    'fitted': self.tf_model.is_fitted,
                    'architecture': 'Neural Network',
                    'purpose': 'Alta sensibilidad para casos complejos'
                }
            },
            'training_stats': self.training_stats,
            'targets': {
                'precision': EvalConfig.TARGET_PRECISION,
                'recall': EvalConfig.TARGET_RECALL,
                'f1_score': EvalConfig.TARGET_F1,
                'max_false_positives': EvalConfig.MAX_FALSE_POSITIVES
            }
        }

        return summary

    def save_system(self, base_path):
        """Guarda todo el sistema ensemble"""
        if not self.is_fitted:
            raise ValueError("El sistema debe ser entrenado primero")

        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar cada modelo si está entrenado
        models_path = base_path / "models"
        models_path.mkdir(exist_ok=True)

        self.director.save_model(models_path / "director")

        if self.rf_model.is_fitted:
            self.rf_model.save_model(models_path / "randomforest")

        if self.tf_model.is_fitted:
            self.tf_model.save_model(models_path / "tensorflow")

        # Guardar metadatos del sistema
        system_metadata = {
            'ensemble_type': 'K2_Director_RF_TF',
            'training_stats': self.training_stats,
            'model_versions': {
                'director': '1.0',
                'randomforest': '1.0',
                'tensorflow': '1.0'
            }
        }

        joblib.dump(system_metadata, base_path / "ensemble_metadata.pkl")

        logger.info(f"✅ Sistema K2 Ensemble guardado en {base_path}")

    def load_system(self, base_path):
        """Carga todo el sistema ensemble"""
        base_path = Path(base_path)
        models_path = base_path / "models"

        # Cargar cada modelo si existe
        self.director.load_model(models_path / "director")

        if (models_path / "randomforest").exists():
            self.rf_model.load_model(models_path / "randomforest")

        if (models_path / "tensorflow").exists():
            self.tf_model.load_model(models_path / "tensorflow")

        # Cargar metadatos
        system_metadata = joblib.load(base_path / "ensemble_metadata.pkl")
        self.training_stats = system_metadata.get('training_stats', {})

        self.is_fitted = True
        logger.info(f"✅ Sistema K2 Ensemble cargado desde {base_path}")

if __name__ == "__main__":
    print("🎯 K2 Ensemble System")
    print("Director + RandomForest + TensorFlow")
    print("Optimizado para detección de exoplanetas K2")
    print(f"Targets: P≥{EvalConfig.TARGET_PRECISION}, R≥{EvalConfig.TARGET_RECALL}, F1≥{EvalConfig.TARGET_F1}")