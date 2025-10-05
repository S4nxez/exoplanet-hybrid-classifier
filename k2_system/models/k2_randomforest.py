#!/usr/bin/env python3
"""
🌲 K2 RANDOMFOREST MODEL
========================
Modelo RandomForest optimizado para casos seguros con alta precisión.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import logging
from pathlib import Path

from config.k2_config import RFConfig, K2Config, LogConfig, EvalConfig

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class K2RandomForest:
    """
    Modelo RandomForest especializado para casos seguros.
    Optimizado para alta precisión y mínimos falsos positivos.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RFConfig.N_ESTIMATORS,
            max_depth=RFConfig.MAX_DEPTH,
            min_samples_split=RFConfig.MIN_SAMPLES_SPLIT,
            min_samples_leaf=RFConfig.MIN_SAMPLES_LEAF,
            max_features=RFConfig.MAX_FEATURES,
            class_weight=RFConfig.CLASS_WEIGHT,
            bootstrap=RFConfig.BOOTSTRAP,
            oob_score=RFConfig.OOB_SCORE,
            n_jobs=RFConfig.N_JOBS,
            random_state=K2Config.RANDOM_SEED
        )

        self.scaler = StandardScaler()
        self.feature_selector = RFE(
            estimator=RandomForestClassifier(n_estimators=50, random_state=K2Config.RANDOM_SEED),
            n_features_to_select=K2Config.N_FEATURES_SELECTED
        )
        self.smote = BorderlineSMOTE(
            sampling_strategy=K2Config.SMOTE_STRATEGY,
            k_neighbors=K2Config.SMOTE_K_NEIGHBORS,
            random_state=K2Config.RANDOM_SEED
        )

        self.is_fitted = False
        self.optimal_threshold = RFConfig.PREDICTION_THRESHOLD

    def preprocess_data(self, X, y=None, fit_preprocessors=False):
        """Preprocesa datos con scaling, selección de características y SMOTE"""

        if fit_preprocessors:
            # Ajustar preprocessadores
            X_scaled = self.scaler.fit_transform(X)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)

            if y is not None:
                # Aplicar SMOTE solo si tenemos etiquetas
                X_resampled, y_resampled = self.smote.fit_resample(X_selected, y)
                return X_resampled, y_resampled
            else:
                return X_selected, y
        else:
            # Solo transformar
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            return X_selected, y

    def optimize_threshold(self, X_val, y_val):
        """Optimiza threshold para máxima precisión"""
        # Obtener probabilidades
        y_proba = self.model.predict_proba(X_val)[:, 1]

        # Calcular curva precision-recall
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # Encontrar threshold que maximiza precisión manteniendo recall > 0.8
        valid_indices = recalls >= 0.8
        if np.any(valid_indices):
            valid_precisions = precisions[valid_indices]
            valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds es 1 elemento menor

            if len(valid_thresholds) > 0:
                best_idx = np.argmax(valid_precisions[:-1])  # Excluir último elemento
                self.optimal_threshold = valid_thresholds[best_idx]
            else:
                self.optimal_threshold = RFConfig.PREDICTION_THRESHOLD
        else:
            self.optimal_threshold = RFConfig.PREDICTION_THRESHOLD

        logger.info(f"🎯 Threshold optimizado: {self.optimal_threshold:.3f}")

    def fit(self, X, y, validation_split=0.2):
        """Entrena el modelo RandomForest"""
        logger.info("🌲 Iniciando entrenamiento RandomForest (casos seguros)")

        # División inicial
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split,
            random_state=K2Config.RANDOM_SEED,
            stratify=y
        )

        # Preprocesamiento
        X_train_processed, y_train_processed = self.preprocess_data(
            X_train, y_train, fit_preprocessors=True
        )
        X_val_processed, _ = self.preprocess_data(X_val, fit_preprocessors=False)

        # Entrenamiento
        logger.info(f"📊 Datos de entrenamiento: {X_train_processed.shape}")
        logger.info(f"📊 Distribución después de SMOTE: {np.bincount(y_train_processed)}")

        self.model.fit(X_train_processed, y_train_processed)

        # Optimizar threshold
        self.optimize_threshold(X_val_processed, y_val)

        self.is_fitted = True

        # Evaluación
        y_pred = self.predict(X_val)
        self._log_performance(y_val, y_pred, "Validación")

        return self

    def predict(self, X):
        """Predicciones con threshold optimizado"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_processed, _ = self.preprocess_data(X, fit_preprocessors=False)
        y_proba = self.model.predict_proba(X_processed)[:, 1]

        return (y_proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        """Probabilidades de predicción"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_processed, _ = self.preprocess_data(X, fit_preprocessors=False)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self, feature_names=None):
        """Retorna importancia de características"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        # Obtener características seleccionadas
        selected_features = self.feature_selector.get_support()
        selected_names = None

        if feature_names is not None:
            selected_names = [feature_names[i] for i, selected in enumerate(selected_features) if selected]

        importance = self.model.feature_importances_

        if selected_names:
            return dict(zip(selected_names, importance))
        else:
            return importance

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

    def evaluate_safe_cases(self, X_test, y_test):
        """Evaluación específica para casos seguros"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        # Métricas básicas
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Análisis de confianza
        high_confidence_mask = y_proba >= 0.8
        high_conf_precision = None

        if np.any(high_confidence_mask):
            y_test_hc = y_test[high_confidence_mask]
            y_pred_hc = y_pred[high_confidence_mask]

            if len(y_pred_hc) > 0:
                cm_hc = confusion_matrix(y_test_hc, y_pred_hc)
                if cm_hc.size == 4:  # 2x2 matrix
                    tn_hc, fp_hc, fn_hc, tp_hc = cm_hc.ravel()
                    high_conf_precision = tp_hc / (tp_hc + fp_hc) if (tp_hc + fp_hc) > 0 else 0

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positives': fp,
            'true_positives': tp,
            'confusion_matrix': cm.tolist(),
            'threshold_used': self.optimal_threshold,
            'high_confidence_cases': np.sum(high_confidence_mask),
            'high_confidence_precision': high_conf_precision,
            'meets_precision_target': precision >= EvalConfig.TARGET_PRECISION,
            'meets_fp_target': fp <= EvalConfig.MAX_FALSE_POSITIVES
        }

        return results

    def save_model(self, base_path):
        """Guarda el modelo completo"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar componentes
        joblib.dump(self.model, base_path / "k2_randomforest_model.pkl")
        joblib.dump(self.scaler, base_path / "k2_rf_scaler.pkl")
        joblib.dump(self.feature_selector, base_path / "k2_rf_feature_selector.pkl")
        joblib.dump(self.smote, base_path / "k2_rf_smote.pkl")

        # Guardar metadatos
        metadata = {
            'optimal_threshold': self.optimal_threshold,
            'model_params': self.model.get_params(),
            'n_features_selected': K2Config.N_FEATURES_SELECTED
        }
        joblib.dump(metadata, base_path / "k2_rf_metadata.pkl")

        logger.info(f"✅ Modelo RandomForest guardado en {base_path}")

    def load_model(self, base_path):
        """Carga el modelo completo"""
        base_path = Path(base_path)

        # Cargar componentes
        self.model = joblib.load(base_path / "k2_randomforest_model.pkl")
        self.scaler = joblib.load(base_path / "k2_rf_scaler.pkl")
        self.feature_selector = joblib.load(base_path / "k2_rf_feature_selector.pkl")
        self.smote = joblib.load(base_path / "k2_rf_smote.pkl")

        # Cargar metadatos
        metadata = joblib.load(base_path / "k2_rf_metadata.pkl")
        self.optimal_threshold = metadata['optimal_threshold']

        self.is_fitted = True
        logger.info(f"✅ Modelo RandomForest cargado desde {base_path}")

if __name__ == "__main__":
    print("🌲 K2 RandomForest Model")
    print("Especializado en casos seguros con alta precisión")
    print(f"Configuración: {RFConfig.N_ESTIMATORS} estimators, depth={RFConfig.MAX_DEPTH}")
    print(f"Target: Precisión ≥ {EvalConfig.TARGET_PRECISION}, FP ≤ {EvalConfig.MAX_FALSE_POSITIVES}")