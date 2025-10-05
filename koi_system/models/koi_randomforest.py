#!/usr/bin/env python3
"""
🌲 KOI RANDOMFOREST MODEL
=========================
Modelo RandomForest optimizado para casos simples/lineales del dataset KOI.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.koi_config import RFConfig, KOIConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KOIRandomForest:
    """
    Modelo RandomForest especializado para casos simples/lineales.
    Optimizado para datos tabulares del KOI dataset.
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
            n_jobs=RFConfig.N_JOBS,
            random_state=KOIConfig.RANDOM_SEED
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena el modelo RandomForest"""
        logger.info("🌲 Iniciando entrenamiento Random Forest")
        logger.info(f"   Muestras de entrenamiento: {len(X_train)}")

        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluar en train
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        logger.info(f"   ✅ Accuracy en train: {train_acc*100:.2f}%")

        # Evaluar en validación si existe
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_pred = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_val_pred)
            logger.info(f"   ✅ Accuracy en validation: {val_acc*100:.2f}%")

            # Reporte detallado
            logger.info("\n📊 Classification Report (Validation):")
            print(classification_report(y_val, y_val_pred,
                                      target_names=['No Confirmed', 'Confirmed']))

        logger.info("✅ Random Forest entrenado exitosamente")

        return self

    def predict(self, X):
        """Predice clases"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llamar a fit() primero.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predice probabilidades"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llamar a fit() primero.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """Obtiene importancia de features"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        return self.model.feature_importances_

    def save(self, model_dir):
        """Guarda el modelo"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / RFConfig.MODEL_FILE
        scaler_path = model_dir / RFConfig.SCALER_FILE

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"✅ Random Forest guardado en {model_dir}")

    def load(self, model_dir):
        """Carga el modelo"""
        model_dir = Path(model_dir)

        model_path = model_dir / RFConfig.MODEL_FILE
        scaler_path = model_dir / RFConfig.SCALER_FILE

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True

        logger.info(f"✅ Random Forest cargado desde {model_dir}")

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logger.info("="*50)
        logger.info("📊 EVALUACIÓN RANDOM FOREST")
        logger.info("="*50)
        logger.info(f"Accuracy: {acc*100:.2f}%")
        logger.info(f"\nMatriz de confusión:\n{cm}")
        logger.info("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred,
                                   target_names=['No Confirmed', 'Confirmed']))

        return {
            'accuracy': acc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
