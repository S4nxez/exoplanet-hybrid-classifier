#!/usr/bin/env python3
"""
🧠 KOI TENSORFLOW MODEL
=======================
Modelo TensorFlow MEJORADO para casos complejos del dataset KOI.
Arquitectura profunda con batch normalization y gradient clipping.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.koi_config import TFConfig, KOIConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KOITensorFlow:
    """
    Modelo TensorFlow MEJORADO para casos complejos.
    Arquitectura profunda: 256 → 128 → 64 → 32 con batch normalization.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.history = None

    def _create_model(self, input_dim):
        """Crea arquitectura MEJORADA del modelo TensorFlow"""
        # Configurar semilla
        tf.random.set_seed(KOIConfig.RANDOM_SEED)

        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),

            # Capa 1: 256 neuronas con Batch Normalization
            keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[0],
                activation=TFConfig.ACTIVATION,
                kernel_initializer='he_normal'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Capa 2: 128 neuronas con Batch Normalization
            keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[1],
                activation=TFConfig.ACTIVATION,
                kernel_initializer='he_normal'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Capa 3: 64 neuronas con Batch Normalization
            keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[2],
                activation=TFConfig.ACTIVATION,
                kernel_initializer='he_normal'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Capa 4: 32 neuronas
            keras.layers.Dense(
                TFConfig.HIDDEN_LAYERS[3],
                activation=TFConfig.ACTIVATION,
                kernel_initializer='he_normal'
            ),
            keras.layers.Dropout(TFConfig.DROPOUT_RATE),

            # Capa de salida
            keras.layers.Dense(1, activation=TFConfig.OUTPUT_ACTIVATION)
        ])

        # Compilar con gradient clipping
        optimizer = keras.optimizers.Adam(
            learning_rate=TFConfig.LEARNING_RATE,
            clipnorm=1.0  # Gradient clipping
        )

        model.compile(
            optimizer=optimizer,
            loss=TFConfig.LOSS,
            metrics=TFConfig.METRICS
        )

        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena el modelo TensorFlow MEJORADO"""
        logger.info("🧠 Iniciando entrenamiento TensorFlow MEJORADO")
        logger.info(f"   Arquitectura: {TFConfig.HIDDEN_LAYERS}")
        logger.info(f"   Learning rate: {TFConfig.LEARNING_RATE}")
        logger.info(f"   Batch size: {TFConfig.BATCH_SIZE}")
        logger.info(f"   Épocas: {TFConfig.EPOCHS}")

        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None

        # Crear modelo
        input_dim = X_train_scaled.shape[1]
        self.model = self._create_model(input_dim)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=TFConfig.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Entrenar
        logger.info("   🔄 Entrenando red neuronal...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=TFConfig.EPOCHS,
            batch_size=TFConfig.BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True

        # Evaluar
        y_train_pred = (self.model.predict(X_train_scaled) > 0.5).astype(int).flatten()
        train_acc = accuracy_score(y_train, y_train_pred)
        logger.info(f"   ✅ Accuracy final en train: {train_acc*100:.2f}%")

        if validation_data:
            y_val_pred = (self.model.predict(X_val_scaled) > 0.5).astype(int).flatten()
            val_acc = accuracy_score(y_val, y_val_pred)
            logger.info(f"   ✅ Accuracy final en validation: {val_acc*100:.2f}%")

            # Reporte detallado
            logger.info("\n📊 Classification Report (Validation):")
            print(classification_report(y_val, y_val_pred,
                                      target_names=['No Confirmed', 'Confirmed']))

        logger.info("✅ TensorFlow entrenado exitosamente")

        return self

    def predict(self, X):
        """Predice clases"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llamar a fit() primero.")

        X_scaled = self.scaler.transform(X)
        y_proba = self.model.predict(X_scaled, verbose=0)
        return (y_proba > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Predice probabilidades"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llamar a fit() primero.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def save(self, model_dir):
        """Guarda el modelo"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / TFConfig.MODEL_FILE
        scaler_path = model_dir / TFConfig.SCALER_FILE
        history_path = model_dir / TFConfig.HISTORY_FILE

        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

        # Guardar historia
        if self.history:
            history_dict = {k: [float(v) for v in vals]
                          for k, vals in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)

        logger.info(f"✅ TensorFlow guardado en {model_dir}")

    def load(self, model_dir):
        """Carga el modelo"""
        model_dir = Path(model_dir)

        model_path = model_dir / TFConfig.MODEL_FILE
        scaler_path = model_dir / TFConfig.SCALER_FILE

        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True

        logger.info(f"✅ TensorFlow cargado desde {model_dir}")

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logger.info("="*50)
        logger.info("📊 EVALUACIÓN TENSORFLOW")
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
