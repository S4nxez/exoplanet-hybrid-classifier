#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO HÍBRIDO CON TENSORFLOW
============================
Modelo híbrido que combina predicciones de modelos de scikit-learn usando TensorFlow
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class TensorFlowHybridModel:
    """Modelo híbrido que usa TensorFlow para combinar predicciones de modelos de scikit-learn"""

    def __init__(self, partial_model=None, total_model=None, data_processor=None):
        self.partial_model = partial_model
        self.total_model = total_model
        self.data_processor = data_processor
        self.tf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Configurar TensorFlow para usar CPU si no hay GPU disponible
        tf.config.experimental.set_memory_growth = True

    def _extract_features(self, X):
        """Extraer características combinadas de ambos modelos"""
        features = []

        # 1. Características originales
        features.append(X)

        # 2. Probabilidades del modelo total
        if self.total_model:
            total_probs = self.total_model.predict_proba(X)
            features.append(total_probs)

        # 3. Probabilidades del modelo parcial para casos extremos
        if self.partial_model and self.data_processor:
            partial_probs = np.zeros((len(X), 2))  # 2 clases
            extreme_mask = np.array([self.data_processor.is_extreme_case(x) for x in X])

            if np.any(extreme_mask):
                X_extreme = X[extreme_mask]
                partial_probs_extreme = self.partial_model.predict_proba(X_extreme)
                partial_probs[extreme_mask] = partial_probs_extreme

            features.append(partial_probs)

        # 4. Características de confianza
        if self.total_model:
            total_conf = np.max(self.total_model.predict_proba(X), axis=1).reshape(-1, 1)
            features.append(total_conf)

        # 5. Indicador de caso extremo
        if self.data_processor:
            extreme_indicator = np.array([self.data_processor.is_extreme_case(x) for x in X]).reshape(-1, 1)
            features.append(extreme_indicator.astype(float))

        # Concatenar todas las características
        combined_features = np.concatenate(features, axis=1)
        return combined_features

    def _build_model(self, input_dim):
        """Construir el modelo de red neuronal con TensorFlow"""
        model = keras.Sequential([
            # Capa de entrada
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            # Capas ocultas
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),

            # Capa de salida
            keras.layers.Dense(2, activation='softmax')  # 2 clases: Exoplaneta, No exoplaneta
        ])

        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """Entrenar el modelo híbrido"""
        print("🚀 Entrenando modelo híbrido con TensorFlow...")

        # Extraer características combinadas
        combined_features = self._extract_features(X)

        # Escalar características
        combined_features_scaled = self.scaler.fit_transform(combined_features)

        # Convertir etiquetas a numéricas
        if isinstance(y[0], str):
            label_map = {'No exoplaneta': 0, 'Exoplaneta': 1}
            y_numeric = np.array([label_map[label] for label in y])
        else:
            y_numeric = y

        # Construir modelo
        self.tf_model = self._build_model(combined_features_scaled.shape[1])

        # Callbacks para mejorar entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]

        # Entrenar modelo
        history = self.tf_model.fit(
            combined_features_scaled, y_numeric,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluar en datos de entrenamiento
        train_preds = self.tf_model.predict(combined_features_scaled)
        train_preds_classes = np.argmax(train_preds, axis=1)
        train_accuracy = accuracy_score(y_numeric, train_preds_classes)

        self.is_trained = True

        print(f"✅ Modelo híbrido entrenado con TensorFlow")
        print(f"📊 Precisión en entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

        return train_accuracy, history

    def predict(self, X):
        """Predecir usando el modelo híbrido"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Extraer características combinadas
        combined_features = self._extract_features(X)
        combined_features_scaled = self.scaler.transform(combined_features)

        # Predecir probabilidades
        probs = self.tf_model.predict(combined_features_scaled, verbose=0)

        # Convertir a clases
        predictions = np.argmax(probs, axis=1)

        # Convertir a etiquetas string
        label_map = {0: 'No exoplaneta', 1: 'Exoplaneta'}
        predictions_labels = [label_map[pred] for pred in predictions]

        return np.array(predictions_labels)

    def predict_proba(self, X):
        """Predecir probabilidades"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        combined_features = self._extract_features(X)
        combined_features_scaled = self.scaler.transform(combined_features)

        return self.tf_model.predict(combined_features_scaled, verbose=0)

    def evaluate(self, X_test, y_test):
        """Evaluar el modelo híbrido"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Predicciones del modelo híbrido
        hybrid_preds = self.predict(X_test)
        hybrid_accuracy = accuracy_score(y_test, hybrid_preds)

        # Comparar con modelos individuales
        total_preds = self.total_model.predict(X_test)
        total_accuracy = accuracy_score(y_test, total_preds)

        # Evaluar modelo parcial en casos extremos
        extreme_indices = [i for i in range(len(X_test)) if self.data_processor.is_extreme_case(X_test[i])]
        if len(extreme_indices) > 0:
            X_extreme = X_test[extreme_indices]
            y_extreme = y_test[extreme_indices]
            partial_preds = self.partial_model.predict(X_extreme)
            partial_accuracy = accuracy_score(y_extreme, partial_preds)
            partial_coverage = len(extreme_indices) / len(y_test) * 100
        else:
            partial_accuracy = 0
            partial_coverage = 0

        results = {
            'hybrid_accuracy': hybrid_accuracy,
            'total_accuracy': total_accuracy,
            'partial_accuracy': partial_accuracy,
            'partial_coverage': partial_coverage,
            'improvement_over_total': hybrid_accuracy - total_accuracy,
            'improvement_over_partial': hybrid_accuracy - partial_accuracy if partial_accuracy > 0 else 0,
            'test_samples': len(y_test),
            'extreme_cases': len(extreme_indices),
            'confusion_matrix': confusion_matrix(y_test, hybrid_preds),
            'classification_report': classification_report(y_test, hybrid_preds)
        }

        return results

    def save(self, model_path='saved_models/hybrid_tf_model.keras', scaler_path='saved_models/hybrid_tf_scaler.pkl'):
        """Guardar el modelo híbrido"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Guardar modelo TensorFlow con extensión .keras
        self.tf_model.save(model_path)

        # Guardar scaler
        joblib.dump(self.scaler, scaler_path)

        print(f"✅ Modelo híbrido guardado en {model_path}")

    def load(self, model_path='saved_models/hybrid_tf_model.keras', scaler_path='saved_models/hybrid_tf_scaler.pkl'):
        """Cargar el modelo híbrido"""
        try:
            # Cargar modelo TensorFlow
            self.tf_model = keras.models.load_model(model_path)

            # Cargar scaler
            self.scaler = joblib.load(scaler_path)

            self.is_trained = True
            print(f"✅ Modelo híbrido cargado desde {model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")

    def get_feature_importance(self):
        """Obtener información sobre las características del modelo"""
        if not self.is_trained:
            return "Modelo no entrenado"

        trainable_params = sum([np.prod(var.shape) for var in self.tf_model.trainable_variables])

        return {
            'layers': len(self.tf_model.layers),
            'total_params': self.tf_model.count_params(),
            'trainable_params': trainable_params
        }