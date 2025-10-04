#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORKHESTRA - SISTEMA HÍBRIDO INTELIGENTE
=======================================
Sistema avanzado que orquesta modelos de scikit-learn y TensorFlow
para clasificación de exoplanetas con fusión inteligente basada en confianza.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from scipy.stats import entropy
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class OrkhestraftHybridModel:
    """
    Sistema Orkhestra: Fusión inteligente de modelos especializados

    Combina:
    - Modelo parcial (RandomForest ultra-preciso) para casos seguros
    - Modelo total (TensorFlow completo) para cobertura universal

    Con lógica de fusión basada en confianza configurable.
    """

    def __init__(self, partial_model=None, total_model=None, data_processor=None,
                 confidence_threshold=0.85, enable_fusion=True, auto_optimize=True):
        """
        Inicializar sistema Orkhestra

        Args:
            partial_model: Modelo RandomForest ultra-preciso
            total_model: Modelo TensorFlow de cobertura completa
            data_processor: Procesador de datos
            confidence_threshold: Umbral de confianza para fusión (0.1-0.99)
            enable_fusion: Habilitar lógica de fusión inteligente
            auto_optimize: Auto-optimización de umbral
        """
        self.partial_model = partial_model
        self.total_model = total_model
        self.data_processor = data_processor
        self.confidence_threshold = confidence_threshold
        self.enable_fusion = enable_fusion
        self.auto_optimize = auto_optimize

        # Modelo TensorFlow para fusión avanzada
        self.tf_fusion_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Métricas de rendimiento
        self.fusion_metrics = {
            'partial_usage': 0,
            'total_usage': 0,
            'fusion_ratio': 0,
            'confidence_scores': [],
            'accuracy_by_method': {}
        }

        # Configurar TensorFlow
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """Configurar TensorFlow para rendimiento óptimo"""
        try:
            # Configurar memoria GPU si está disponible
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    def set_confidence_threshold(self, threshold):
        """
        Ajustar umbral de confianza para fusión

        Args:
            threshold (float): Nuevo umbral (0.1-0.99)
        """
        if not 0.1 <= threshold <= 0.99:
            raise ValueError("Umbral debe estar entre 0.1 y 0.99")

        self.confidence_threshold = threshold
        print(f"🎯 Umbral de confianza actualizado a: {threshold:.3f}")

    def predict_with_confidence(self, X):
        """
        Predicción con fusión inteligente Orkhestra SIMPLIFICADA Y FUNCIONAL
        """
        if not self.is_trained:
            raise ValueError("El sistema Orkhestra debe ser entrenado primero")

        X = np.array(X)
        n_samples = len(X)

        # 1. Obtener predicciones del modelo total (base confiable)
        total_predictions = self.total_model.predict(X)
        if hasattr(self.total_model, 'predict_proba'):
            total_probs = self.total_model.predict_proba(X)
            total_confidences = np.max(total_probs, axis=1)
        else:
            total_confidences = np.ones(n_samples) * 0.8

        # 2. Inicializar con predicciones del modelo total
        final_predictions = total_predictions.copy()
        final_confidences = total_confidences.copy()

        # Información de fusión
        fusion_info = {
            'used_partial': np.zeros(n_samples, dtype=bool),
            'used_total': np.ones(n_samples, dtype=bool),
            'confidence_scores': total_confidences.copy()
        }

        # 3. Usar modelo parcial donde sea ultra-confiado
        if self.partial_model is not None and self.enable_fusion:
            partial_confident = self.partial_model.predict_with_confidence(X)
            partial_mask = partial_confident != 'Unknown'

            if np.any(partial_mask):
                # Reemplazar con predicciones del modelo parcial
                final_predictions[partial_mask] = partial_confident[partial_mask]

                # Obtener confianzas del modelo parcial
                if hasattr(self.partial_model, 'get_confidence_scores'):
                    partial_confidences = self.partial_model.get_confidence_scores(X)
                    final_confidences[partial_mask] = partial_confidences[partial_mask]
                else:
                    final_confidences[partial_mask] = 0.98  # Ultra-alta confianza

                # Actualizar información de fusión
                fusion_info['used_partial'][partial_mask] = True
                fusion_info['used_total'][partial_mask] = False

        # 4. Actualizar métricas de fusión
        self._update_fusion_metrics(fusion_info)

        return final_predictions, final_confidences, fusion_info

    def _get_total_predictions(self, X):
        """Obtener predicciones del modelo total con confianza"""
        if hasattr(self.total_model, 'predict_proba'):
            probs = self.total_model.predict_proba(X)
            predictions = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
        else:
            predictions = self.total_model.predict(X)
            confidences = np.ones(len(predictions)) * 0.5  # Confianza por defecto

        return predictions, confidences

    def _get_partial_predictions(self, X):
        """Obtener predicciones del modelo parcial SOLO para casos ultra-confiados"""
        if len(X) == 0:
            return np.array([]), np.array([])

        # Usar el método especializado que filtra por confianza
        confident_predictions = self.partial_model.predict_with_confidence(X)
        confident_mask = confident_predictions != 'Unknown'

        if not np.any(confident_mask):
            # No hay predicciones confiadas
            return np.array([]), np.array([])

        # Solo devolver las predicciones ultra-confiadas
        X_confident = X[confident_mask]
        if len(X_confident) == 0:
            return np.array([]), np.array([])

        # Obtener predicciones y confianzas solo para casos ultra-seguros
        if hasattr(self.partial_model, 'predict_proba'):
            probs = self.partial_model.predict_proba(X_confident)
            predictions = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
        else:
            predictions = self.partial_model.predict(X_confident)
            confidences = np.ones(len(predictions)) * 0.98  # Ultra-alta confianza

        return predictions, confidences

    def predict(self, X):
        """Predicción simple (compatibilidad)"""
        predictions, _, _ = self.predict_with_confidence(X)
        # Convertir de binario a etiquetas de texto
        return self._convert_to_labels(predictions)

    def predict_proba(self, X):
        """Predicción con probabilidades (compatibilidad)"""
        predictions, confidences, _ = self.predict_with_confidence(X)

        # Convertir a probabilidades binarias
        probs = np.zeros((len(predictions), 2))
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if pred == 1:  # Exoplaneta
                probs[i] = [1-conf, conf]
            else:  # No exoplaneta
                probs[i] = [conf, 1-conf]

        return probs

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrenar el sistema Orkhestra completo

        Args:
            X_train: Datos de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Labels de validación (opcional)
        """
        print("🎼 Entrenando sistema Orkhestra...")

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 1. Entrenar modelos base si no están entrenados
        if self.partial_model is not None and not hasattr(self.partial_model, 'classes_'):
            print("   🎯 Entrenando modelo parcial...")
            self.partial_model.train(X_train, y_train)

        if self.total_model is not None and not hasattr(self.total_model, 'predict'):
            print("   🌐 Entrenando modelo total...")
            self.total_model.train(X_train, y_train)

        # 2. Crear características para modelo de fusión TensorFlow
        print("   🤖 Entrenando modelo de fusión TensorFlow...")
        self._train_fusion_model(X_train, y_train)

        # Marcar como entrenado antes de optimización
        self.is_trained = True

        # 3. Optimizar umbral si está habilitado
        if self.auto_optimize and X_val is not None and y_val is not None:
            print("   🔧 Optimizando umbral de confianza...")
            self.optimize_threshold(X_val, y_val)

        print("✅ Sistema Orkhestra entrenado correctamente")

    def _train_fusion_model(self, X_train, y_train):
        """Entrenar modelo de fusión TensorFlow avanzado"""
        # Extraer características de fusión
        fusion_features = self._extract_fusion_features(X_train)

        # Normalizar características
        fusion_features = self.scaler.fit_transform(fusion_features)

        # Crear modelo TensorFlow para fusión
        input_dim = fusion_features.shape[1]

        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Entrenar con early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        )

        # Convertir labels string a binario si es necesario
        y_binary = self._convert_to_binary(y_train)

        model.fit(
            fusion_features, y_binary,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        self.tf_fusion_model = model

    def _extract_fusion_features(self, X):
        """Extraer características optimizadas para fusión"""
        features = []

        # 1. Características originales
        features.append(X)

        # 2. Predicciones de modelos base
        if self.total_model:
            total_probs = self.total_model.predict_proba(X) if hasattr(self.total_model, 'predict_proba') else None
            if total_probs is not None:
                features.append(total_probs)
                # Confianza del modelo total
                features.append(np.max(total_probs, axis=1).reshape(-1, 1))

        if self.partial_model:
            try:
                partial_probs = self.partial_model.predict_proba(X) if hasattr(self.partial_model, 'predict_proba') else None
                if partial_probs is not None:
                    features.append(partial_probs)
                    # Confianza del modelo parcial
                    features.append(np.max(partial_probs, axis=1).reshape(-1, 1))
            except:
                # Si el modelo parcial no puede predecir todos los casos
                dummy_probs = np.ones((len(X), 2)) * 0.5
                features.append(dummy_probs)
                features.append(np.ones((len(X), 1)) * 0.5)

        return np.concatenate(features, axis=1)

    def optimize_threshold(self, X_val, y_val, threshold_range=(0.8, 0.95), n_steps=20):
        """
        Optimizar umbral de confianza usando datos de validación

        Args:
            X_val: Datos de validación
            y_val: Labels de validación
            threshold_range: Rango de umbrales a probar (ajustado para mejor rendimiento)
            n_steps: Número de pasos en el rango
        """
        print(f"🔍 Optimizando umbral en rango {threshold_range}...")

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        best_threshold = self.confidence_threshold
        best_accuracy = 0

        results = []

        for threshold in thresholds:
            # Probar umbral temporal
            old_threshold = self.confidence_threshold
            self.confidence_threshold = threshold

            # Evaluar con este umbral
            predictions, _, fusion_info = self.predict_with_confidence(X_val)
            y_val_binary = self._convert_to_binary(y_val)
            predictions_binary = self._convert_to_binary(predictions)

            accuracy = accuracy_score(y_val_binary, predictions_binary)

            # Penalizar umbrales que den mal rendimiento
            coverage = np.mean(fusion_info['used_partial'])
            if coverage > 0.5:  # Si usa demasiado el modelo parcial
                accuracy *= 0.9  # Penalizar

            results.append((threshold, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

            # Restaurar umbral
            self.confidence_threshold = old_threshold

        # Establecer mejor umbral
        self.confidence_threshold = best_threshold

        print(f"✅ Umbral óptimo encontrado: {best_threshold:.3f} (accuracy: {best_accuracy:.4f})")

        return best_threshold, results

    def get_comparative_metrics(self, X_test, y_test):
        """
        Obtener métricas comparativas detalladas del sistema

        Args:
            X_test: Datos de prueba
            y_test: Labels de prueba

        Returns:
            dict: Métricas detalladas por componente
        """
        y_test_binary = self._convert_to_binary(y_test)

        # Predicciones del sistema Orkhestra
        orkhestra_pred, orkhestra_conf, fusion_info = self.predict_with_confidence(X_test)
        orkhestra_pred_binary = self._convert_to_binary(orkhestra_pred)

        # Predicciones de modelos individuales
        total_pred = self.total_model.predict(X_test) if self.total_model else None
        partial_pred = None
        if self.partial_model:
            try:
                partial_pred = self.partial_model.predict(X_test)
            except:
                partial_pred = None

        metrics = {
            'orkhestra': self._calculate_metrics(y_test_binary, orkhestra_pred_binary),
            'fusion_stats': {
                'partial_usage_ratio': np.mean(fusion_info['used_partial']),
                'total_usage_ratio': np.mean(fusion_info['used_total']),
                'average_confidence': np.mean(orkhestra_conf),
                'high_confidence_ratio': np.mean(orkhestra_conf >= self.confidence_threshold)
            }
        }

        if total_pred is not None:
            total_pred_binary = self._convert_to_binary(total_pred)
            metrics['total_model'] = self._calculate_metrics(y_test_binary, total_pred_binary)

        if partial_pred is not None:
            partial_pred_binary = self._convert_to_binary(partial_pred)
            metrics['partial_model'] = self._calculate_metrics(y_test_binary, partial_pred_binary)

        return metrics

    def _calculate_metrics(self, y_true, y_pred):
        """Calcular métricas estándar"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _convert_to_binary(self, labels):
        """Convertir labels a formato binario (0/1)"""
        if len(labels) == 0:
            return np.array([])

        # Filtrar casos 'Unknown' antes de conversión
        valid_labels = labels[labels != 'Unknown']
        if len(valid_labels) == 0:
            return np.array([])

        if isinstance(valid_labels[0], str):
            return np.array([1 if label == 'Exoplaneta' else 0 for label in valid_labels])
        return np.array(valid_labels)

    def _convert_to_labels(self, binary_predictions):
        """Convertir predicciones binarias de vuelta a labels de texto"""
        return np.array(['Exoplaneta' if pred == 1 else 'No exoplaneta' for pred in binary_predictions])

    def _update_fusion_metrics(self, fusion_info):
        """Actualizar métricas internas de fusión"""
        self.fusion_metrics['partial_usage'] = np.sum(fusion_info['used_partial'])
        self.fusion_metrics['total_usage'] = np.sum(fusion_info['used_total'])

        total_predictions = len(fusion_info['used_partial'])
        if total_predictions > 0:
            self.fusion_metrics['fusion_ratio'] = self.fusion_metrics['partial_usage'] / total_predictions

    def save_model(self, base_path='saved_models/orkhestra'):
        """Guardar el sistema Orkhestra completo"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # Guardar modelo de fusión TensorFlow
        if self.tf_fusion_model:
            self.tf_fusion_model.save(f"{base_path}_fusion_model.keras")

        # Guardar scaler y configuración
        joblib.dump(self.scaler, f"{base_path}_scaler.pkl")

        # Guardar configuración
        config = {
            'confidence_threshold': self.confidence_threshold,
            'enable_fusion': self.enable_fusion,
            'auto_optimize': self.auto_optimize,
            'fusion_metrics': self.fusion_metrics,
            'is_trained': self.is_trained
        }
        joblib.dump(config, f"{base_path}_config.pkl")

        print(f"💾 Sistema Orkhestra guardado en {base_path}_*")

    def load_model(self, base_path='saved_models/orkhestra'):
        """Cargar el sistema Orkhestra completo"""
        # Cargar modelo de fusión TensorFlow
        if os.path.exists(f"{base_path}_fusion_model.keras"):
            self.tf_fusion_model = keras.models.load_model(f"{base_path}_fusion_model.keras")

        # Cargar scaler
        if os.path.exists(f"{base_path}_scaler.pkl"):
            self.scaler = joblib.load(f"{base_path}_scaler.pkl")

        # Cargar configuración
        if os.path.exists(f"{base_path}_config.pkl"):
            config = joblib.load(f"{base_path}_config.pkl")
            self.confidence_threshold = config.get('confidence_threshold', 0.9)
            self.enable_fusion = config.get('enable_fusion', True)
            self.auto_optimize = config.get('auto_optimize', True)
            self.fusion_metrics = config.get('fusion_metrics', {})
            self.is_trained = config.get('is_trained', False)

        print(f"📥 Sistema Orkhestra cargado desde {base_path}_*")

    def get_fusion_analysis(self, X_test, y_test):
        """
        Análisis detallado del comportamiento de fusión

        Returns:
            dict: Análisis completo de fusión
        """
        predictions, confidences, fusion_info = self.predict_with_confidence(X_test)

        analysis = {
            'confidence_distribution': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'usage_statistics': {
                'partial_model_usage': np.sum(fusion_info['used_partial']),
                'total_model_usage': np.sum(fusion_info['used_total']),
                'partial_percentage': np.mean(fusion_info['used_partial']) * 100,
                'total_percentage': np.mean(fusion_info['used_total']) * 100
            },
            'performance_by_method': {},
            'threshold_analysis': {
                'current_threshold': self.confidence_threshold,
                'samples_above_threshold': np.sum(confidences >= self.confidence_threshold),
                'percentage_above_threshold': np.mean(confidences >= self.confidence_threshold) * 100
            }
        }

        # Análisis de rendimiento por método
        y_test_binary = self._convert_to_binary(y_test)
        predictions_binary = self._convert_to_binary(predictions)

        # Casos donde se usó modelo parcial
        partial_mask = fusion_info['used_partial']
        if np.any(partial_mask):
            partial_acc = accuracy_score(y_test_binary[partial_mask], predictions_binary[partial_mask])
            analysis['performance_by_method']['partial'] = {
                'accuracy': partial_acc,
                'count': np.sum(partial_mask)
            }

        # Casos donde se usó modelo total
        total_mask = fusion_info['used_total']
        if np.any(total_mask):
            total_acc = accuracy_score(y_test_binary[total_mask], predictions_binary[total_mask])
            analysis['performance_by_method']['total'] = {
                'accuracy': total_acc,
                'count': np.sum(total_mask)
            }

        return analysis


# Alias para compatibilidad con código existente
TensorFlowHybridModel = OrkhestraftHybridModel