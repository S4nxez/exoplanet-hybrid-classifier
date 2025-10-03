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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.stats import entropy
import joblib
import os

class TensorFlowHybridModel:
    """Modelo híbrido jerárquico + stacking con TensorFlow"""

    def __init__(self, partial_model=None, total_model=None, data_processor=None, 
                 partial_threshold=0.9, enable_cascade=True):
        self.partial_model = partial_model
        self.total_model = total_model
        self.data_processor = data_processor
        self.partial_threshold = partial_threshold
        self.enable_cascade = enable_cascade
        self.tf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Configurar TensorFlow para usar CPU si no hay GPU disponible
        tf.config.experimental.set_memory_growth = True

    def _extract_features(self, X):
        """Extraer características optimizadas y focalizadas"""
        features = []

        # 1. Características originales (las más importantes)
        features.append(X)

        # 2. Probabilidades del modelo total (información valiosa)
        total_probs = None
        if self.total_model:
            total_probs = self.total_model.predict_proba(X)
            features.append(total_probs)

        # 3. Probabilidades del modelo parcial (solo para casos extremos)
        partial_probs = None
        if self.partial_model and self.data_processor:
            partial_probs = np.zeros((len(X), 2))
            extreme_mask = np.array([self.data_processor.is_extreme_case(x) for x in X])

            if np.any(extreme_mask):
                X_extreme = X[extreme_mask]
                partial_probs_extreme = self.partial_model.predict_proba(X_extreme)
                partial_probs[extreme_mask] = partial_probs_extreme

            features.append(partial_probs)

        # 4. Características de confianza (valores clave para decisión)
        if total_probs is not None:
            total_conf = np.max(total_probs, axis=1).reshape(-1, 1)
            features.append(total_conf)

        if partial_probs is not None:
            partial_conf = np.max(partial_probs, axis=1).reshape(-1, 1)
            features.append(partial_conf)

        # 5. Diferencia de confianza (información discriminativa)
        if total_probs is not None and partial_probs is not None:
            total_max_conf = np.max(total_probs, axis=1)
            partial_max_conf = np.max(partial_probs, axis=1)
            conf_diff = np.abs(total_max_conf - partial_max_conf).reshape(-1, 1)
            features.append(conf_diff)

        # 6. Indicador de casos extremos (información estructural importante)
        if self.data_processor:
            extreme_indicators = np.array([
                [1.0 if self.data_processor.is_extreme_case(x) else 0.0] 
                for x in X
            ])
            features.append(extreme_indicators)

        # ELIMINAMOS características que pueden introducir ruido:
        # - Entropía de probabilidades
        # - Feature importance weights (pueden causar mismatch)
        
        # Concatenar todas las características
        return np.concatenate(features, axis=1)

        # 5. Indicador de caso extremo
        if self.data_processor:
            extreme_indicator = np.array([self.data_processor.is_extreme_case(x) for x in X]).reshape(-1, 1)
            features.append(extreme_indicator.astype(float))

        # Concatenar todas las características
        combined_features = np.concatenate(features, axis=1)
        return combined_features

    def _build_model(self, input_dim):
        """Construir modelo de red neuronal optimizado para superar al baseline"""
        model = keras.Sequential([
            # Capa de entrada más robusta
            keras.layers.Dense(512, 
                             activation='relu', 
                             input_shape=(input_dim,),
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Capas profundas para mejor representación
            keras.layers.Dense(256, 
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            
            keras.layers.Dense(128, 
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer='he_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            # Capas de refinamiento
            keras.layers.Dense(64, 
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer='he_normal'),
            keras.layers.Dropout(0.15),
            
            keras.layers.Dense(32,
                             activation='relu',
                             kernel_initializer='he_normal'),
            keras.layers.Dropout(0.1),

            # Capa de salida con softmax calibrado
            keras.layers.Dense(2, activation='softmax')
        ])

        # Compilar con optimizador más sofisticado
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
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

        # Callbacks mejorados para entrenamiento robusto
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # Más paciencia para encontrar mejor solución
                restore_best_weights=True,
                min_delta=0.001  # Mínima mejora requerida
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Reducción más gradual
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            # Callback personalizado para monitorear progreso
            keras.callbacks.ModelCheckpoint(
                'temp_best_hybrid_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
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
        """Predicción híbrida optimizada: Cascada + Total Model como backbone fuerte"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        predictions = []
        cascade_used = []
        
        for i, x in enumerate(X):
            x_single = x.reshape(1, -1)
            
            # 🔹 NIVEL 1: CASCADA - Modelo parcial para casos extremos de alta confianza
            if (self.enable_cascade and 
                self.partial_model and 
                hasattr(self.partial_model, 'predict_with_confidence') and
                self.data_processor and 
                self.data_processor.is_extreme_case(x)):
                
                confident_pred = self.partial_model.predict_with_confidence(x_single)
                
                if confident_pred[0] != 'uncertain':
                    # Cascada: usar modelo parcial de alta confianza
                    predictions.append(confident_pred[0])
                    cascade_used.append(True)
                    continue
            
            # 🔹 NIVEL 2: BACKBONE INTELIGENTE
            # Para todos los demás casos, usar el modelo total como base sólida
            # Solo aplicar híbrido TensorFlow cuando pueda aportar valor real
            
            total_probs = self.total_model.predict_proba(x_single)
            total_pred = self.total_model.predict(x_single)[0]
            total_confidence = np.max(total_probs)
            
            # Si el modelo total está muy seguro (>85%), confiar en él directamente
            if total_confidence > 0.85:
                predictions.append(total_pred)
                cascade_used.append(False)
                continue
            
            # Solo para casos de baja confianza del total, consultar híbrido
            combined_features = self._extract_features(x_single)
            combined_features_scaled = self.scaler.transform(combined_features)
            
            hybrid_probs = self.tf_model.predict(combined_features_scaled, verbose=0)
            hybrid_confidence = np.max(hybrid_probs)
            
            # Solo usar híbrido si está significativamente más seguro
            if hybrid_confidence > total_confidence + 0.15:  # Margen alto
                hybrid_pred_idx = np.argmax(hybrid_probs, axis=1)[0]
                label_map = {0: 'No exoplaneta', 1: 'Exoplaneta'}
                predictions.append(label_map[hybrid_pred_idx])
            else:
                # Default: usar modelo total (más confiable)
                predictions.append(total_pred)
            
            cascade_used.append(False)

        return np.array(predictions), np.array(cascade_used)
    
    def predict_simple(self, X):
        """Predecir sin información de cascada (para compatibilidad)"""
        predictions, _ = self.predict(X)
        return predictions

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

    def evaluate(self, X_test, y_test):
        """Evaluar el modelo híbrido jerárquico"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Predicciones del modelo híbrido con información de cascada
        hybrid_preds, cascade_used = self.predict(X_test)
        hybrid_accuracy = accuracy_score(y_test, hybrid_preds)
        hybrid_f1 = f1_score(y_test, hybrid_preds, average='weighted', zero_division=0)

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

        # Métricas de cascada
        cascade_usage = np.mean(cascade_used) * 100  # Porcentaje de casos que usaron cascada
        cascade_indices = np.where(cascade_used)[0]
        stacking_indices = np.where(~cascade_used)[0]
        
        cascade_accuracy = 0
        stacking_accuracy = 0
        
        if len(cascade_indices) > 0:
            cascade_accuracy = accuracy_score(y_test[cascade_indices], hybrid_preds[cascade_indices])
        
        if len(stacking_indices) > 0:
            stacking_accuracy = accuracy_score(y_test[stacking_indices], hybrid_preds[stacking_indices])

        results = {
            'hybrid_accuracy': hybrid_accuracy,
            'hybrid_f1_score': hybrid_f1,
            'total_accuracy': total_accuracy,
            'partial_accuracy': partial_accuracy,
            'partial_coverage': partial_coverage,
            'cascade_usage': cascade_usage,
            'cascade_accuracy': cascade_accuracy,
            'stacking_accuracy': stacking_accuracy,
            'improvement_over_total': hybrid_accuracy - total_accuracy,
            'improvement_over_partial': hybrid_accuracy - partial_accuracy if partial_accuracy > 0 else 0,
            'test_samples': len(y_test),
            'extreme_cases': len(extreme_indices),
            'cascade_cases': len(cascade_indices),
            'stacking_cases': len(stacking_indices),
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
    
    def explain_prediction(self, X_sample, method='feature_importance'):
        """Explicar predicción individual"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        if method == 'feature_importance':
            # Usar gradientes para explicar importancia de características
            X_sample_reshaped = X_sample.reshape(1, -1)
            combined_features = self._extract_features(X_sample_reshaped)
            combined_features_scaled = self.scaler.transform(combined_features)
            
            # Convertir a tensor
            x_tensor = tf.Variable(combined_features_scaled, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                predictions = self.tf_model(x_tensor)
                predicted_class = tf.argmax(predictions, axis=1)
                prediction_score = tf.reduce_max(predictions)
            
            # Calcular gradientes
            gradients = tape.gradient(prediction_score, x_tensor)
            feature_importance = tf.abs(gradients).numpy().flatten()
            
            return {
                'prediction': predicted_class.numpy()[0],
                'confidence': prediction_score.numpy(),
                'feature_importance': feature_importance,
                'features_used': combined_features.flatten()
            }
        
        else:
            return {'error': f'Método {method} no implementado'}
    
    def get_cascade_statistics(self, X_test, y_test):
        """Obtener estadísticas detalladas del sistema de cascada"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        predictions, cascade_used = self.predict(X_test)
        
        # Casos que usaron cascada (modelo parcial)
        cascade_indices = np.where(cascade_used)[0]
        stacking_indices = np.where(~cascade_used)[0]
        
        stats = {
            'total_samples': len(X_test),
            'cascade_count': len(cascade_indices),
            'stacking_count': len(stacking_indices),
            'cascade_percentage': len(cascade_indices) / len(X_test) * 100,
            'stacking_percentage': len(stacking_indices) / len(X_test) * 100
        }
        
        if len(cascade_indices) > 0:
            cascade_accuracy = accuracy_score(y_test[cascade_indices], predictions[cascade_indices])
            stats['cascade_accuracy'] = cascade_accuracy
        
        if len(stacking_indices) > 0:
            stacking_accuracy = accuracy_score(y_test[stacking_indices], predictions[stacking_indices])
            stats['stacking_accuracy'] = stacking_accuracy
        
        return stats

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