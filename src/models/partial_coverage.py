#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO DE COBERTURA PARCIAL
===========================
Modelo de alta precisión para casos extremos
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib

class PartialCoverageModel:
    """Modelo de cobertura parcial con alta precisión y calibración"""

    def __init__(self, confidence_threshold=0.9):
        # Modelo base con calibración
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Calibración con validación cruzada
        self.model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=3
        )
        
        self.scaler = StandardScaler()
        self.confidence_threshold = confidence_threshold
        self.is_trained = False

    def train(self, X_extreme, y_extreme):
        """Entrenar modelo con casos extremos"""
        # División entrenamiento/validación
        X_train, X_test, y_train, y_test = train_test_split(
            X_extreme, y_extreme, test_size=0.3, random_state=42, stratify=y_extreme
        )

        # Escalado
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar modelo calibrado
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True  # Marcar como entrenado ANTES de usar predict_with_confidence

        # Evaluar con métricas extendidas
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Métricas con umbral de confianza
        confident_predictions = self.predict_with_confidence(X_test_scaled)
        confident_mask = confident_predictions != 'uncertain'
        
        coverage = np.mean(confident_mask) if len(confident_mask) > 0 else 0
        confident_accuracy = 0
        if np.any(confident_mask):
            confident_accuracy = accuracy_score(
                y_test[confident_mask], 
                confident_predictions[confident_mask]
            )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': coverage,
            'confident_accuracy': confident_accuracy
        }

    def predict_with_confidence(self, X):
        """Predecir solo cuando la confianza supera el umbral"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        max_probs = np.max(probabilities, axis=1)
        predictions = self.model.predict(X_scaled)
        
        # Solo devolver predicción si supera el umbral de confianza
        confident_predictions = np.where(
            max_probs >= self.confidence_threshold,
            predictions,
            'uncertain'
        )
        
        return confident_predictions
    
    def get_confidence_scores(self, X):
        """Obtener puntuaciones de confianza"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return np.max(probabilities, axis=1)

    def predict(self, X):
        """Predecir casos"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predecir probabilidades"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, model_path, scaler_path):
        """Guardar modelo y escalador"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path, scaler_path):
        """Cargar modelo y escalador"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True