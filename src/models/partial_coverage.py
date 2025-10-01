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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

class PartialCoverageModel:
    """Modelo de cobertura parcial con alta precisión"""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.scaler = StandardScaler()
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

        # Entrenar
        self.model.fit(X_train_scaled, y_train)

        # Evaluar
        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)

        self.is_trained = True
        return accuracy

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