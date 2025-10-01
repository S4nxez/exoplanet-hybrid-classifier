#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO DE COBERTURA TOTAL
=========================
Modelo que clasifica todos los casos
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

class TotalCoverageModel:
    """Modelo de cobertura total"""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, X, y):
        """Entrenar modelo con todos los datos"""
        # División entrenamiento/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalado
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Codificar etiquetas
        self.label_encoder.fit(y_train)

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

    def save(self, model_path, scaler_path, encoder_path):
        """Guardar modelo, escalador y codificador"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)

    def load(self, model_path, scaler_path, encoder_path):
        """Cargar modelo, escalador y codificador"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_trained = True