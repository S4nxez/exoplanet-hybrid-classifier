#!/usr/bin/env python3
"""
BASE TRAINER
============
Clase base para todos los entrenadores de modelos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score
from src.utils.data_processor import DataProcessor
import pickle
import os
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Clase base para entrenadores de modelos"""

    def __init__(self, data_path='data/dataset.csv', test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.processor = DataProcessor()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Cargar y preparar datos"""
        print("📊 Cargando datos...")
        data = self.processor.load_clean_data(self.data_path)

        X = data[self.processor.features].values
        y = data['binary_class'].values

        # Split estratificado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"   📈 Datos de entrenamiento: {len(self.X_train)} muestras")
        print(f"   📈 Datos de prueba: {len(self.X_test)} muestras")
        print(f"   📊 Características: {X.shape[1]}")

    @abstractmethod
    def create_model(self):
        """Crear el modelo específico"""
        pass

    @abstractmethod
    def train_model(self):
        """Entrenar el modelo específico"""
        pass

    def evaluate_model(self):
        """Evaluar el modelo"""
        print("\n📊 Evaluando modelo...")

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)

        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return accuracy, predictions

    def save_model(self, model_path):
        """Guardar el modelo"""
        print(f"\n💾 Guardando modelo...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"   ✅ Modelo guardado en {model_path}")

    def print_detailed_report(self, predictions):
        """Imprimir reporte detallado"""
        print(f"\n📋 REPORTE DETALLADO:")
        print(classification_report(self.y_test, predictions))

    def run_training_pipeline(self, model_path):
        """Ejecutar pipeline completo de entrenamiento"""
        self.load_data()
        self.create_model()
        self.train_model()
        accuracy, predictions = self.evaluate_model()
        self.print_detailed_report(predictions)
        self.save_model(model_path)

        print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
        print(f"   📊 Accuracy final: {accuracy*100:.2f}%")

        return self.model, accuracy