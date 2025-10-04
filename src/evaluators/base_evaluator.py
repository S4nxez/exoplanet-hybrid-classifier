#!/usr/bin/env python3
"""
BASE EVALUATOR
==============
Clase base para evaluadores de modelos
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.utils.data_processor import DataProcessor
from abc import ABC, abstractmethod
import os


class BaseEvaluator(ABC):
    """Clase base para evaluadores de modelos"""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.processor = DataProcessor()
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_test_data(self, data_path='data/dataset.csv'):
        """Cargar datos de prueba"""
        data = self.processor.load_clean_data(data_path)

        X = data[self.processor.features].values
        y = data['binary_class'].values

        # Split estratificado (mismo usado en entrenamiento)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    @abstractmethod
    def load_model(self, model_path):
        """Cargar modelo específico"""
        pass

    @abstractmethod
    def predict(self):
        """Realizar predicción específica"""
        pass

    def calculate_metrics(self, predictions):
        """Calcular métricas básicas"""
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, pos_label='Exoplaneta')

        return accuracy, f1

    def print_results(self, accuracy, f1=None):
        """Imprimir resultados"""
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if f1 is not None:
            print(f"   📊 F1-Score: {f1:.4f}")
        print(f"   📊 Muestras evaluadas: {len(self.X_test)}")

    def print_detailed_report(self, predictions):
        """Imprimir reporte detallado"""
        print("\n📋 Reporte detallado:")
        print(classification_report(self.y_test, predictions))

    def evaluate(self, model_path):
        """Pipeline completo de evaluación"""
        print(f"🔄 Realizando predicción...")

        self.load_test_data()
        self.load_model(model_path)
        predictions = self.predict()
        accuracy, f1 = self.calculate_metrics(predictions)

        self.print_results(accuracy, f1)
        self.print_detailed_report(predictions)

        print(f"\n🎉 Predicción completada!")
        print(f"   📊 Accuracy final: {accuracy*100:.2f}%")

        return predictions, accuracy