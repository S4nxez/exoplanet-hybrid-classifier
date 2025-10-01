#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTOR DEL MODELO HÍBRIDO CON TENSORFLOW
===========================================
Script para usar el modelo híbrido entrenado con TensorFlow
"""

import sys
import os

# Agregar la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from utils.data_processor import DataProcessor
from models.partial_coverage import PartialCoverageModel
from models.total_coverage import TotalCoverageModel
from models.tensorflow_hybrid import TensorFlowHybridModel

class HybridTensorFlowPredictor:
    """Predictor que usa el modelo híbrido con TensorFlow"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.partial_model = PartialCoverageModel()
        self.total_model = TotalCoverageModel()
        self.hybrid_model = None
        self.is_loaded = False

    def load_models(self, base_path='saved_models'):
        """Cargar todos los modelos"""
        try:
            # Cargar modelos base
            self.total_model.load(
                f"{base_path}/total_model.pkl",
                f"{base_path}/total_scaler.pkl",
                f"{base_path}/total_encoder.pkl"
            )

            self.partial_model.load(
                f"{base_path}/partial_model.pkl",
                f"{base_path}/partial_scaler.pkl"
            )

            # Cargar modelo híbrido
            self.hybrid_model = TensorFlowHybridModel(
                partial_model=self.partial_model,
                total_model=self.total_model,
                data_processor=self.data_processor
            )

            self.hybrid_model.load(
                f"{base_path}/hybrid_tf_model.keras",
                f"{base_path}/hybrid_tf_scaler.pkl"
            )

            self.is_loaded = True
            print("✅ Todos los modelos cargados exitosamente")

        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            print("💡 Ejecuta primero: python train_hybrid_tensorflow.py")

    def predict_single(self, sample_data):
        """Predecir un caso individual"""
        if not self.is_loaded:
            raise ValueError("Los modelos no han sido cargados")

        # Convertir a array si es necesario
        if isinstance(sample_data, dict):
            sample_array = np.array([sample_data[feature] for feature in self.data_processor.features])
        else:
            sample_array = np.array(sample_data)

        # Predicciones de cada modelo
        hybrid_pred = self.hybrid_model.predict([sample_array])[0]
        total_pred = self.total_model.predict([sample_array])[0]

        # Verificar si es caso extremo para modelo parcial
        is_extreme = self.data_processor.is_extreme_case(sample_array)
        if is_extreme:
            try:
                partial_pred = self.partial_model.predict([sample_array])[0]
            except:
                partial_pred = "N/A"
        else:
            partial_pred = "N/A (caso normal)"

        return {
            'hybrid_prediction': hybrid_pred,
            'total_prediction': total_pred,
            'partial_prediction': partial_pred,
            'is_extreme_case': is_extreme,
            'model_used': 'Híbrido TensorFlow'
        }

    def evaluate_all_models(self, test_filepath='data/dataset.csv'):
        """Evaluar todos los modelos en datos de test"""
        if not self.is_loaded:
            raise ValueError("Los modelos no han sido cargados")

        # Cargar datos de test
        data = self.data_processor.load_clean_data(test_filepath)
        X_train, X_test, y_train, y_test = self.data_processor.prepare_train_test_split(data)

        # Evaluar modelo híbrido
        hybrid_results = self.hybrid_model.evaluate(X_test, y_test)

        return hybrid_results

    def get_model_info(self):
        """Obtener información de todos los modelos"""
        if not self.is_loaded:
            return "Modelos no cargados"

        hybrid_info = self.hybrid_model.get_feature_importance()

        info = {
            'sistema': 'Modelo Híbrido con TensorFlow',
            'modelo_total': 'RandomForestClassifier (base)',
            'modelo_parcial': 'GradientBoostingClassifier (especializado)',
            'modelo_hibrido': 'Red Neuronal TensorFlow (combinador)',
            'caracteristicas': len(self.data_processor.features),
            'features': self.data_processor.features,
            'tf_layers': hybrid_info['layers'],
            'tf_params': hybrid_info['total_params'],
            'tf_trainable_params': hybrid_info['trainable_params']
        }

        return info

def main():
    """Función principal para demostrar el uso"""
    print("🤖 PREDICTOR HÍBRIDO CON TENSORFLOW")
    print("=" * 50)

    # Crear predictor
    predictor = HybridTensorFlowPredictor()

    # Cargar modelos
    predictor.load_models()

    if predictor.is_loaded:
        # Mostrar información
        info = predictor.get_model_info()
        print(f"🔮 Sistema: {info['sistema']}")
        print(f"📊 TensorFlow: {info['tf_layers']} capas, {info['tf_params']:,} parámetros")
        print(f"📈 Características: {info['caracteristicas']}")

        # Evaluar en datos de test
        print(f"\n📊 Evaluando todos los modelos...")
        results = predictor.evaluate_all_models()

        print(f"\n🎉 RESULTADOS COMPARATIVOS:")
        print(f"🤖 Modelo Híbrido (TensorFlow): {results['hybrid_accuracy']:.4f} ({results['hybrid_accuracy']*100:.2f}%)")
        print(f"⚡ Modelo Total (RandomForest): {results['total_accuracy']:.4f} ({results['total_accuracy']*100:.2f}%)")
        if results['partial_accuracy'] > 0:
            print(f"🎯 Modelo Parcial (GradientBoosting): {results['partial_accuracy']:.4f} ({results['partial_accuracy']*100:.2f}%)")

        mejora = results['improvement_over_total'] * 100
        print(f"\n📈 MEJORA DEL HÍBRIDO: {mejora:+.2f} puntos porcentuales")

        # Ejemplo de predicción
        print(f"\n🔮 Ejemplo de predicción:")
        sample = np.array([10.5, 1500, 3.2, 2.1, 800, 1.2, 0.8, 4500, 0.95])
        print(f"📊 Muestra: Período=10.5, Profundidad=1500")

        try:
            prediction_results = predictor.predict_single(sample)
            print(f"🤖 Híbrido: {prediction_results['hybrid_prediction']}")
            print(f"⚡ Total: {prediction_results['total_prediction']}")
            print(f"🎯 Parcial: {prediction_results['partial_prediction']}")
            print(f"🔍 Caso extremo: {'Sí' if prediction_results['is_extreme_case'] else 'No'}")
        except Exception as e:
            print(f"❌ Error en predicción: {e}")

if __name__ == "__main__":
    main()