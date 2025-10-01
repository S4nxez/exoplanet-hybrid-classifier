#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTOR PRINCIPAL
==================
Script para usar los modelos entrenados y hacer predicciones
"""

import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from utils.data_processor import DataProcessor
from models.partial_coverage import PartialCoverageModel
from models.total_coverage import TotalCoverageModel

class ExoplanetPredictor:
    """Predictor principal del sistema de exoplanetas"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.total_model = TotalCoverageModel()
        self.partial_model = PartialCoverageModel()
        self.is_loaded = False
        self.has_partial = False

    def load_models(self, base_path='saved_models'):
        """Cargar modelos entrenados"""
        try:
            # Cargar modelo total
            self.total_model.load(
                f"{base_path}/total_model.pkl",
                f"{base_path}/total_scaler.pkl",
                f"{base_path}/total_encoder.pkl"
            )
            
            # Cargar modelo parcial si existe
            try:
                self.partial_model.load(
                    f"{base_path}/partial_model.pkl",
                    f"{base_path}/partial_scaler.pkl"
                )
                self.has_partial = True
            except:
                self.has_partial = False
                print("⚠️ Modelo parcial no encontrado, usando solo modelo total")
            
            self.is_loaded = True
            print("✅ Modelos cargados exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            print("💡 Ejecuta primero el entrenamiento: python train.py")

    def predict_single(self, sample_data):
        """Predecir un caso individual"""
        if not self.is_loaded:
            raise ValueError("Los modelos no han sido cargados")

        # Convertir a array si es necesario
        if isinstance(sample_data, dict):
            sample_array = np.array([sample_data[feature] for feature in self.data_processor.features])
        else:
            sample_array = np.array(sample_data)

        # Predecir con modelo total
        total_pred = self.total_model.predict([sample_array])[0]
        
        # Predecir con modelo parcial si está disponible y es caso extremo
        partial_pred = None
        model_used = "Total"
        
        if self.has_partial and self.data_processor.is_extreme_case(sample_array):
            try:
                partial_pred = self.partial_model.predict([sample_array])[0]
                model_used = "Parcial"
            except:
                partial_pred = None
        
        # Determinar qué predicción usar
        if partial_pred is not None:
            final_prediction = partial_pred
            model_used = "Parcial"
        else:
            final_prediction = total_pred
            model_used = "Total"

        return final_prediction, model_used

    def predict_batch(self, data):
        """Predecir múltiples casos"""
        if not self.is_loaded:
            raise ValueError("Los modelos no han sido cargados")

        # Preparar datos
        if isinstance(data, pd.DataFrame):
            X = data[self.data_processor.features].values
        else:
            X = np.array(data)

        # Predecir con modelo total
        total_predictions = self.total_model.predict(X)
        
        results = {
            'predictions': total_predictions,
            'total_cases': len(X),
            'partial_cases': 0,
            'model_used': 'Total'
        }
        
        # Si hay modelo parcial, evaluar casos extremos
        if self.has_partial:
            partial_cases = 0
            for i in range(len(X)):
                if self.data_processor.is_extreme_case(X[i]):
                    try:
                        partial_pred = self.partial_model.predict([X[i]])[0]
                        results['predictions'][i] = partial_pred
                        partial_cases += 1
                    except:
                        pass
            
            results['partial_cases'] = partial_cases
            results['total_cases'] = len(X) - partial_cases
            if partial_cases > 0:
                results['model_used'] = 'Híbrido (Total + Parcial)'

        return results

    def evaluate_on_test_data(self, test_filepath='data/dataset.csv'):
        """Evaluar en datos de test"""
        if not self.is_loaded:
            raise ValueError("Los modelos no han sido cargados")

        from sklearn.metrics import accuracy_score, classification_report

        # Cargar datos de test
        data = self.data_processor.load_clean_data(test_filepath)
        X_train, X_test, y_train, y_test = self.data_processor.prepare_train_test_split(data)

        # Evaluar modelo total
        total_preds = self.total_model.predict(X_test)
        total_accuracy = accuracy_score(y_test, total_preds)
        
        results = {
            'total_accuracy': total_accuracy,
            'total_coverage': 100.0,
            'test_samples': len(y_test)
        }
        
        # Evaluar modelo parcial si existe
        if self.has_partial:
            extreme_indices = [i for i in range(len(X_test)) if self.data_processor.is_extreme_case(X_test[i])]
            if len(extreme_indices) > 0:
                X_extreme = X_test[extreme_indices]
                y_extreme = y_test[extreme_indices]
                partial_preds = self.partial_model.predict(X_extreme)
                partial_accuracy = accuracy_score(y_extreme, partial_preds)
                partial_coverage = len(extreme_indices) / len(y_test) * 100
                
                results.update({
                    'partial_accuracy': partial_accuracy,
                    'partial_coverage': partial_coverage,
                    'extreme_cases': len(extreme_indices)
                })

        return results

    def get_model_info(self):
        """Obtener información de los modelos"""
        if not self.is_loaded:
            return "Modelos no cargados"

        info = {
            'sistema': 'Modelos Individuales de Exoplanetas',
            'modelo_total': 'RandomForestClassifier (todos los casos)',
            'caracteristicas': len(self.data_processor.features),
            'features': self.data_processor.features
        }
        
        if self.has_partial:
            info['modelo_parcial'] = 'GradientBoostingClassifier (casos extremos)'
            info['estrategia'] = 'Total como principal, Parcial como especializado'
        else:
            info['estrategia'] = 'Solo modelo total'

        return info

def main():
    """Función principal para demostrar el uso"""
    print("🔮 PREDICTOR DE EXOPLANETAS")
    print("=" * 40)

    # Crear predictor
    predictor = ExoplanetPredictor()

    # Cargar modelos
    predictor.load_models()

    if predictor.is_loaded:
        # Mostrar información
        info = predictor.get_model_info()
        print(f"📊 Sistema: {info['sistema']}")
        print(f"🎯 Estrategia: {info['estrategia']}")
        print(f"📈 Características: {info['caracteristicas']}")

        # Evaluar en datos de test
        print(f"\n📊 Evaluando en datos de test...")
        results = predictor.evaluate_on_test_data()
        
        print(f"\n🎉 RESULTADOS:")
        print(f"🎯 Modelo Total: {results['total_accuracy']:.4f} ({results['total_accuracy']*100:.2f}%) precisión, {results['total_coverage']:.1f}% cobertura")
        
        if 'partial_accuracy' in results:
            print(f"⚡ Modelo Parcial: {results['partial_accuracy']:.4f} ({results['partial_accuracy']*100:.2f}%) precisión, {results['partial_coverage']:.1f}% cobertura")

        # Ejemplo de predicción
        print(f"\n🔮 Ejemplo de predicción:")
        sample = np.array([10.5, 1500, 3.2, 2.1, 800, 1.2, 0.8, 4500, 0.95])  # Ejemplo
        print(f"📊 Muestra: Período=10.5, Profundidad=1500")
        try:
            prediction, model_used = predictor.predict_single(sample)
            print(f"🎯 Predicción: {prediction}")
            print(f"🤖 Modelo usado: {model_used}")
        except Exception as e:
            print(f"❌ Error en predicción: {e}")

if __name__ == "__main__":
    main()