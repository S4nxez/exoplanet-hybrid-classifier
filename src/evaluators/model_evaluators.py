#!/usr/bin/env python3
"""
MODEL EVALUATORS
================
Evaluadores específicos para cada tipo de modelo
"""

import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score
from src.evaluators.base_evaluator import BaseEvaluator
from src.models.tensorflow_hybrid import TensorFlowHybridModel


class TotalModelEvaluator(BaseEvaluator):
    """Evaluador para modelo total"""
    
    def load_model(self, model_path='saved_models/total_model.pkl'):
        """Cargar modelo total"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self):
        """Predecir con modelo total"""
        return self.model.predict(self.X_test)


class PartialModelEvaluator(BaseEvaluator):
    """Evaluador para modelo parcial"""
    
    def load_model(self, model_path='saved_models/partial_model.pkl'):
        """Cargar modelo parcial"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self):
        """Predecir con modelo parcial"""
        return self.model.predict(self.X_test)
    
    def print_results(self, accuracy, f1=None):
        """Imprimir resultados con análisis de coverage"""
        super().print_results(accuracy, f1)
        
        # Análisis con confianza si está disponible
        if hasattr(self.model, 'predict_with_confidence'):
            confident_count = 0
            for x in self.X_test:
                x_single = x.reshape(1, -1)
                confident_pred = self.model.predict_with_confidence(x_single)
                if confident_pred[0] != 'uncertain':
                    confident_count += 1
            
            coverage = confident_count / len(self.X_test)
            print(f"   🎯 Coverage: {coverage:.4f} ({coverage*100:.2f}%)")


class HybridModelEvaluator(BaseEvaluator):
    """Evaluador para modelo híbrido"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.partial_model = None
        self.total_model = None
    
    def load_model(self, model_path='saved_models/hybrid_tf_model.keras'):
        """Cargar modelo híbrido completo"""
        # Cargar modelos base
        with open('saved_models/partial_model.pkl', 'rb') as f:
            self.partial_model = pickle.load(f)
        
        with open('saved_models/total_model.pkl', 'rb') as f:
            self.total_model = pickle.load(f)
        
        # Cargar modelo híbrido TensorFlow
        tf_model = tf.keras.models.load_model(model_path)
        
        with open('saved_models/hybrid_tf_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Crear instancia del modelo híbrido
        self.model = TensorFlowHybridModel(
            partial_model=self.partial_model,
            total_model=self.total_model,
            data_processor=self.processor,
            enable_cascade=True
        )
        
        # Asignar componentes cargados
        self.model.tf_model = tf_model
        self.model.scaler = scaler
        self.model.is_trained = True
    
    def predict(self):
        """Predecir con modelo híbrido"""
        predictions, self.cascade_used = self.model.predict(self.X_test)
        return predictions
    
    def print_results(self, accuracy, f1=None):
        """Imprimir resultados con análisis de cascada"""
        super().print_results(accuracy, f1)
        
        # Análisis del sistema de cascada
        cascade_count = np.sum(self.cascade_used)
        stacking_count = len(self.cascade_used) - cascade_count
        
        cascade_accuracy = 0
        stacking_accuracy = 0
        
        if cascade_count > 0:
            cascade_accuracy = accuracy_score(
                self.y_test[self.cascade_used], 
                self.predict()[self.cascade_used] if hasattr(self, '_last_predictions') else self.model.predict(self.X_test)[0][self.cascade_used]
            )
        
        if stacking_count > 0:
            stacking_accuracy = accuracy_score(
                self.y_test[~self.cascade_used], 
                self.predict()[~self.cascade_used] if hasattr(self, '_last_predictions') else self.model.predict(self.X_test)[0][~self.cascade_used]
            )
        
        print(f"\n🎯 Análisis del sistema:")
        print(f"   📊 Uso de cascada: {cascade_count/len(self.y_test)*100:.1f}%")
        print(f"   🎯 Accuracy cascada: {cascade_accuracy:.4f} ({cascade_accuracy*100:.2f}%)")
        print(f"   🤖 Accuracy stacking: {stacking_accuracy:.4f} ({stacking_accuracy*100:.2f}%)")
    
    def predict(self):
        """Predecir y guardar resultados para análisis"""
        predictions, self.cascade_used = self.model.predict(self.X_test)
        self._last_predictions = predictions
        return predictions