#!/usr/bin/env python3
"""
PARTIAL MODEL TRAINER
======================
Entrenador específico para el modelo de cobertura parcial
"""

from src.trainers.base_trainer import BaseTrainer
from src.models.partial_coverage import PartialCoverageModel
from sklearn.metrics import accuracy_score
import numpy as np


class PartialModelTrainer(BaseTrainer):
    """Entrenador para el modelo de cobertura parcial"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Modelo Parcial"
    
    def create_model(self):
        """Crear modelo de cobertura parcial"""
        print(f"\n🎯 Creando {self.model_name}...")
        self.model = PartialCoverageModel()
    
    def train_model(self):
        """Entrenar modelo de cobertura parcial"""
        print(f"🎯 Entrenando {self.model_name}...")
        self.model.train(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluar modelo con análisis de confianza"""
        print(f"\n📊 Evaluando {self.model_name}...")
        
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Evaluación con confianza si está disponible
        if hasattr(self.model, 'predict_with_confidence'):
            confident_predictions = []
            coverage_count = 0
            
            for i, x in enumerate(self.X_test):
                x_single = x.reshape(1, -1)
                confident_pred = self.model.predict_with_confidence(x_single)
                
                if confident_pred[0] != 'uncertain':
                    confident_predictions.append((i, confident_pred[0]))
                    coverage_count += 1
            
            if confident_predictions:
                confident_indices, confident_preds = zip(*confident_predictions)
                confident_true = self.y_test[list(confident_indices)]
                confident_accuracy = accuracy_score(confident_true, confident_preds)
                coverage = coverage_count / len(self.X_test)
                
                print(f"   🎯 Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
                print(f"   🎯 Confident Accuracy: {confident_accuracy:.4f} ({confident_accuracy*100:.2f}%)")
        
        return accuracy, predictions


def main():
    """Función principal para entrenar modelo parcial"""
    print("🎯 ENTRENADOR DEL MODELO PARCIAL")
    print("="*50)
    
    trainer = PartialModelTrainer()
    model, accuracy = trainer.run_training_pipeline('saved_models/partial_model.pkl')
    
    return model, accuracy


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    model, acc = main()