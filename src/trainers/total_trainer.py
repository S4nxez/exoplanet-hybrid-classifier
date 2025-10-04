#!/usr/bin/env python3
"""
TOTAL MODEL TRAINER
===================
Entrenador específico para el modelo de cobertura total
"""

from src.trainers.base_trainer import BaseTrainer
from src.models.total_coverage import TotalCoverageModel
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score


class TotalModelTrainer(BaseTrainer):
    """Entrenador para el modelo de cobertura total"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Modelo Total"

    def create_model(self):
        """Crear modelo de cobertura total"""
        print(f"\n🌐 Creando {self.model_name}...")
        self.model = TotalCoverageModel()

    def train_model(self):
        """Entrenar modelo de cobertura total"""
        print(f"🌐 Entrenando {self.model_name}...")
        self.model.train(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluar modelo con métricas adicionales"""
        print(f"\n📊 Evaluando {self.model_name}...")

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        balanced_acc = balanced_accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, pos_label='Exoplaneta')

        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ✅ Balanced Accuracy: {balanced_acc:.4f}")
        print(f"   ✅ F1-Score: {f1:.4f}")
        print(f"   ✅ Modelo total entrenado correctamente")

        return accuracy, predictions

    def save_model(self, model_path='saved_models/total_model.pkl'):
        """Guardar modelo con feature importance"""
        super().save_model(model_path)

        # Guardar feature importance si está disponible
        if hasattr(self.model, 'get_feature_importance'):
            feature_importance = self.model.get_feature_importance()
            if feature_importance is not None:
                import pickle

                with open('saved_models/total_feature_importance.pkl', 'wb') as f:
                    pickle.dump(feature_importance, f)

                with open('saved_models/total_feature_names.pkl', 'wb') as f:
                    pickle.dump(self.processor.features, f)

                print("   ✅ Feature importance guardada")


def main():
    """Función principal para entrenar modelo total"""
    print("🌐 ENTRENADOR DEL MODELO TOTAL")
    print("="*50)

    trainer = TotalModelTrainer()
    model, accuracy = trainer.run_training_pipeline('saved_models/total_model.pkl')

    return model, accuracy


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    model, acc = main()