#!/usr/bin/env python3
"""
HYBRID MODEL TRAINER
====================
Entrenador específico para el modelo híbrido TensorFlow
"""

from src.trainers.base_trainer import BaseTrainer
from src.models.partial_coverage import PartialCoverageModel
from src.models.total_coverage import TotalCoverageModel
from src.models.tensorflow_hybrid import TensorFlowHybridModel
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pickle


class HybridModelTrainer(BaseTrainer):
    """Entrenador para el modelo híbrido TensorFlow"""
    
    def __init__(self, epochs=100, batch_size=64, validation_split=0.2, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Modelo Híbrido"
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.partial_model = None
        self.total_model = None
    
    def create_model(self):
        """Crear modelos base y modelo híbrido"""
        print(f"\n🤖 Creando {self.model_name}...")
        
        # Entrenar modelos base
        print("🎯 Entrenando modelo parcial...")
        self.partial_model = PartialCoverageModel()
        self.partial_model.train(self.X_train, self.y_train)
        
        print("🌐 Entrenando modelo total...")
        self.total_model = TotalCoverageModel()
        self.total_model.train(self.X_train, self.y_train)
        
        # Crear modelo híbrido
        self.model = TensorFlowHybridModel(
            partial_model=self.partial_model,
            total_model=self.total_model,
            data_processor=self.processor,
            enable_cascade=True
        )
    
    def train_model(self):
        """Entrenar modelo híbrido TensorFlow"""
        print(f"🤖 Entrenando {self.model_name}...")
        
        train_accuracy, history = self.model.train(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1
        )
        
        print(f"   ✅ Modelo híbrido entrenado")
        print(f"   📊 Accuracy en entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        
        return train_accuracy
    
    def evaluate_model(self):
        """Evaluar modelo híbrido con análisis comparativo"""
        print(f"\n🔬 Evaluando {self.model_name}...")
        
        # Evaluaciones individuales
        total_pred = self.total_model.predict(self.X_test)
        total_accuracy = accuracy_score(self.y_test, total_pred)
        
        partial_pred = self.partial_model.predict(self.X_test)
        partial_accuracy = accuracy_score(self.y_test, partial_pred)
        
        # Evaluación híbrida
        hybrid_pred, cascade_used = self.model.predict(self.X_test)
        hybrid_accuracy = accuracy_score(self.y_test, hybrid_pred)
        hybrid_f1 = f1_score(self.y_test, hybrid_pred, pos_label='Exoplaneta')
        
        print(f"📊 RESULTADOS COMPARATIVOS:")
        print(f"   🌐 Modelo Total:     {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
        print(f"   🎯 Modelo Parcial:   {partial_accuracy:.4f} ({partial_accuracy*100:.2f}%)")
        print(f"   🤖 Modelo Híbrido:   {hybrid_accuracy:.4f} ({hybrid_accuracy*100:.2f}%)")
        print(f"   📊 F1-Score Híbrido: {hybrid_f1:.4f}")
        
        # Análisis de mejora
        mejora_total = hybrid_accuracy - total_accuracy
        mejora_parcial = hybrid_accuracy - partial_accuracy
        
        print(f"\n🚀 MEJORAS CONSEGUIDAS:")
        print(f"   📈 Mejora sobre Total:   {mejora_total:+.4f} ({mejora_total*100:+.2f}%)")
        print(f"   📈 Mejora sobre Parcial: {mejora_parcial:+.4f} ({mejora_parcial*100:+.2f}%)")
        
        # Análisis del sistema de cascada
        self._analyze_cascade_system(cascade_used, hybrid_pred)
        
        return hybrid_accuracy, hybrid_pred
    
    def _analyze_cascade_system(self, cascade_used, hybrid_pred):
        """Analizar el rendimiento del sistema de cascada"""
        cascade_count = np.sum(cascade_used)
        stacking_count = len(cascade_used) - cascade_count
        
        cascade_accuracy = 0
        stacking_accuracy = 0
        
        if cascade_count > 0:
            cascade_accuracy = accuracy_score(
                self.y_test[cascade_used], 
                hybrid_pred[cascade_used]
            )
        
        if stacking_count > 0:
            stacking_accuracy = accuracy_score(
                self.y_test[~cascade_used], 
                hybrid_pred[~cascade_used]
            )
        
        print(f"\n🎯 SISTEMA DE CASCADA:")
        print(f"   📊 Uso de cascada:     {cascade_count/len(self.y_test)*100:.1f}%")
        print(f"   📊 Casos con cascada:  {cascade_count} muestras")
        print(f"   📊 Casos con stacking: {stacking_count} muestras")
        print(f"   🎯 Accuracy cascada:   {cascade_accuracy:.4f} ({cascade_accuracy*100:.2f}%)")
        print(f"   🤖 Accuracy stacking:  {stacking_accuracy:.4f} ({stacking_accuracy*100:.2f}%)")
    
    def save_model(self, model_path='saved_models/hybrid_tf_model.keras'):
        """Guardar todos los modelos del sistema híbrido"""
        print(f"\n💾 Guardando modelos...")
        
        # Guardar modelos base
        with open('saved_models/partial_model.pkl', 'wb') as f:
            pickle.dump(self.partial_model, f)
        
        with open('saved_models/total_model.pkl', 'wb') as f:
            pickle.dump(self.total_model, f)
        
        # Guardar modelo híbrido TensorFlow
        self.model.tf_model.save(model_path)
        
        with open('saved_models/hybrid_tf_scaler.pkl', 'wb') as f:
            pickle.dump(self.model.scaler, f)
        
        print("   ✅ Modelo parcial guardado en saved_models/partial_model.pkl")
        print("   ✅ Modelo total guardado en saved_models/total_model.pkl")
        print(f"   ✅ Modelo híbrido guardado en {model_path}")
        print("   ✅ Scaler híbrido guardado en saved_models/hybrid_tf_scaler.pkl")


def main():
    """Función principal para entrenar modelo híbrido"""
    print("🤖 ENTRENADOR DEL MODELO HÍBRIDO")
    print("="*50)
    
    trainer = HybridModelTrainer()
    model, accuracy = trainer.run_training_pipeline('saved_models/hybrid_tf_model.keras')
    
    # Conclusión específica para híbrido
    total_accuracy = accuracy_score(trainer.y_test, trainer.total_model.predict(trainer.X_test))
    
    if accuracy > total_accuracy:
        print(f"   ✅ ¡ÉXITO! El modelo híbrido supera al baseline total")
        print(f"   📈 Mejora conseguida: {(accuracy - total_accuracy)*100:+.2f}%")
    else:
        print(f"   ⚠️  El modelo híbrido aún no supera al baseline total")
        print(f"   📉 Diferencia: {(accuracy - total_accuracy)*100:.2f}%")
    
    return model, accuracy


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    model, acc = main()