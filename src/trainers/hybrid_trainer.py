#!/usr/bin/env python3
"""
ORKHESTRA HYBRID TRAINER
========================
Entrenador específico para el sistema híbrido Orkhestra
"""

from src.trainers.base_trainer import BaseTrainer
from src.models.partial_coverage import PartialCoverageModel
from src.models.total_coverage import TotalCoverageModel
from src.models.tensorflow_hybrid import OrkhestraftHybridModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


class HybridModelTrainer(BaseTrainer):
    """Entrenador para el sistema híbrido Orkhestra"""

    def __init__(self, confidence_threshold=0.85, auto_optimize=True,
                 enable_fusion=True, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Sistema Orkhestra"
        self.confidence_threshold = confidence_threshold
        self.auto_optimize = auto_optimize
        self.enable_fusion = enable_fusion
        self.partial_model = None
        self.total_model = None

    def create_model(self):
        """Crear modelos base y sistema Orkhestra"""
        print(f"\n🎼 Creando {self.model_name}...")

        # Entrenar modelos base
        print("🎯 Entrenando modelo parcial (RandomForest ultra-preciso)...")
        self.partial_model = PartialCoverageModel()
        self.partial_model.train(self.X_train, self.y_train)

        print("🌐 Entrenando modelo total (TensorFlow completo)...")
        self.total_model = TotalCoverageModel()
        self.total_model.train(self.X_train, self.y_train)

        # Crear sistema Orkhestra
        print("🎼 Inicializando sistema de fusión Orkhestra...")
        self.model = OrkhestraftHybridModel(
            partial_model=self.partial_model,
            total_model=self.total_model,
            data_processor=self.processor,
            confidence_threshold=self.confidence_threshold,
            enable_fusion=self.enable_fusion,
            auto_optimize=self.auto_optimize
        )

    def train_model(self):
        """Entrenar sistema Orkhestra con optimización automática"""
        print(f"🎼 Entrenando {self.model_name}...")

        # Crear datos de validación para optimización de umbral
        if self.auto_optimize:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
        else:
            X_train_split, X_val, y_train_split, y_val = self.X_train, None, self.y_train, None

        # Entrenar sistema completo
        train_accuracy = self.model.train(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val
        )

        print(f"   ✅ Sistema Orkhestra entrenado correctamente")
        print(f"   📊 Umbral de confianza optimizado: {self.model.confidence_threshold:.3f}")

        return train_accuracy

    def evaluate_model(self):
        """Evaluar sistema Orkhestra con análisis comparativo avanzado"""
        print(f"\n🔬 Evaluando {self.model_name}...")

        # Evaluaciones individuales
        total_pred = self.total_model.predict(self.X_test)
        total_accuracy = accuracy_score(self.y_test, total_pred)

        partial_pred = self.partial_model.predict(self.X_test)
        partial_accuracy = accuracy_score(self.y_test, partial_pred)

        # Evaluación Orkhestra con información detallada
        orkhestra_pred, orkhestra_conf, fusion_info = self.model.predict_with_confidence(self.X_test)
        orkhestra_accuracy = accuracy_score(self.y_test, orkhestra_pred)
        orkhestra_f1 = f1_score(self.y_test, orkhestra_pred, pos_label='Exoplaneta')

        print(f"📊 RESULTADOS COMPARATIVOS:")
        print(f"   🌐 Modelo Total:     {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
        print(f"   🎯 Modelo Parcial:   {partial_accuracy:.4f} ({partial_accuracy*100:.2f}%)")
        print(f"   🎼 Sistema Orkhestra: {orkhestra_accuracy:.4f} ({orkhestra_accuracy*100:.2f}%)")
        print(f"   📊 F1-Score Orkhestra: {orkhestra_f1:.4f}")

        # Análisis de mejora
        mejora_total = orkhestra_accuracy - total_accuracy
        mejora_parcial = orkhestra_accuracy - partial_accuracy

        print(f"\n🚀 MEJORAS CONSEGUIDAS:")
        print(f"   📈 Mejora sobre Total:   {mejora_total:+.4f} ({mejora_total*100:+.2f}%)")
        print(f"   📈 Mejora sobre Parcial: {mejora_parcial:+.4f} ({mejora_parcial*100:+.2f}%)")

        # Estadísticas de fusión
        partial_usage = np.mean(fusion_info['used_partial']) * 100
        total_usage = np.mean(fusion_info['used_total']) * 100
        avg_confidence = np.mean(orkhestra_conf)

        print(f"\n🎯 ANÁLISIS DE FUSIÓN ORKHESTRA:")
        print(f"   📊 Uso de modelo parcial: {partial_usage:.1f}%")
        print(f"   📊 Uso de modelo total:   {total_usage:.1f}%")
        print(f"   📊 Confianza promedio:    {avg_confidence:.3f}")
        print(f"   🎯 Umbral de confianza:   {self.model.confidence_threshold:.3f}")

        # Análisis de casos por método
        if np.any(fusion_info['used_partial']):
            partial_mask = fusion_info['used_partial']
            partial_cases_acc = accuracy_score(
                np.array(self.y_test)[partial_mask],
                orkhestra_pred[partial_mask]
            )
            print(f"   🎯 Accuracy casos parciales: {partial_cases_acc:.3f} ({partial_cases_acc*100:.1f}%)")

        if np.any(fusion_info['used_total']):
            total_mask = fusion_info['used_total']
            total_cases_acc = accuracy_score(
                np.array(self.y_test)[total_mask],
                orkhestra_pred[total_mask]
            )
            print(f"   🌐 Accuracy casos totales:   {total_cases_acc:.3f} ({total_cases_acc*100:.1f}%)")

        return orkhestra_accuracy, orkhestra_pred

    def save_model(self, model_path='saved_models/orkhestra'):
        """Guardar sistema Orkhestra completo"""
        print(f"\n💾 Guardando sistema Orkhestra...")

        # Guardar modelos base
        with open('saved_models/partial_model.pkl', 'wb') as f:
            pickle.dump(self.partial_model, f)

        with open('saved_models/total_model.pkl', 'wb') as f:
            pickle.dump(self.total_model, f)

        # Guardar sistema Orkhestra completo
        self.model.save_model(model_path)

        print("   ✅ Modelo parcial guardado en saved_models/partial_model.pkl")
        print("   ✅ Modelo total guardado en saved_models/total_model.pkl")
        print(f"   ✅ Sistema Orkhestra guardado en {model_path}_*")


def main():
    """Función principal para entrenar sistema Orkhestra"""
    print("🎼 ENTRENADOR DEL SISTEMA ORKHESTRA")
    print("="*50)

    trainer = HybridModelTrainer(
        confidence_threshold=0.85,
        auto_optimize=True,
        enable_fusion=True
    )

    model, accuracy = trainer.run_training_pipeline('saved_models/orkhestra')

    # Análisis comparativo final
    total_accuracy = accuracy_score(trainer.y_test, trainer.total_model.predict(trainer.X_test))
    partial_accuracy = accuracy_score(trainer.y_test, trainer.partial_model.predict(trainer.X_test))

    print(f"\n🏆 RESULTADO FINAL DEL SISTEMA ORKHESTRA:")

    if accuracy > max(total_accuracy, partial_accuracy):
        print(f"   ✅ ¡ÉXITO! Orkhestra supera a ambos modelos individuales")
        print(f"   📈 Mejora sobre modelo total: {(accuracy - total_accuracy)*100:+.2f}%")
        print(f"   📈 Mejora sobre modelo parcial: {(accuracy - partial_accuracy)*100:+.2f}%")
    else:
        print(f"   ⚠️  Orkhestra necesita más optimización")
        print(f"   📊 Comparado con total: {(accuracy - total_accuracy)*100:+.2f}%")
        print(f"   � Comparado con parcial: {(accuracy - partial_accuracy)*100:+.2f}%")

    # Mostrar configuración final
    print(f"\n🎯 CONFIGURACIÓN FINAL:")
    print(f"   📊 Umbral de confianza: {trainer.model.confidence_threshold:.3f}")
    print(f"   🔧 Fusión habilitada: {trainer.model.enable_fusion}")
    print(f"   🤖 Auto-optimización: {trainer.model.auto_optimize}")

    return model, accuracy


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    model, acc = main()