#!/usr/bin/env python3
"""
INFORMACIÓN DEL PROYECTO
========================
Muestra información sobre la estructura del proyecto y modelos disponibles.
"""

import sys
import os
import glob
from pathlib import Path

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_project_structure():
    """Muestra la estructura organizada del proyecto"""
    print("📁 ESTRUCTURA DEL PROYECTO")
    print("=" * 50)
    
    structure = {
        "📦 Raíz del proyecto": [
            "train_*.py (wrappers de compatibilidad)",
            "predict_model.py (wrapper de compatibilidad)",
            "clean_project.py (wrapper de compatibilidad)"
        ],
        "🔧 src/ - Código fuente modular": [
            "trainers/ - Clases de entrenamiento con herencia",
            "evaluators/ - Clases de evaluación",
            "models/ - Definiciones de modelos",
            "utils/ - Utilidades compartidas"
        ],
        "📜 scripts/ - Scripts organizados": [
            "training/ - Scripts de entrenamiento específicos",
            "prediction/ - Scripts de predicción",
            "utils/ - Utilidades y limpieza",
            "main_training.py - Script principal de entrenamiento",
            "main_prediction.py - Script principal de predicción"
        ],
        "💾 Datos y modelos": [
            "data/ - Conjunto de datos",
            "saved_models/ - Modelos entrenados"
        ]
    }
    
    for category, items in structure.items():
        print(f"\n{category}")
        for item in items:
            print(f"   • {item}")


def show_available_models():
    """Muestra información sobre los modelos disponibles"""
    print("\n🤖 MODELOS DISPONIBLES")
    print("=" * 50)
    
    models_info = {
        "🌐 Modelo Total": {
            "descripción": "Modelo base que clasifica todos los casos",
            "trainer": "src/trainers/total_trainer.py",
            "evaluator": "src/evaluators/model_evaluators.py",
            "script": "scripts/training/train_total.py",
            "wrapper": "train_total_model.py"
        },
        "🎯 Modelo Parcial": {
            "descripción": "Modelo con cobertura parcial y alta confianza",
            "trainer": "src/trainers/partial_trainer.py", 
            "evaluator": "src/evaluators/model_evaluators.py",
            "script": "scripts/training/train_partial.py",
            "wrapper": "train_partial_model.py"
        },
        "🤖 Modelo Híbrido": {
            "descripción": "Sistema híbrido con cascada y stacking (TensorFlow)",
            "trainer": "src/trainers/hybrid_trainer.py",
            "evaluator": "src/evaluators/model_evaluators.py", 
            "script": "scripts/training/train_hybrid.py",
            "wrapper": "train_hybrid_model.py"
        }
    }
    
    for model_name, info in models_info.items():
        print(f"\n{model_name}")
        print(f"   📄 {info['descripción']}")
        print(f"   🏗️  Trainer: {info['trainer']}")
        print(f"   📊 Evaluator: {info['evaluator']}")
        print(f"   📜 Script: {info['script']}")
        print(f"   🔗 Wrapper: {info['wrapper']}")


def check_saved_models():
    """Verifica qué modelos están guardados"""
    print("\n💾 MODELOS GUARDADOS")
    print("=" * 50)
    
    saved_models_dir = "saved_models"
    if not os.path.exists(saved_models_dir):
        print("❌ Directorio saved_models/ no encontrado")
        return
    
    model_files = {
        "🌐 Modelo Total": [
            "total_model.pkl",
            "total_scaler.pkl", 
            "total_encoder.pkl",
            "total_feature_importance.pkl",
            "total_feature_names.pkl"
        ],
        "🎯 Modelo Parcial": [
            "partial_model.pkl",
            "partial_scaler.pkl"
        ],
        "🤖 Modelo Híbrido": [
            "hybrid_tf_model.keras",
            "hybrid_tf_scaler.pkl",
            "hybrid_tf_enhanced_model.keras", 
            "hybrid_tf_enhanced_scaler.pkl"
        ]
    }
    
    for model_name, files in model_files.items():
        print(f"\n{model_name}")
        for file in files:
            file_path = os.path.join(saved_models_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                print(f"   ✅ {file} ({size_str})")
            else:
                print(f"   ❌ {file} (no encontrado)")


def show_usage_examples():
    """Muestra ejemplos de uso"""
    print("\n🚀 EJEMPLOS DE USO")
    print("=" * 50)
    
    examples = [
        ("Entrenar todos los modelos:", "python scripts/main_training.py --model all"),
        ("Entrenar modelo específico:", "python scripts/main_training.py --model total"),
        ("Evaluar todos los modelos:", "python scripts/main_prediction.py --compare"),
        ("Evaluar modelo específico:", "python scripts/main_prediction.py --model hybrid"),
        ("Usar wrappers de compatibilidad:", "python train_total_model.py"),
        ("Limpiar archivos temporales:", "python scripts/utils/clean_project.py"),
        ("Ver información del proyecto:", "python scripts/project_info.py")
    ]
    
    for description, command in examples:
        print(f"\n📌 {description}")
        print(f"   {command}")


def main():
    """Función principal"""
    print("ℹ️  INFORMACIÓN COMPLETA DEL PROYECTO")
    print("=" * 60)
    
    show_project_structure()
    show_available_models()
    check_saved_models()
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("✨ Proyecto organizado con arquitectura modular y herencia")
    print("🔄 Compatibilidad hacia atrás mantenida con wrappers") 
    print("📊 Scripts principales unificados disponibles")


if __name__ == "__main__":
    main()