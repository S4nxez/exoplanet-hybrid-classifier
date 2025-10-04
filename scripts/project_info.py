#!/usr/bin/env python3
"""
ORKHESTRA - INFORMACIÓN DEL PROYECTO
===================================
Muestra información sobre el sistema Orkhestra y la estructura del proyecto.

Orkhestra es un sistema de fusión inteligente que combina:
- Modelo Parcial (Scikit-learn RandomForest) para casos extremos/seguros
- Modelo Total (TensorFlow) para cobertura completa
- Sistema de Fusión inteligente basado en confianza
"""

import sys
import os
import glob
from pathlib import Path

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_orkhestra_architecture():
    """Muestra la arquitectura del sistema Orkhestra"""
    print("🎼 ARQUITECTURA DEL SISTEMA ORKHESTRA")
    print("=" * 50)

    architecture = {
        "🎯 Modelo Parcial (Scikit-learn)": [
            "RandomForestClassifier especializado",
            "Alta precisión en casos extremos/seguros",
            "Confianza alta en predicciones específicas",
            "Manejo de casos 'Unknown' para baja confianza"
        ],
        "🤖 Modelo Total (TensorFlow)": [
            "Red neuronal para cobertura completa",
            "Procesa todos los casos sin excepción",
            "Arquitectura robusta y generalizable",
            "Backbone para decisiones complejas"
        ],
        "🎼 Sistema de Fusión Inteligente": [
            "Lógica basada en confianza adaptativa",
            "Umbral configurable (default: 0.9)",
            "Optimización automática de thresholds",
            "Métricas de análisis de fusión detalladas"
        ]
    }

    for component, features in architecture.items():
        print(f"\n{component}")
        for feature in features:
            print(f"   ✨ {feature}")

    print(f"\n🔄 FLUJO DE DECISIÓN ORKHESTRA:")
    print("   1. Modelo Parcial evalúa confianza")
    print("   2. Si confianza >= umbral → Usa predicción Parcial")
    print("   3. Si confianza < umbral → Usa predicción Total")
    print("   4. Registro de métricas y análisis de fusión")


def show_project_structure():
    """Muestra la estructura organizada del proyecto"""
    print("\n📁 ESTRUCTURA DEL PROYECTO ORKHESTRA")
    print("=" * 50)

    structure = {
        "📦 Raíz del proyecto": [
            "train_*.py (wrappers de compatibilidad)",
            "predict_model.py (wrapper de compatibilidad)",
            "clean_project.py (wrapper de compatibilidad)"
        ],
        "🔧 src/ - Código fuente modular": [
            "trainers/ - Clases de entrenamiento Orkhestra",
            "evaluators/ - Evaluadores con soporte de fusión",
            "models/ - Definiciones de modelos especializados",
            "utils/ - Utilidades compartidas"
        ],
        "📜 scripts/ - Scripts Orkhestra": [
            "training/ - Scripts de entrenamiento específicos",
            "prediction/ - Scripts de predicción",
            "utils/ - Utilidades y limpieza",
            "main_training.py - Script principal Orkhestra",
            "main_prediction.py - Script principal de evaluación"
        ],
        "💾 Datos y modelos": [
            "data/ - Conjunto de datos de exoplanetas",
            "saved_models/ - Modelos Orkhestra entrenados",
            "legacy/ - Archivos de versiones anteriores"
        ]
    }

    for category, items in structure.items():
        print(f"\n{category}")
        for item in items:
            print(f"   • {item}")


def show_available_models():
    """Muestra información sobre los modelos disponibles"""
    print("\n🤖 MODELOS DISPONIBLES EN ORKHESTRA")
    print("=" * 50)

    models_info = {
        "🎼 Sistema Orkhestra (RECOMENDADO)": {
            "descripción": "Sistema de fusión inteligente con confianza adaptativa",
            "trainer": "src/trainers/hybrid_trainer.py (OrkhestraftHybridTrainer)",
            "model": "src/models/tensorflow_hybrid.py (OrkhestraftHybridModel)",
            "script": "scripts/training/train_hybrid.py",
            "comando": "python scripts/main_training.py --model orkhestra"
        },
        "🎯 Modelo Parcial (Componente Orkhestra)": {
            "descripción": "RandomForest especializado en casos extremos/seguros",
            "trainer": "src/trainers/partial_trainer.py",
            "model": "src/models/partial_coverage.py",
            "script": "scripts/training/train_partial.py",
            "comando": "python scripts/main_training.py --model partial"
        },
        "🤖 Modelo Total (Componente Orkhestra)": {
            "descripción": "TensorFlow para cobertura completa de todos los casos",
            "trainer": "src/trainers/total_trainer.py",
            "model": "src/models/total_coverage.py",
            "script": "scripts/training/train_total.py",
            "comando": "python scripts/main_training.py --model total"
        },
        "⚙️ Modelo Híbrido Clásico (Legacy)": {
            "descripción": "Sistema híbrido anterior (mantenido por compatibilidad)",
            "trainer": "src/trainers/hybrid_trainer.py",
            "model": "src/models/tensorflow_hybrid.py",
            "script": "scripts/training/train_hybrid.py",
            "comando": "python scripts/main_training.py --model hybrid"
        }
    }

    for model_name, info in models_info.items():
        print(f"\n{model_name}")
        print(f"   📄 {info['descripción']}")
        print(f"   🏗️  Trainer: {info['trainer']}")
        print(f"   🧠 Model: {info['model']}")
        print(f"   📜 Script: {info['script']}")
        print(f"   ⚡ Comando: {info['comando']}")


def check_saved_models():
    """Verifica qué modelos están guardados"""
    print("\n💾 MODELOS ORKHESTRA GUARDADOS")
    print("=" * 50)

    saved_models_dir = "saved_models"
    if not os.path.exists(saved_models_dir):
        print("❌ Directorio saved_models/ no encontrado")
        return

    model_files = {
        "🎼 Sistema Orkhestra": [
            "hybrid_tf_enhanced_model.keras",
            "hybrid_tf_enhanced_scaler.pkl",
            "hybrid_tf_model.keras",
            "hybrid_tf_scaler.pkl"
        ],
        "🎯 Modelo Parcial": [
            "partial_model.pkl",
            "partial_scaler.pkl"
        ],
        "🤖 Modelo Total": [
            "total_model.pkl",
            "total_scaler.pkl",
            "total_encoder.pkl",
            "total_feature_importance.pkl",
            "total_feature_names.pkl"
        ],
        "📁 Modelos temporales": [
            "temp_best_hybrid_model.keras"
        ]
    }

    for model_name, files in model_files.items():
        print(f"\n{model_name}")
        for file in files:
            file_path = os.path.join(saved_models_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                status = "✅" if "enhanced" in file or "orkhestra" in file.lower() else "📦"
                print(f"   {status} {file} ({size_str})")
            else:
                print(f"   ❌ {file} (no encontrado)")


def show_usage_examples():
    """Muestra ejemplos de uso de Orkhestra"""
    print("\n🚀 EJEMPLOS DE USO - SISTEMA ORKHESTRA")
    print("=" * 50)

    examples = [
        ("🎼 Entrenar Sistema Orkhestra (RECOMENDADO):", "python scripts/main_training.py --model orkhestra"),
        ("🎼 Evaluar Sistema Orkhestra:", "python scripts/main_prediction.py --model orkhestra"),
        ("📊 Comparar todos los modelos:", "python scripts/main_prediction.py --compare"),
        ("🎯 Entrenar solo modelo parcial:", "python scripts/main_training.py --model partial"),
        ("🤖 Entrenar solo modelo total:", "python scripts/main_training.py --model total"),
        ("⚙️ Entrenar modelo híbrido clásico:", "python scripts/main_training.py --model hybrid"),
        ("🧹 Limpiar archivos temporales:", "python scripts/utils/clean_project.py"),
        ("ℹ️ Ver información del proyecto:", "python scripts/project_info.py")
    ]

    for description, command in examples:
        print(f"\n📌 {description}")
        print(f"   {command}")

    print(f"\n💡 FLUJO DE TRABAJO RECOMENDADO:")
    print("   1. python scripts/main_training.py --model orkhestra")
    print("   2. python scripts/main_prediction.py --model orkhestra")
    print("   3. python scripts/main_prediction.py --compare")


def show_orkhestra_features():
    """Muestra las características avanzadas de Orkhestra"""
    print("\n✨ CARACTERÍSTICAS AVANZADAS DE ORKHESTRA")
    print("=" * 50)

    features = {
        "🧠 Inteligencia de Fusión": [
            "Decisiones basadas en confianza",
            "Umbral adaptativo configurable",
            "Optimización automática de parámetros",
            "Análisis detallado de rendimiento"
        ],
        "📊 Métricas Avanzadas": [
            "Análisis de uso por modelo",
            "Confianza promedio de predicciones",
            "Distribución de decisiones",
            "Optimización de threshold automática"
        ],
        "🔧 Configurabilidad": [
            "Umbral de confianza personalizable",
            "Compatibilidad con modelos existentes",
            "Extensibilidad para nuevos algoritmos",
            "Integración transparente con pipeline"
        ],
        "🚀 Rendimiento": [
            "Combina precisión y cobertura óptimas",
            "Decisiones inteligentes por caso",
            "Reducción de falsos positivos/negativos",
            "Escalabilidad para grandes datasets"
        ]
    }

    for category, items in features.items():
        print(f"\n{category}")
        for item in items:
            print(f"   ✨ {item}")


def main():
    """Función principal"""
    print("🎼 ORKHESTRA - INFORMACIÓN COMPLETA DEL SISTEMA")
    print("=" * 60)
    print("Sistema de Fusión Inteligente para Clasificación de Exoplanetas")
    print("=" * 60)

    show_orkhestra_architecture()
    show_project_structure()
    show_available_models()
    check_saved_models()
    show_usage_examples()
    show_orkhestra_features()

    print("\n" + "=" * 60)
    print("🎼 ✨ ORKHESTRA - SISTEMA DE FUSIÓN INTELIGENTE ✨")
    print("🎯 Precisión extrema + 🤖 Cobertura completa = 🎼 Rendimiento superior")
    print("📊 Arquitectura modular | 🔄 Compatibilidad Legacy | 🚀 Escalable")
    print("=" * 60)


if __name__ == "__main__":
    main()