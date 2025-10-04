#!/usr/bin/env python3
"""
ORKHESTRA - SCRIPT PRINCIPAL DE ENTRENAMIENTO
============================================
Script unificado para entrenar el sistema de fusión Orkhestra y modelos individuales.

Orkhestra utiliza una arquitectura híbrida con:
- Modelo Parcial (Scikit-learn RandomForest) para casos extremos/seguros
- Modelo Total (TensorFlow) para cobertura completa
- Sistema de Fusión inteligente basado en confianza

Uso:
    python scripts/main_training.py --model orkhestra   # Sistema completo Orkhestra
    python scripts/main_training.py --model all         # Todos los modelos individuales
    python scripts/main_training.py --model total       # Solo modelo total
    python scripts/main_training.py --model partial     # Solo modelo parcial
    python scripts/main_training.py --model hybrid      # Solo modelo híbrido clásico
"""

import sys
import os
import argparse
from typing import Tuple, Optional

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.training.train_total import main as train_total_main
from scripts.training.train_partial import main as train_partial_main
from scripts.training.train_hybrid import main as train_hybrid_main


def train_orkhestra() -> None:
    """Entrena el sistema completo Orkhestra (recomendado)"""
    print("🎼 ENTRENAMIENTO DEL SISTEMA ORKHESTRA")
    print("=" * 50)
    print("Orkhestra - Sistema de Fusión Inteligente")
    print("Arquitectura: Partial (ScikitLearn) + Total (TensorFlow) + Fusión")
    print("=" * 50)

    results = {}

    try:
        print("\n1️⃣ Entrenando Modelo Parcial (Scikit-learn RandomForest)...")
        print("   → Especializado en casos extremos y predicciones seguras")
        model_partial, acc_partial = train_partial_main()
        results['partial'] = acc_partial
        print(f"✅ Modelo Parcial: {acc_partial:.4f} ({acc_partial*100:.2f}%)")

        print("\n2️⃣ Entrenando Modelo Total (TensorFlow Neural Network)...")
        print("   → Cobertura completa para todos los casos")
        model_total, acc_total = train_total_main()
        results['total'] = acc_total
        print(f"✅ Modelo Total: {acc_total:.4f} ({acc_total*100:.2f}%)")

        print("\n3️⃣ Entrenando Sistema de Fusión Orkhestra...")
        print("   → Combinando ambos modelos con lógica de confianza")
        model_orkhestra, metrics = train_hybrid_main()
        results['orkhestra'] = metrics.get('accuracy', 0.0) if isinstance(metrics, dict) else metrics
        print(f"✅ Sistema Orkhestra: {results['orkhestra']:.4f} ({results['orkhestra']*100:.2f}%)")

        # Mostrar métricas específicas de Orkhestra
        if isinstance(metrics, dict):
            print("\n📊 MÉTRICAS DETALLADAS DE ORKHESTRA:")
            print("-" * 40)
            if 'fusion_analysis' in metrics:
                fusion = metrics['fusion_analysis']
                print(f"   Uso Modelo Parcial: {fusion.get('partial_usage', 0)*100:.1f}%")
                print(f"   Uso Modelo Total:   {fusion.get('total_usage', 0)*100:.1f}%")
                print(f"   Confianza Media:    {fusion.get('avg_confidence', 0):.3f}")
            if 'threshold_analysis' in metrics:
                threshold = metrics['threshold_analysis']
                print(f"   Umbral Óptimo:      {threshold.get('optimal_threshold', 0.9):.3f}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento Orkhestra: {e}")
        return

    # Resumen final
    print("\n" + "=" * 50)
    print("🎼 RESUMEN FINAL - SISTEMA ORKHESTRA ENTRENADO")
    print("=" * 50)
    for model_name, accuracy in results.items():
        icon = "🎼" if model_name == "orkhestra" else "🤖" if model_name == "total" else "🎯"
        print(f"   {icon} {model_name.upper():>12}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\n🏆 SISTEMA ORKHESTRA LISTO - Fusión Inteligente Activada")


def train_all_models() -> None:
    """Entrena todos los modelos individuales (sin Orkhestra)"""
    print("🚀 ENTRENAMIENTO COMPLETO DE MODELOS INDIVIDUALES")
    print("=" * 50)

    results = {}

    try:
        print("\n1️⃣ Entrenando Modelo Total...")
        model_total, acc_total = train_total_main()
        results['total'] = acc_total
        print(f"✅ Modelo Total completado: {acc_total:.4f}")

        print("\n2️⃣ Entrenando Modelo Parcial...")
        model_partial, acc_partial = train_partial_main()
        results['partial'] = acc_partial
        print(f"✅ Modelo Parcial completado: {acc_partial:.4f}")

        print("\n3️⃣ Entrenando Modelo Híbrido Clásico...")
        model_hybrid, acc_hybrid = train_hybrid_main()
        results['hybrid'] = acc_hybrid if isinstance(acc_hybrid, (int, float)) else acc_hybrid.get('accuracy', 0.0)
        print(f"✅ Modelo Híbrido completado: {results['hybrid']:.4f}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return

    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN FINAL DE ENTRENAMIENTOS")
    print("=" * 50)
    for model_name, accuracy in results.items():
        print(f"   {model_name.upper():>10}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Determinar mejor modelo
    best_model = max(results.keys(), key=lambda k: results[k])
    best_acc = results[best_model]
    print(f"\n🏆 MEJOR MODELO: {best_model.upper()} con {best_acc:.4f} ({best_acc*100:.2f}%)")


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Entrenar sistema Orkhestra y modelos de clasificación de exoplanetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/main_training.py --model orkhestra  # Sistema Orkhestra (RECOMENDADO)
  python scripts/main_training.py --model all        # Entrenar todos los modelos individuales
  python scripts/main_training.py --model total      # Solo modelo total (TensorFlow)
  python scripts/main_training.py --model partial    # Solo modelo parcial (Scikit-learn)
  python scripts/main_training.py --model hybrid     # Solo modelo híbrido clásico

ORKHESTRA - Sistema de Fusión Inteligente:
  ✨ Combina Scikit-learn (precisión) + TensorFlow (cobertura)
  ✨ Decisiones basadas en confianza con umbral adaptativo
  ✨ Optimización automática del rendimiento
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['orkhestra', 'all', 'total', 'partial', 'hybrid'],
        required=True,
        help='Modelo(s) a entrenar (orkhestra recomendado)'
    )

    args = parser.parse_args()

    if args.model == 'orkhestra':
        train_orkhestra()
    elif args.model == 'all':
        train_all_models()
    elif args.model == 'total':
        print("🌐 Entrenando solo Modelo Total (TensorFlow)...")
        model, accuracy = train_total_main()
        print(f"✅ Completado: {accuracy:.4f} ({accuracy*100:.2f}%)")
    elif args.model == 'partial':
        print("🎯 Entrenando solo Modelo Parcial (Scikit-learn)...")
        model, accuracy = train_partial_main()
        print(f"✅ Completado: {accuracy:.4f} ({accuracy*100:.2f}%)")
    elif args.model == 'hybrid':
        print("🤖 Entrenando solo Modelo Híbrido Clásico...")
        model, accuracy = train_hybrid_main()
        accuracy_value = accuracy if isinstance(accuracy, (int, float)) else accuracy.get('accuracy', 0.0)
        print(f"✅ Completado: {accuracy_value:.4f} ({accuracy_value*100:.2f}%)")


if __name__ == "__main__":
    main()