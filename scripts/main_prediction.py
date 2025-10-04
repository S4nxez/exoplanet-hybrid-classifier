#!/usr/bin/env python3
"""
ORKHESTRA - SCRIPT PRINCIPAL DE PREDICCIÓN
==========================================
Script unificado para realizar predicciones con el sistema Orkhestra y modelos individuales.

Orkhestra utiliza una arquitectura híbrida con:
- Modelo Parcial (Scikit-learn RandomForest) para casos extremos/seguros
- Modelo Total (TensorFlow) para cobertura completa  
- Sistema de Fusión inteligente basado en confianza

Uso:
    python scripts/main_prediction.py --model orkhestra # Sistema completo Orkhestra
    python scripts/main_prediction.py --model all       # Evaluar todos los modelos
    python scripts/main_prediction.py --model total     # Solo modelo total
    python scripts/main_prediction.py --model partial   # Solo modelo parcial
    python scripts/main_prediction.py --model hybrid    # Solo modelo híbrido clásico
    python scripts/main_prediction.py --compare         # Comparar todos los modelos
"""

import sys
import os
import argparse
from typing import Dict, Any

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluators.model_evaluators import TotalModelEvaluator, PartialModelEvaluator, HybridModelEvaluator
from src.utils.data_processor import DataProcessor


def load_test_data():
    """Cargar datos de prueba"""
    print("📊 Cargando datos de prueba...")
    data_processor = DataProcessor()
    data = data_processor.load_clean_data('data/dataset.csv')
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_split(data)
    return X_test, y_test


def predict_orkhestra() -> Dict[str, Any]:
    """Realiza predicciones con el sistema completo Orkhestra"""
    print("🎼 PREDICCIÓN CON SISTEMA ORKHESTRA")
    print("=" * 50)
    print("Orkhestra - Sistema de Fusión Inteligente")
    print("Arquitectura: Partial (ScikitLearn) + Total (TensorFlow) + Fusión")
    print("=" * 50)
    
    results = {}
    
    try:
        print("\n🎼 Evaluando Sistema Orkhestra...")
        print("   → Fusión inteligente basada en confianza")
        
        # Usar el evaluador híbrido que ahora maneja Orkhestra
        orkhestra_evaluator = HybridModelEvaluator()
        orkhestra_evaluator.load_test_data()
        
        # Intentar cargar el modelo Orkhestra
        model_path = 'saved_models/hybrid_tf_enhanced_model.keras'
        if not os.path.exists(model_path):
            model_path = 'saved_models/hybrid_tf_model.keras'
            
        orkhestra_evaluator.load_model(model_path)
        predictions = orkhestra_evaluator.predict()
        accuracy, f1 = orkhestra_evaluator.calculate_metrics(predictions)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model_type': 'orkhestra',
            'predictions': predictions
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ✅ F1-Score: {f1:.4f}")
        
        # Intentar obtener métricas detalladas si están disponibles
        if hasattr(orkhestra_evaluator.model, 'get_fusion_analysis'):
            try:
                fusion_analysis = orkhestra_evaluator.model.get_fusion_analysis()
                results['fusion_analysis'] = fusion_analysis
                print("\n📊 ANÁLISIS DE FUSIÓN ORKHESTRA:")
                print("-" * 40)
                print(f"   Uso Modelo Parcial: {fusion_analysis.get('partial_usage', 0)*100:.1f}%")
                print(f"   Uso Modelo Total:   {fusion_analysis.get('total_usage', 0)*100:.1f}%")
                print(f"   Confianza Media:    {fusion_analysis.get('avg_confidence', 0):.3f}")
                print(f"   Umbral Confianza:   {fusion_analysis.get('confidence_threshold', 0.9):.3f}")
            except:
                print("   ℹ️  Métricas de fusión no disponibles en este modelo")
        
    except Exception as e:
        print(f"❌ Error durante la predicción Orkhestra: {e}")
        import traceback
        print("Traceback completo:")
        traceback.print_exc()
        results = {'accuracy': 0.0, 'f1_score': 0.0, 'error': str(e)}
    
    print("\n🎼 SISTEMA ORKHESTRA - Predicción completada")
    return results


def predict_all_models() -> Dict[str, Dict[str, Any]]:
    """Realiza predicciones con todos los modelos y compara resultados"""
    print("🚀 PREDICCIÓN CON TODOS LOS MODELOS")
    print("=" * 50)
    
    results = {}
    
    try:
        # Modelo Total
        print("\n1️⃣ Evaluando Modelo Total (TensorFlow)...")
        total_evaluator = TotalModelEvaluator()
        total_evaluator.load_test_data()
        total_evaluator.load_model('saved_models/total_model.pkl')
        predictions = total_evaluator.predict()
        accuracy, f1 = total_evaluator.calculate_metrics(predictions)
        results['total'] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Modelo Parcial
        print("\n2️⃣ Evaluando Modelo Parcial (Scikit-learn)...")
        partial_evaluator = PartialModelEvaluator()
        partial_evaluator.load_test_data()
        partial_evaluator.load_model('saved_models/partial_model.pkl')
        predictions = partial_evaluator.predict()
        accuracy, f1 = partial_evaluator.calculate_metrics(predictions)
        # Calcular coverage específico del modelo parcial
        coverage = len([p for p in predictions if p != 'Unknown']) / len(predictions)
        results['partial'] = {'accuracy': accuracy, 'f1_score': f1, 'coverage': coverage}
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
        
        # Modelo Híbrido Clásico
        print("\n3️⃣ Evaluando Modelo Híbrido Clásico...")
        hybrid_evaluator = HybridModelEvaluator()
        hybrid_evaluator.load_test_data()
        hybrid_evaluator.load_model('saved_models/hybrid_tf_model.keras')
        predictions = hybrid_evaluator.predict()
        accuracy, f1 = hybrid_evaluator.calculate_metrics(predictions)
        results['hybrid'] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Sistema Orkhestra
        print("\n4️⃣ Evaluando Sistema Orkhestra...")
        orkhestra_result = predict_orkhestra()
        results['orkhestra'] = orkhestra_result
        
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        return results
    
    return results


def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    """Compara los resultados de todos los modelos"""
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN DETALLADA DE MODELOS")
    print("=" * 70)
    
    print(f"{'Modelo':<15} {'Accuracy':<12} {'F1-Score':<12} {'Características'}")
    print("-" * 70)
    
    for model_name, result in results.items():
        accuracy = result.get('accuracy', 0)
        f1_score = result.get('f1_score', 0)
        
        # Características especiales de cada modelo
        features = ""
        icon = ""
        if model_name == 'partial':
            coverage = result.get('coverage', 0)
            features = f"Coverage: {coverage:.2%} | Precisión extrema"
            icon = "🎯"
        elif model_name == 'total':
            features = "Cobertura completa | TensorFlow"
            icon = "🤖"
        elif model_name == 'hybrid':
            features = "Cascada + Stacking | Clásico"
            icon = "⚙️"
        elif model_name == 'orkhestra':
            features = "Fusión Inteligente | Confianza adaptativa"
            icon = "🎼"
        else:
            features = "Modelo base"
            icon = "📦"
            
        print(f"{icon} {model_name.upper():<13} {accuracy:.4f} ({accuracy:.2%}) {f1_score:.4f} {features}")
    
    # Determinar mejor modelo
    best_model = max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
    best_acc = results[best_model]['accuracy']
    
    if best_model == 'orkhestra':
        print(f"\n🏆 MEJOR MODELO: 🎼 ORKHESTRA con {best_acc:.4f} ({best_acc*100:.2f}%)")
        print("    → Sistema de Fusión Inteligente líder en rendimiento")
    else:
        print(f"\n🏆 MEJOR MODELO: {best_model.upper()} con {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Mostrar análisis de fusión si está disponible para Orkhestra
    if 'orkhestra' in results and 'fusion_analysis' in results['orkhestra']:
        fusion = results['orkhestra']['fusion_analysis']
        print("\n🎼 ANÁLISIS DETALLADO DE ORKHESTRA:")
        print("-" * 40)
        print(f"   Decisiones Modelo Parcial: {fusion.get('partial_usage', 0)*100:.1f}%")
        print(f"   Decisiones Modelo Total:   {fusion.get('total_usage', 0)*100:.1f}%")
        print(f"   Confianza Promedio:        {fusion.get('avg_confidence', 0):.3f}")
        print(f"   Umbral de Confianza:       {fusion.get('confidence_threshold', 0.9):.3f}")


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Realizar predicciones con sistema Orkhestra y modelos de clasificación de exoplanetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/main_prediction.py --model orkhestra  # Sistema Orkhestra (RECOMENDADO)
  python scripts/main_prediction.py --model all        # Evaluar todos los modelos
  python scripts/main_prediction.py --model total      # Solo modelo total (TensorFlow)
  python scripts/main_prediction.py --model partial    # Solo modelo parcial (Scikit-learn)
  python scripts/main_prediction.py --model hybrid     # Solo modelo híbrido clásico
  python scripts/main_prediction.py --compare          # Comparar todos los modelos

ORKHESTRA - Sistema de Fusión Inteligente:
  ✨ Combina Scikit-learn (precisión) + TensorFlow (cobertura)
  ✨ Decisiones basadas en confianza con umbral adaptativo
  ✨ Optimización automática del rendimiento
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--model',
        type=str,
        choices=['orkhestra', 'all', 'total', 'partial', 'hybrid'],
        help='Modelo(s) a evaluar (orkhestra recomendado)'
    )
    group.add_argument(
        '--compare',
        action='store_true',
        help='Comparar todos los modelos incluyendo Orkhestra'
    )
    
    args = parser.parse_args()
    
    if args.compare or args.model == 'all':
        results = predict_all_models()
        if results:
            compare_models(results)
    else:
        if args.model == 'orkhestra':
            predict_orkhestra()
            
        elif args.model == 'total':
            print("🤖 Evaluando solo Modelo Total (TensorFlow)...")
            evaluator = TotalModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/total_model.pkl')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        elif args.model == 'partial':
            print("🎯 Evaluando solo Modelo Parcial (Scikit-learn)...")
            evaluator = PartialModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/partial_model.pkl')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            coverage = len([p for p in predictions if p != 'Unknown']) / len(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"🎯 Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
            
        elif args.model == 'hybrid':
            print("⚙️ Evaluando solo Modelo Híbrido Clásico...")
            evaluator = HybridModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/hybrid_tf_model.keras')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()