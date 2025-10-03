#!/usr/bin/env python3
"""
SCRIPT PRINCIPAL DE PREDICCIÓN
==============================
Script unificado para realizar predicciones con todos los modelos.

Uso:
    python scripts/main_prediction.py --model all
    python scripts/main_prediction.py --model total
    python scripts/main_prediction.py --model partial
    python scripts/main_prediction.py --model hybrid
    python scripts/main_prediction.py --compare
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


def predict_all_models() -> Dict[str, Dict[str, Any]]:
    """Realiza predicciones con todos los modelos y compara resultados"""
    print("🚀 PREDICCIÓN CON TODOS LOS MODELOS")
    print("=" * 50)
    
    results = {}
    
    try:
        # Modelo Total
        print("\n1️⃣ Evaluando Modelo Total...")
        total_evaluator = TotalModelEvaluator()
        total_evaluator.load_test_data()
        total_evaluator.load_model('saved_models/total_model.pkl')
        predictions = total_evaluator.predict()
        accuracy, f1 = total_evaluator.calculate_metrics(predictions)
        results['total'] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Modelo Parcial
        print("\n2️⃣ Evaluando Modelo Parcial...")
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
        
        # Modelo Híbrido
        print("\n3️⃣ Evaluando Modelo Híbrido...")
        hybrid_evaluator = HybridModelEvaluator()
        hybrid_evaluator.load_test_data()
        hybrid_evaluator.load_model('saved_models/hybrid_tf_model.keras')
        predictions = hybrid_evaluator.predict()
        accuracy, f1 = hybrid_evaluator.calculate_metrics(predictions)
        results['hybrid'] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        return results
    
    return results


def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    """Compara los resultados de todos los modelos"""
    print("\n" + "=" * 60)
    print("📊 COMPARACIÓN DETALLADA DE MODELOS")
    print("=" * 60)
    
    print(f"{'Modelo':<15} {'Accuracy':<12} {'F1-Score':<12} {'Características'}")
    print("-" * 60)
    
    for model_name, result in results.items():
        accuracy = result.get('accuracy', 0)
        f1_score = result.get('f1_score', 0)
        
        # Características especiales de cada modelo
        features = ""
        if model_name == 'partial':
            coverage = result.get('coverage', 0)
            features = f"Coverage: {coverage:.2%}"
        elif model_name == 'hybrid':
            features = "Cascada + Stacking"
        else:
            features = "Modelo base"
            
        print(f"{model_name.upper():<15} {accuracy:.4f} ({accuracy:.2%}) {f1_score:.4f} {features}")
    
    # Determinar mejor modelo
    best_model = max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
    best_acc = results[best_model]['accuracy']
    print(f"\n🏆 MEJOR MODELO: {best_model.upper()} con {best_acc:.4f} ({best_acc*100:.2f}%)")


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Realizar predicciones con modelos de clasificación de exoplanetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/main_prediction.py --model all       # Evaluar todos los modelos
  python scripts/main_prediction.py --model total     # Solo modelo total
  python scripts/main_prediction.py --model partial   # Solo modelo parcial
  python scripts/main_prediction.py --model hybrid    # Solo modelo híbrido
  python scripts/main_prediction.py --compare         # Comparar todos los modelos
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--model',
        type=str,
        choices=['all', 'total', 'partial', 'hybrid'],
        help='Modelo(s) a evaluar'
    )
    group.add_argument(
        '--compare',
        action='store_true',
        help='Comparar todos los modelos'
    )
    
    args = parser.parse_args()
    
    if args.compare or args.model == 'all':
        results = predict_all_models()
        if results:
            compare_models(results)
    else:
        if args.model == 'total':
            print("🌐 Evaluando solo Modelo Total...")
            evaluator = TotalModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/total_model.pkl')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        elif args.model == 'partial':
            print("🎯 Evaluando solo Modelo Parcial...")
            evaluator = PartialModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/partial_model.pkl')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            coverage = len([p for p in predictions if p != 'Unknown']) / len(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"🎯 Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
            
        elif args.model == 'hybrid':
            print("🤖 Evaluando solo Modelo Híbrido...")
            evaluator = HybridModelEvaluator()
            evaluator.load_test_data()
            evaluator.load_model('saved_models/hybrid_tf_model.keras')
            predictions = evaluator.predict()
            accuracy, f1 = evaluator.calculate_metrics(predictions)
            print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()