#!/usr/bin/env python3
"""
SCRIPT DE PREDICCIÓN UNIVERSAL
===============================
Script unificado para predicción con cualquier modelo
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.evaluators import TotalModelEvaluator, PartialModelEvaluator, HybridModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Predictor Universal para Modelos de Exoplanetas')
    parser.add_argument('--model', choices=['partial', 'total', 'hybrid'], required=True,
                       help='Tipo de modelo a usar: partial, total o hybrid')

    args = parser.parse_args()

    print("🚀 PREDICTOR UNIVERSAL")
    print("="*50)
    print(f"📊 Cargando datos de prueba...")

    # Mapeo de modelos a evaluadores
    evaluators = {
        'partial': PartialModelEvaluator,
        'total': TotalModelEvaluator,
        'hybrid': HybridModelEvaluator
    }

    # Mapeo de archivos de modelo
    model_files = {
        'partial': 'saved_models/partial_model.pkl',
        'total': 'saved_models/total_model.pkl',
        'hybrid': 'saved_models/hybrid_tf_model.keras'
    }

    # Verificar que existen los archivos del modelo
    required_files = {
        'partial': ['saved_models/partial_model.pkl'],
        'total': ['saved_models/total_model.pkl'],
        'hybrid': [
            'saved_models/partial_model.pkl',
            'saved_models/total_model.pkl',
            'saved_models/hybrid_tf_model.keras',
            'saved_models/hybrid_tf_scaler.pkl'
        ]
    }

    for file_path in required_files[args.model]:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra {file_path}")
            print(f"   Ejecuta scripts/training/train_{args.model}.py primero")
            return

    # Crear evaluador y realizar predicción
    print(f"🎯 PREDICCIÓN CON MODELO {args.model.upper()}")
    print("="*40)

    evaluator = evaluators[args.model]()
    predictions, accuracy = evaluator.evaluate(model_files[args.model])

if __name__ == "__main__":
    main()