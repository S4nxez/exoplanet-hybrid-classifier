#!/usr/bin/env python3
"""
SCRIPT PRINCIPAL DE ENTRENAMIENTO
=================================
Script unificado para entrenar todos los modelos de forma individual o conjunta.

Uso:
    python scripts/main_training.py --model all
    python scripts/main_training.py --model total
    python scripts/main_training.py --model partial
    python scripts/main_training.py --model hybrid
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


def train_all_models() -> None:
    """Entrena todos los modelos en secuencia"""
    print("🚀 ENTRENAMIENTO COMPLETO DE TODOS LOS MODELOS")
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
        
        print("\n3️⃣ Entrenando Modelo Híbrido...")
        model_hybrid, acc_hybrid = train_hybrid_main()
        results['hybrid'] = acc_hybrid
        print(f"✅ Modelo Híbrido completado: {acc_hybrid:.4f}")
        
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
        description="Entrenar modelos de clasificación de exoplanetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/main_training.py --model all      # Entrenar todos los modelos
  python scripts/main_training.py --model total    # Solo modelo total
  python scripts/main_training.py --model partial  # Solo modelo parcial
  python scripts/main_training.py --model hybrid   # Solo modelo híbrido
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'total', 'partial', 'hybrid'],
        required=True,
        help='Modelo(s) a entrenar'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models()
    elif args.model == 'total':
        print("🌐 Entrenando solo Modelo Total...")
        model, accuracy = train_total_main()
        print(f"✅ Completado: {accuracy:.4f} ({accuracy*100:.2f}%)")
    elif args.model == 'partial':
        print("🎯 Entrenando solo Modelo Parcial...")
        model, accuracy = train_partial_main()
        print(f"✅ Completado: {accuracy:.4f} ({accuracy*100:.2f}%)")
    elif args.model == 'hybrid':
        print("🤖 Entrenando solo Modelo Híbrido...")
        model, accuracy = train_hybrid_main()
        print(f"✅ Completado: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()