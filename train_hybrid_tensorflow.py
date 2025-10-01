#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO DEL MODELO HÍBRIDO CON TENSORFLOW
===============================================
Script para entrenar el modelo híbrido que combina scikit-learn + TensorFlow
"""

import sys
import os
import numpy as np

# Agregar la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_processor import DataProcessor
from models.partial_coverage import PartialCoverageModel
from models.total_coverage import TotalCoverageModel
from models.tensorflow_hybrid import TensorFlowHybridModel

def main():
    print("🚀 ENTRENAMIENTO DE MODELO HÍBRIDO CON TENSORFLOW")
    print("=" * 70)

    # Inicializar procesador de datos
    data_processor = DataProcessor()

    # Cargar y procesar datos
    print("📊 Cargando datos...")
    data = data_processor.load_clean_data('data/dataset.csv')
    print(f"✅ Datos cargados: {len(data)} registros")
    print(f"📈 Características: {len(data_processor.features)}")

    # Mostrar distribución
    class_counts = data['binary_class'].value_counts()
    print("📊 Distribución de clases:")
    for class_name, count in class_counts.items():
        print(f"  • {class_name}: {count} ({count/len(data)*100:.1f}%)")

    # Preparar datos
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_split(data, test_size=0.2, random_state=42)
    print(f"\n📊 División de datos:")
    print(f"  • Entrenamiento: {len(X_train)} muestras")
    print(f"  • Test: {len(X_test)} muestras")

    # === PASO 1: ENTRENAR MODELOS BASE ===
    print(f"\n🎯 PASO 1: ENTRENANDO MODELOS BASE")
    print("=" * 50)

    # Entrenar modelo total
    print("⚡ Entrenando modelo total (RandomForest)...")
    total_model = TotalCoverageModel()
    total_acc_train = total_model.train(X_train, y_train)
    total_preds_test = total_model.predict(X_test)
    from sklearn.metrics import accuracy_score
    total_acc_test = accuracy_score(y_test, total_preds_test)
    print(f"✅ Modelo total - Test: {total_acc_test:.4f} ({total_acc_test*100:.2f}%)")

    # Entrenar modelo parcial
    print("\n🎯 Entrenando modelo parcial (GradientBoosting)...")
    # Identificar casos extremos en entrenamiento
    extreme_indices_train = data_processor.identify_extreme_cases(
        data.loc[data.index.isin(pd.DataFrame(X_train, columns=data_processor.features).index)]
    )

    if len(extreme_indices_train) > 0:
        # Obtener datos extremos del conjunto completo
        extreme_data = data.loc[extreme_indices_train].copy()
        # Dividir casos extremos en train/test
        X_extreme_all = extreme_data[data_processor.features].values
        y_extreme_all = extreme_data['binary_class'].values

        # Entrenar modelo parcial con todos los casos extremos
        partial_model = PartialCoverageModel()
        partial_acc_train = partial_model.train(X_extreme_all, y_extreme_all)

        # Evaluar en casos extremos del test
        extreme_indices_test = [i for i in range(len(X_test)) if data_processor.is_extreme_case(X_test[i])]
        if len(extreme_indices_test) > 0:
            X_extreme_test = X_test[extreme_indices_test]
            y_extreme_test = y_test[extreme_indices_test]
            partial_preds_test = partial_model.predict(X_extreme_test)
            partial_acc_test = accuracy_score(y_extreme_test, partial_preds_test)
            partial_coverage = len(extreme_indices_test) / len(y_test) * 100
            print(f"✅ Modelo parcial - Test: {partial_acc_test:.4f} ({partial_acc_test*100:.2f}%), Cobertura: {partial_coverage:.1f}%")
        else:
            print("⚠️ No hay casos extremos en el conjunto de test")
            partial_acc_test = 0
            partial_coverage = 0
    else:
        print("⚠️ No se encontraron casos extremos en entrenamiento")
        partial_model = None
        partial_acc_test = 0
        partial_coverage = 0

    # === PASO 2: ENTRENAR MODELO HÍBRIDO CON TENSORFLOW ===
    print(f"\n🤖 PASO 2: ENTRENANDO MODELO HÍBRIDO CON TENSORFLOW")
    print("=" * 60)

    # Crear modelo híbrido
    hybrid_model = TensorFlowHybridModel(
        partial_model=partial_model,
        total_model=total_model,
        data_processor=data_processor
    )

    # Entrenar modelo híbrido
    print("🚀 Iniciando entrenamiento del modelo híbrido...")
    hybrid_acc_train, history = hybrid_model.train(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # === PASO 3: EVALUACIÓN COMPLETA ===
    print(f"\n📊 PASO 3: EVALUACIÓN COMPLETA")
    print("=" * 40)

    # Evaluar modelo híbrido
    results = hybrid_model.evaluate(X_test, y_test)

    # Mostrar resultados
    print(f"\n🎉 RESULTADOS FINALES:")
    print("=" * 50)
    print(f"🤖 MODELO HÍBRIDO (TensorFlow):")
    print(f"   Precisión: {results['hybrid_accuracy']:.4f} ({results['hybrid_accuracy']*100:.2f}%)")
    print(f"   Muestras test: {results['test_samples']}")

    print(f"\n⚡ MODELO TOTAL (RandomForest):")
    print(f"   Precisión: {results['total_accuracy']:.4f} ({results['total_accuracy']*100:.2f}%)")
    print(f"   Cobertura: 100.0%")

    if results['partial_accuracy'] > 0:
        print(f"\n🎯 MODELO PARCIAL (GradientBoosting):")
        print(f"   Precisión: {results['partial_accuracy']:.4f} ({results['partial_accuracy']*100:.2f}%)")
        print(f"   Cobertura: {results['partial_coverage']:.1f}%")

    print(f"\n📈 MEJORAS:")
    improvement_total = results['improvement_over_total'] * 100
    print(f"   Híbrido vs Total: {improvement_total:+.2f} puntos porcentuales")

    if results['partial_accuracy'] > 0:
        improvement_partial = results['improvement_over_partial'] * 100
        print(f"   Híbrido vs Parcial: {improvement_partial:+.2f} puntos porcentuales")

    # Verificar si superamos el objetivo
    print(f"\n🎯 VERIFICACIÓN DEL OBJETIVO:")
    if results['hybrid_accuracy'] > results['total_accuracy']:
        print(f"✅ ¡ÉXITO! El modelo híbrido supera al modelo total")
        print(f"   {results['hybrid_accuracy']*100:.2f}% > {results['total_accuracy']*100:.2f}%")
    else:
        print(f"❌ El modelo híbrido no supera al modelo total")
        print(f"   {results['hybrid_accuracy']*100:.2f}% vs {results['total_accuracy']*100:.2f}%")

    # Mostrar arquitectura del modelo
    feature_info = hybrid_model.get_feature_importance()
    print(f"\n🏗️ ARQUITECTURA DEL MODELO:")
    print(f"   Capas: {feature_info['layers']}")
    print(f"   Parámetros totales: {feature_info['total_params']:,}")
    print(f"   Parámetros entrenables: {feature_info['trainable_params']:,}")

    # Guardar modelos
    print(f"\n💾 GUARDANDO MODELOS...")
    # Guardar modelos base
    total_model.save('saved_models/total_model.pkl', 'saved_models/total_scaler.pkl', 'saved_models/total_encoder.pkl')
    if partial_model:
        partial_model.save('saved_models/partial_model.pkl', 'saved_models/partial_scaler.pkl')

    # Guardar modelo híbrido
    hybrid_model.save('saved_models/hybrid_tf_model.keras', 'saved_models/hybrid_tf_scaler.pkl')

    print("✅ Todos los modelos guardados exitosamente")

    # Mostrar reporte de clasificación
    print(f"\n📋 REPORTE DE CLASIFICACIÓN DEL MODELO HÍBRIDO:")
    print(results['classification_report'])

    return {
        'hybrid_model': hybrid_model,
        'total_model': total_model,
        'partial_model': partial_model,
        'results': results
    }

if __name__ == "__main__":
    # Importar pandas aquí para evitar problemas de import
    import pandas as pd
    main()