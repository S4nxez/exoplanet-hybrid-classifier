#!/usr/bin/env python3
"""
🚀 KOI MAIN TRAINING SCRIPT
===========================
Script principal para entrenar el sistema KOI completo con Director MEJORADO.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
import time
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Importar componentes del sistema
from models.koi_randomforest import KOIRandomForest
from models.koi_tensorflow import KOITensorFlow
from core.director import KOIDirector
from utils.data_utils import load_and_prepare_data
from config.koi_config import KOIConfig, LogConfig

# Configurar logging
log_dir = Path(LogConfig.LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LogConfig.LOG_LEVEL),
    format=LogConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LogConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Función principal de entrenamiento"""
    logger.info("="*80)
    logger.info("🚀 INICIANDO ENTRENAMIENTO SISTEMA KOI COMPLETO")
    logger.info("="*80)
    logger.info(f"Configuración:")
    logger.info(f"   Random Forest: {vars(RFConfig)}")
    logger.info(f"   TensorFlow: Arquitectura {TFConfig.HIDDEN_LAYERS}")
    logger.info(f"   Director: Umbrales mejorados para 15-20% RF usage")
    logger.info("="*80)
    print()

    start_time = time.time()

    try:
        # =====================================================================
        # FASE 1: CARGAR Y PREPARAR DATOS
        # =====================================================================
        logger.info("📁 FASE 1: Carga de datos")
        logger.info("-"*80)

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        # Dividir train en train/val para los modelos individuales
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=KOIConfig.RANDOM_SEED,
            stratify=y_train
        )

        logger.info(f"   Train final: {len(X_train_split)} muestras")
        logger.info(f"   Validation: {len(X_val)} muestras")
        logger.info(f"   Test: {len(X_test)} muestras")
        print()

        # =====================================================================
        # FASE 2: ENTRENAR RANDOM FOREST
        # =====================================================================
        logger.info("🌲 FASE 2: Entrenamiento Random Forest")
        logger.info("-"*80)

        rf_model = KOIRandomForest()
        rf_model.fit(X_train_split, y_train_split, X_val, y_val)

        # Evaluar RF en test
        rf_results = rf_model.evaluate(X_test, y_test)
        rf_test_acc = rf_results['accuracy']

        logger.info(f"✅ Random Forest completado - Accuracy test: {rf_test_acc*100:.2f}%")
        print()

        # =====================================================================
        # FASE 3: ENTRENAR TENSORFLOW
        # =====================================================================
        logger.info("🧠 FASE 3: Entrenamiento TensorFlow MEJORADO")
        logger.info("-"*80)

        tf_model = KOITensorFlow()
        tf_model.fit(X_train_split, y_train_split, X_val, y_val)

        # Evaluar TF en test
        tf_results = tf_model.evaluate(X_test, y_test)
        tf_test_acc = tf_results['accuracy']

        logger.info(f"✅ TensorFlow completado - Accuracy test: {tf_test_acc*100:.2f}%")
        print()

        # =====================================================================
        # FASE 4: ENTRENAR DIRECTOR
        # =====================================================================
        logger.info("🎯 FASE 4: Entrenamiento Director MEJORADO")
        logger.info("-"*80)
        logger.info("⚠️  OBJETIVO: Usar RF en 15-20% de casos")

        # Crear Director
        director = KOIDirector()

        # Asignar modelos al Director
        director.rf_model = rf_model.model
        director.rf_scaler = rf_model.scaler
        director.tf_model = tf_model.model
        director.tf_scaler = tf_model.scaler
        director.is_trained = True

        logger.info("✅ Director configurado con modelos entrenados")

        # Evaluar Director en test
        logger.info("\n🔍 Evaluando Director en conjunto de test...")

        director_preds = []
        rf_count = 0
        tf_count = 0
        rf_correct = 0
        tf_correct = 0

        for i in range(len(X_test)):
            sample = X_test[i:i+1]
            pred, model_used = director.predict(sample, return_model_choice=True)
            director_preds.append(pred[0])

            if model_used == 'RandomForest':
                rf_count += 1
                if pred[0] == y_test[i]:
                    rf_correct += 1
            else:
                tf_count += 1
                if pred[0] == y_test[i]:
                    tf_correct += 1

        director_preds = np.array(director_preds)
        director_acc = accuracy_score(y_test, director_preds)

        rf_pct = (rf_count / len(X_test)) * 100
        tf_pct = (tf_count / len(X_test)) * 100

        logger.info(f"\n📊 Resultados del Director:")
        logger.info(f"   Accuracy: {director_acc*100:.2f}%")
        logger.info(f"   RF usado: {rf_count}/{len(X_test)} ({rf_pct:.2f}%)")
        logger.info(f"   TF usado: {tf_count}/{len(X_test)} ({tf_pct:.2f}%)")

        if rf_count > 0:
            rf_acc_when_chosen = (rf_correct / rf_count) * 100
            logger.info(f"   RF accuracy cuando elegido: {rf_acc_when_chosen:.2f}%")

        if tf_count > 0:
            tf_acc_when_chosen = (tf_correct / tf_count) * 100
            logger.info(f"   TF accuracy cuando elegido: {tf_acc_when_chosen:.2f}%")

        # Verificar objetivo
        if 15 <= rf_pct <= 20:
            logger.info(f"   ✅ OBJETIVO ALCANZADO: RF usage = {rf_pct:.2f}% (meta: 15-20%)")
        elif rf_pct < 15:
            logger.info(f"   ⚠️  RF usage bajo: {rf_pct:.2f}% (ajustar umbrales)")
        else:
            logger.info(f"   ⚠️  RF usage alto: {rf_pct:.2f}% (ajustar umbrales)")

        print()

        # =====================================================================
        # FASE 5: GUARDAR MODELOS
        # =====================================================================
        logger.info("💾 FASE 5: Guardando modelos")
        logger.info("-"*80)

        models_dir = KOIConfig.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelos individuales
        rf_model.save(models_dir)
        tf_model.save(models_dir)

        # Guardar metadata del Director
        director_metadata = {
            'rf_usage_percentage': float(rf_pct),
            'tf_usage_percentage': float(tf_pct),
            'rf_count': int(rf_count),
            'tf_count': int(tf_count),
            'director_accuracy': float(director_acc),
            'rf_accuracy': float(rf_test_acc),
            'tf_accuracy': float(tf_test_acc)
        }

        metadata_path = models_dir / 'koi_director_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(director_metadata, f, indent=2)

        logger.info(f"✅ Modelos guardados en {models_dir}")
        print()

        # =====================================================================
        # FASE 6: RESUMEN FINAL
        # =====================================================================
        elapsed_time = time.time() - start_time

        logger.info("="*80)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*80)
        logger.info(f"\n🎯 ACCURACIES EN TEST:")
        logger.info(f"   Random Forest: {rf_test_acc*100:.2f}%")
        logger.info(f"   TensorFlow:    {tf_test_acc*100:.2f}%")
        logger.info(f"   Director:      {director_acc*100:.2f}%")

        logger.info(f"\n🔀 USO DE MODELOS:")
        logger.info(f"   RF: {rf_count}/{len(X_test)} ({rf_pct:.2f}%)")
        logger.info(f"   TF: {tf_count}/{len(X_test)} ({tf_pct:.2f}%)")

        # Calcular potencial máximo
        rf_correct_all = (rf_results['predictions'] == y_test)
        tf_correct_all = (tf_results['predictions'] == y_test)
        at_least_one = (rf_correct_all | tf_correct_all).sum()
        max_possible = at_least_one / len(y_test) * 100

        logger.info(f"\n🎯 POTENCIAL:")
        logger.info(f"   Máximo teórico: {max_possible:.2f}%")
        logger.info(f"   Margen de mejora: {max_possible - director_acc*100:.2f} puntos")

        logger.info(f"\n⏱️  TIEMPO TOTAL: {elapsed_time/60:.2f} minutos")
        logger.info("="*80)
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Importar configs aquí para evitar problemas de importación circular
    from config.koi_config import RFConfig, TFConfig, DirectorConfig

    main()
