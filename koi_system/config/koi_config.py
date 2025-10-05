#!/usr/bin/env python3
"""
🔧 KOI SYSTEM CONFIGURATION
==========================
Configuración centralizada del sistema KOI.
"""

import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

class KOIConfig:
    """Configuración principal del sistema KOI"""

    # Directorios
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "saved_models"
    DATA_DIR = BASE_DIR.parent / "data"
    RESULTS_DIR = BASE_DIR / "results"
    LOGS_DIR = BASE_DIR / "logs"

    # Semillas para reproducibilidad
    RANDOM_SEED = 42

    # Configuración de datos
    TEST_SIZE = 0.2

    # Features del dataset KOI
    FEATURES = [
        'koi_period', 'koi_depth', 'koi_duration',
        'koi_prad', 'koi_teq', 'koi_insol',
        'koi_steff', 'koi_slogg', 'koi_srad'
    ]

# =============================================================================
# CONFIGURACIÓN RANDOM FOREST (CASOS SIMPLES/LINEALES)
# =============================================================================

class RFConfig:
    """Configuración para modelo RandomForest (casos simples)"""

    # Hiperparámetros optimizados
    N_ESTIMATORS = 200
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 1
    MAX_FEATURES = 'sqrt'
    CLASS_WEIGHT = 'balanced'

    # Configuración de entrenamiento
    BOOTSTRAP = True
    N_JOBS = -1

    # Archivos
    MODEL_FILE = "koi_randomforest_model.pkl"
    SCALER_FILE = "koi_randomforest_scaler.pkl"

# =============================================================================
# CONFIGURACIÓN TENSORFLOW (CASOS COMPLEJOS)
# =============================================================================

class TFConfig:
    """Configuración para modelo TensorFlow MEJORADO"""

    # Arquitectura MEJORADA - más profunda con batch normalization
    HIDDEN_LAYERS = [256, 128, 64, 32]  # 4 capas para mejor aprendizaje
    DROPOUT_RATE = 0.25
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'sigmoid'

    # Configuración de entrenamiento OPTIMIZADA
    LEARNING_RATE = 0.0005  # Learning rate reducido para estabilidad
    BATCH_SIZE = 64  # Batch size aumentado
    EPOCHS = 150  # Más épocas
    PATIENCE = 20  # Más paciencia para early stopping

    # Configuración de optimización
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    # Archivos
    MODEL_FILE = "koi_tensorflow_model.h5"
    SCALER_FILE = "koi_tensorflow_scaler.pkl"
    HISTORY_FILE = "koi_tensorflow_history.json"

# =============================================================================
# CONFIGURACIÓN DIRECTOR (SELECTOR DE MODELO)
# =============================================================================

class DirectorConfig:
    """Configuración para modelo Director MEJORADO"""

    # Umbrales MEJORADOS para decisión RF
    # OBJETIVO: Usar RF en 15-20% de casos
    RF_CONFIDENCE_THRESHOLD = 0.85  # Confianza mínima RF
    COMPLEXITY_THRESHOLD = 0.40  # Complejidad máxima para casos simples

    # Umbrales adicionales
    RF_CONFIDENCE_MEDIUM = 0.78  # Confianza media RF
    COMPLEXITY_VERY_SIMPLE = 0.35  # Casos muy simples
    CONFIDENCE_GAP = 0.15  # Gap entre RF y TF
    RF_CONFIDENCE_HIGH = 0.90  # Confianza muy alta RF
    COMPLEXITY_SIMPLE = 0.45  # Casos razonablemente simples

    # Para condición de consenso
    RF_CONFIDENCE_CONSENSUS = 0.80
    TF_CONFIDENCE_CONSENSUS = 0.75
    COMPLEXITY_CONSENSUS = 0.40

    # Archivos
    MODEL_FILE = "koi_director_model.h5"
    SCALER_FILE = "koi_director_scaler.pkl"
    METADATA_FILE = "koi_director_metadata.json"

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

class LogConfig:
    """Configuración de logging"""

    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/koi_system.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
