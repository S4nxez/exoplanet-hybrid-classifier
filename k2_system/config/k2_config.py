#!/usr/bin/env python3
"""
🔧 K2 SYSTEM CONFIGURATION
=========================
Configuración centralizada del sistema K2 final.
"""

import numpy as np

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

class K2Config:
    """Configuración principal del sistema K2"""

    # Directorios
    MODEL_DIR = "models"
    DATA_DIR = "data"
    RESULTS_DIR = "results"

    # Semillas para reproducibilidad
    RANDOM_SEED = 42

    # Configuración de datos
    TRAIN_SIZE = 0.75
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.10

    # Threshold para decisión de modelo
    CONFIDENCE_THRESHOLD = 0.75  # Confianza para RandomForest
    UNCERTAINTY_THRESHOLD = 0.6  # Umbral de incertidumbre

    # Configuración de características
    N_FEATURES_SELECTED = 20
    FEATURE_SELECTION_METHOD = 'rfe'

    # Configuración de balanceo
    SMOTE_STRATEGY = 0.8
    SMOTE_K_NEIGHBORS = 5

# =============================================================================
# CONFIGURACIÓN RANDOM FOREST (CASOS SEGUROS)
# =============================================================================

class RFConfig:
    """Configuración para modelo RandomForest (casos seguros)"""

    # Hiperparámetros optimizados para alta precisión
    N_ESTIMATORS = 300
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 15
    MIN_SAMPLES_LEAF = 8
    MAX_FEATURES = 'sqrt'
    CLASS_WEIGHT = {0: 1, 1: 6}  # Reducido para evitar falsos positivos

    # Configuración de entrenamiento
    BOOTSTRAP = True
    OOB_SCORE = True
    N_JOBS = -1

    # Threshold para predicciones
    PREDICTION_THRESHOLD = 0.65  # Alto para alta precisión

# =============================================================================
# CONFIGURACIÓN TENSORFLOW (CASOS COMPLEJOS)
# =============================================================================

class TFConfig:
    """Configuración para modelos TensorFlow"""

    # Arquitectura de red neuronal
    HIDDEN_LAYERS = [64, 32, 16]
    DROPOUT_RATE = 0.3
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'sigmoid'

    # Configuración de entrenamiento
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 15  # Early stopping

    # Configuración de optimización
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy', 'precision', 'recall']

    # Threshold para predicciones
    PREDICTION_THRESHOLD = 0.5

# =============================================================================
# CONFIGURACIÓN DIRECTOR (SELECTOR DE MODELO)
# =============================================================================

class DirectorConfig:
    """Configuración para modelo Director"""

    # Arquitectura simplificada para decisión rápida
    HIDDEN_LAYERS = [32, 16]
    DROPOUT_RATE = 0.2
    ACTIVATION = 'relu'

    # Configuración de entrenamiento
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 10

    # Clases de salida: 0=RandomForest, 1=TensorFlow
    N_CLASSES = 2
    OUTPUT_ACTIVATION = 'softmax'
    LOSS = 'sparse_categorical_crossentropy'

# =============================================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS
# =============================================================================

class FeatureConfig:
    """Configuración de características K2"""

    # Características básicas K2 (alineadas con el Director y los scalers guardados)
    BASE_FEATURES = [
        'pl_orbper',
        'pl_rade',
        'st_teff',
        'st_rad',
        'st_mass',
        'sy_dist',
        'pl_eqt',
        'pl_orbsmax',
    ]

    # Características derivadas deshabilitadas para evitar desalineación con los modelos persistidos
    DERIVED_FEATURES = []

    # Rangos para validación de datos
    FEATURE_RANGES = {
        'pl_orbper': (0.1, 1000),
        'pl_rade': (0.1, 50),
        'pl_trandep': (1, 50000),
        'st_teff': (2000, 10000),
        'sy_kepmag': (5, 20)
    }

# =============================================================================
# CONFIGURACIÓN DE EVALUACIÓN
# =============================================================================

class EvalConfig:
    """Configuración para evaluación del sistema"""

    # Métricas objetivo
    TARGET_PRECISION = 0.70    # Mínimo 70% precisión
    TARGET_RECALL = 0.85       # Mínimo 85% recall
    TARGET_F1 = 0.75           # Mínimo 75% F1-Score
    MAX_FALSE_POSITIVES = 80   # Máximo 80 falsos positivos

    # Configuración de validación cruzada
    CV_FOLDS = 5
    CV_SCORING = ['precision', 'recall', 'f1']

    # Configuración de gráficos
    FIGURE_SIZE = (12, 8)
    DPI = 300
    SAVE_FORMAT = 'png'

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

class LogConfig:
    """Configuración de logging"""

    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'k2_system.log'

    # Configuración de progreso
    SHOW_PROGRESS = True
    VERBOSE = 1

# =============================================================================
# CONFIGURACIÓN DE PRODUCCIÓN
# =============================================================================

class ProductionConfig:
    """Configuración para entorno de producción"""

    # Archivos de modelos
    DIRECTOR_MODEL_FILE = 'k2_director_model.h5'
    RF_MODEL_FILE = 'k2_randomforest_model.pkl'
    TF_MODEL_FILE = 'k2_tensorflow_model.h5'
    SCALER_FILE = 'k2_scaler.pkl'
    FEATURE_SELECTOR_FILE = 'k2_feature_selector.pkl'

    # Configuración de inferencia
    BATCH_INFERENCE_SIZE = 1000
    MAX_MEMORY_USAGE = 0.8  # 80% de memoria máxima

    # Configuración de monitoreo
    MONITOR_PERFORMANCE = True
    ALERT_THRESHOLD_PRECISION = 0.60  # Alerta si precisión < 60%
    ALERT_THRESHOLD_RECALL = 0.80     # Alerta si recall < 80%

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_config_summary():
    """Retorna resumen de configuración"""
    return {
        'random_seed': K2Config.RANDOM_SEED,
        'train_size': K2Config.TRAIN_SIZE,
        'rf_estimators': RFConfig.N_ESTIMATORS,
        'tf_layers': TFConfig.HIDDEN_LAYERS,
        'target_precision': EvalConfig.TARGET_PRECISION,
        'target_recall': EvalConfig.TARGET_RECALL
    }

def validate_config():
    """Valida configuración del sistema"""
    assert K2Config.TRAIN_SIZE + K2Config.VALIDATION_SIZE + K2Config.TEST_SIZE == 1.0
    assert 0 < K2Config.CONFIDENCE_THRESHOLD < 1
    assert EvalConfig.TARGET_PRECISION > 0.5
    assert EvalConfig.TARGET_RECALL > 0.5
    return True

if __name__ == "__main__":
    print("🔧 K2 Configuration Loaded")
    print(f"Random Seed: {K2Config.RANDOM_SEED}")
    print(f"Target Precision: {EvalConfig.TARGET_PRECISION}")
    print(f"Target Recall: {EvalConfig.TARGET_RECALL}")
    validate_config()
    print("✅ Configuration validated successfully")