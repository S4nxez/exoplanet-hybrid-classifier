#!/usr/bin/env python3
"""
📊 KOI DATA UTILITIES
====================
Utilidades para cargar y preparar datos del dataset KOI.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.koi_config import KOIConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_path=None):
    """
    Carga y prepara datos del dataset KOI

    Returns:
        X_train, X_test, y_train, y_test
    """
    if data_path is None:
        data_path = KOIConfig.DATA_DIR / 'koi.csv'

    logger.info(f"📁 Cargando datos desde: {data_path}")

    # Cargar CSV
    df = pd.read_csv(data_path)
    logger.info(f"   Total registros: {len(df)}")

    # Filtrar solo CANDIDATE (positivos potenciales) y FALSE POSITIVE
    df_filtered = df[df['koi_pdisposition'].isin(['CANDIDATE', 'FALSE POSITIVE'])].copy()
    logger.info(f"   Registros filtrados: {len(df_filtered)}")

    # Crear target binario (1 = CANDIDATE/exoplaneta, 0 = FALSE POSITIVE)
    df_filtered['target'] = (df_filtered['koi_pdisposition'] == 'CANDIDATE').astype(int)

    # Seleccionar features
    features = KOIConfig.FEATURES

    # Eliminar filas con valores faltantes
    df_clean = df_filtered[features + ['target']].dropna()
    logger.info(f"   Registros limpios: {len(df_clean)}")

    # Separar X e y
    X = df_clean[features].values
    y = df_clean['target'].values

    # Información de distribución
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"   Distribución de clases:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        class_name = "CANDIDATE (exoplanet)" if label == 1 else "FALSE POSITIVE"
        logger.info(f"      {class_name} ({label}): {count}")

    if len(unique) > 1:
        logger.info(f"      Ratio: {counts[unique.tolist().index(1)]/counts[unique.tolist().index(0)]:.2f}")
    else:
        logger.error("⚠️  ¡Solo hay una clase en los datos! Verificar filtrado.")
        raise ValueError("Dataset tiene solo una clase")

    # Split train/test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=KOIConfig.TEST_SIZE,
        random_state=KOIConfig.RANDOM_SEED,
        stratify=y
    )

    logger.info(f"   Train set: {len(X_train)} muestras")
    logger.info(f"   Test set: {len(X_test)} muestras")
    logger.info(f"✅ Datos cargados y preparados exitosamente")

    return X_train, X_test, y_train, y_test

def get_feature_names():
    """Retorna nombres de features"""
    return KOIConfig.FEATURES
