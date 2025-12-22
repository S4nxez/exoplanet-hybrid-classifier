#!/usr/bin/env python3
"""Utilities for loading, cleaning, and preparing K2 mission data."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from ..config.k2_config import K2Config, FeatureConfig, LogConfig

PHYSICAL_TARGET_FEATURES = [
    'pl_orbper',
    'pl_rade',
    'st_teff',
    'st_rad',
    'st_mass',
    'sy_dist',
]


def build_physical_target(df: pd.DataFrame) -> np.ndarray:
    """Derive a deterministic binary label based on simple physical bounds."""

    missing = [col for col in PHYSICAL_TARGET_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for synthetic K2 target: {missing}")

    work = df[PHYSICAL_TARGET_FEATURES].copy()
    work = work.fillna(work.median())

    mask = (
        work['pl_orbper'].between(0.3, 500)
        & work['pl_rade'].between(0.3, 25)
        & work['st_teff'].between(3000, 8000)
        & work['st_rad'].between(0.1, 10)
        & work['st_mass'].between(0.1, 5)
        & (work['sy_dist'] < 2000)
    )

    return mask.astype(int).to_numpy()

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class K2DataLoader:
    """Cargador y preprocessador de datos K2"""

    def __init__(self):
        self.feature_columns = None
        self.target_column = None
        self.imputer = SimpleImputer(strategy='median')
        self.data_stats = {}

    def load_data(self, data_path, target_column=None):
        """Carga datos desde archivo"""
        logger.info(f"Loading data from {data_path}")

        # Detectar formato de archivo
        data_path = Path(data_path)
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() in ['.pkl', '.pickle']:
            df = pd.read_pickle(data_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {data_path.suffix}")

        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Column count: {len(df.columns)}")

        # Guardar información básica
        self.target_column = target_column
        self.data_stats['original_shape'] = df.shape
        self.data_stats['columns'] = df.columns.tolist()

        return df

    def prepare_features(self, df):
        """Prepara características para el modelo (solo columnas base)."""
        logger.info("Preparing feature columns (base set)")

        available_features = []

        # Características base alineadas con los modelos persistidos y el director
        for feature in FeatureConfig.BASE_FEATURES:
            if feature in df.columns:
                available_features.append(feature)
            else:
                logger.warning(f"Missing feature: {feature}")

        # No se generan derivadas para evitar desalineación con los scalers entrenados
        self.feature_columns = available_features
        logger.info(f"Features available: {len(self.feature_columns)} -> {self.feature_columns}")

        return self.feature_columns

    def _create_derived_features(self, df):
        """Crea características derivadas"""
        derived = []

        try:
            # Transit depth ratio
            if 'pl_trandep' in df.columns and 'st_rad' in df.columns:
                df['transit_depth_ratio'] = df['pl_trandep'] / (df['st_rad'] ** 2)
                derived.append('transit_depth_ratio')

            # Orbital period log
            if 'pl_orbper' in df.columns:
                df['orbital_period_log'] = np.log10(df['pl_orbper'] + 1)
                derived.append('orbital_period_log')

            # Stellar brightness
            if 'sy_kepmag' in df.columns:
                df['stellar_brightness'] = 10 ** (-0.4 * df['sy_kepmag'])
                derived.append('stellar_brightness')

            # Planet-star radius ratio
            if 'pl_rade' in df.columns and 'st_rad' in df.columns:
                df['planet_star_radius_ratio'] = df['pl_rade'] / df['st_rad']
                derived.append('planet_star_radius_ratio')

            # Transit probability (simplified)
            if 'st_rad' in df.columns and 'pl_orbper' in df.columns:
                df['transit_probability'] = df['st_rad'] / df['pl_orbper']
                derived.append('transit_probability')

            # Stellar density proxy
            if 'st_mass' in df.columns and 'st_rad' in df.columns:
                df['stellar_density'] = df['st_mass'] / (df['st_rad'] ** 3)
                derived.append('stellar_density')

            logger.info(f"Derived features created: {len(derived)}")

        except Exception as e:
            logger.warning(f"Error creating derived features: {e}")

        return derived

    def clean_data(self, df):
        """Limpia y valida datos"""
        logger.info("Cleaning dataset")

        original_size = len(df)

        # Eliminar filas completamente vacías
        df = df.dropna(how='all')

        # Validar rangos de características
        for feature, (min_val, max_val) in FeatureConfig.FEATURE_RANGES.items():
            if feature in df.columns:
                # Filtrar valores fuera de rango
                mask = (df[feature] >= min_val) & (df[feature] <= max_val)
                df = df[mask]

        # Eliminar duplicados
        df = df.drop_duplicates()

        final_size = len(df)
        removed = original_size - final_size

        logger.info("Cleaning summary:")
        logger.info(f"   Original rows: {original_size}")
        logger.info(f"   Final rows: {final_size}")
        logger.info(f"   Removed rows: {removed} ({removed/original_size*100:.1f}%)")

        self.data_stats['cleaned_shape'] = df.shape
        self.data_stats['removed_rows'] = removed

        return df

    def prepare_target(self, df):
        """Return a binary target vector, deriving it when necessary."""
        if self.target_column and self.target_column in df.columns:
            logger.info(f"Preparing target column: {self.target_column}")
            target = df[self.target_column].copy()

            if target.dtype == 'object':
                positive_labels = {'CONFIRMED', 'CANDIDATE', 'confirmed', 'candidate', '1', 1}
                target = target.isin(positive_labels).astype(int)
            else:
                target = target.fillna(0).astype(int)
        else:
            logger.warning("Target column missing; deriving synthetic label from physical rules")
            generated = build_physical_target(df)
            target = pd.Series(generated, index=df.index, name='k2_physical_target')
            self.target_column = 'k2_physical_target'

        target_array = target.astype(int).to_numpy()
        target_counts = np.bincount(target_array, minlength=2)
        total = len(target)
        logger.info("Target distribution:")
        logger.info(f"   Class 0: {target_counts[0]} ({target_counts[0]/total*100:.1f}%)")
        logger.info(f"   Class 1: {target_counts[1]} ({target_counts[1]/total*100:.1f}%)")

        self.data_stats['target_distribution'] = target_counts.tolist()
        return target

    def split_data(self, X, y, test_size=None, val_size=None):
        """Divide datos en train/validation/test"""
        if test_size is None:
            test_size = K2Config.TEST_SIZE
        if val_size is None:
            val_size = K2Config.VALIDATION_SIZE

        # Ajustar tamaños para que sumen 1
        train_size = 1 - test_size - val_size

        logger.info(f"🔄 Dividiendo datos:")
        logger.info(f"   Entrenamiento: {train_size:.1%}")
        logger.info(f"   Validación: {val_size:.1%}")
        logger.info(f"   Prueba: {test_size:.1%}")

        # Primera división: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=K2Config.RANDOM_SEED,
            stratify=y
        )

        # Segunda división: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=K2Config.RANDOM_SEED,
            stratify=y_temp
        )

        logger.info(f"✅ División completada:")
        logger.info(f"   Train: {X_train.shape[0]} muestras")
        logger.info(f"   Val: {X_val.shape[0]} muestras")
        logger.info(f"   Test: {X_test.shape[0]} muestras")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_processed_data(self, data_input, target_column=None):
        """Pipeline completo de procesamiento"""
        logger.info("🚀 Iniciando pipeline de procesamiento de datos")

        # 1. Cargar datos
        if isinstance(data_input, (str, Path)):
            df = self.load_data(data_input, target_column)
        else:
            # Si es DataFrame, usar directamente
            df = data_input.copy()
            self.target_column = target_column

        # 2. Limpiar datos
        df = self.clean_data(df)

        # 3. Preparar características
        feature_columns = self.prepare_features(df)

        # 4. Preparar objetivo
        y = self.prepare_target(df)

        # 5. Extraer características
        X = df[feature_columns].copy()

        # 6. Imputar valores faltantes
        X_filled = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_filled, columns=feature_columns, index=X.index)

        # 7. Alinear X e y (por si se eliminaron filas)
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        logger.info(f"✅ Procesamiento completado:")
        logger.info(f"   Características: {X.shape[1]}")
        logger.info(f"   Muestras: {X.shape[0]}")
        logger.info(f"   Balanceo: {np.bincount(y)}")

        return X, y

    def get_data_summary(self):
        """Retorna resumen del procesamiento de datos"""
        return {
            'processing_stats': self.data_stats,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'available_features': len(self.feature_columns) if self.feature_columns else 0
        }

class K2DataValidator:
    """Validador de calidad de datos K2"""

    @staticmethod
    def validate_features(X, feature_names=None):
        """Valida calidad de características"""
        issues = []

        # Verificar valores faltantes
        missing_pct = (X.isnull().sum() / len(X) * 100)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            issues.append(f"Características con >20% valores faltantes: {high_missing.index.tolist()}")

        # Verificar características constantes
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        if constant_features:
            issues.append(f"Características constantes: {constant_features}")

        # Verificar outliers extremos
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        extreme_outliers = []
        for col in numeric_cols:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (X[col] < q1 - 3*iqr) | (X[col] > q3 + 3*iqr)
            outlier_pct = outlier_mask.sum() / len(X) * 100
            if outlier_pct > 5:
                extreme_outliers.append(f"{col}: {outlier_pct:.1f}%")
        if extreme_outliers:
            issues.append(f"Características con >5% outliers extremos: {extreme_outliers}")

        return issues

    @staticmethod
    def validate_target(y):
        """Valida variable objetivo"""
        issues = []

        # Verificar balanceo
        counts = np.bincount(y)
        minority_pct = min(counts) / sum(counts) * 100
        if minority_pct < 5:
            issues.append(f"Clase minoritaria muy pequeña: {minority_pct:.1f}%")

        # Verificar valores válidos
        unique_vals = np.unique(y)
        if not all(val in [0, 1] for val in unique_vals):
            issues.append(f"Variable objetivo debe ser binaria (0/1), encontrado: {unique_vals}")

        return issues

def load_k2_sample_data():
    """Carga datos de muestra para testing"""
    logger.info("📝 Generando datos de muestra K2...")

    np.random.seed(K2Config.RANDOM_SEED)
    n_samples = 1000

    # Generar características sintéticas
    data = {
        'pl_orbper': np.random.lognormal(1, 1, n_samples),
        'pl_rade': np.random.lognormal(0, 0.5, n_samples),
        'pl_trandep': np.random.lognormal(8, 1, n_samples),
        'st_teff': np.random.normal(5500, 1000, n_samples),
        'st_rad': np.random.lognormal(0, 0.3, n_samples),
        'st_mass': np.random.lognormal(0, 0.2, n_samples),
        'sy_kepmag': np.random.normal(12, 2, n_samples)
    }

    df = pd.DataFrame(data)

    # Generar target basado en características
    # Exoplanetas más probables con ciertos rangos
    exo_prob = (
        (df['pl_orbper'] < 50) * 0.3 +
        (df['pl_rade'] < 4) * 0.2 +
        (df['st_teff'] > 4000) * 0.2 +
        np.random.random(n_samples) * 0.3
    )

    df['k2_physical_target'] = (exo_prob > 0.6).astype(int)

    logger.info(f"✅ Datos de muestra generados: {df.shape}")
    logger.info(f"📊 Distribución: {np.bincount(df['k2_physical_target'])}")

    return df

if __name__ == "__main__":
    print("🛠︝ K2 Data Utilities")
    print("Utilidades para procesamiento de datos K2")

    # Ejemplo de uso
    sample_data = load_k2_sample_data()
    print(f"Datos de muestra: {sample_data.shape}")
    print(f"Características: {list(sample_data.columns)}")