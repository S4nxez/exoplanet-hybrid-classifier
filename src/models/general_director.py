"""
Director General - Sistema Multi-Misión
Clasifica exoplanetas de KOI, TOI y K2 automáticamente
"""
import numpy as np
import pandas as pd
import logging
import os
import sys
from pathlib import Path

# Agregar rutas de los sistemas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "koi_system"))
sys.path.insert(0, str(PROJECT_ROOT / "toi_system"))
sys.path.insert(0, str(PROJECT_ROOT / "k2_system"))

from koi_system.core.director import KOIDirector
from toi_system.core.director import TOIDirector
from k2_system.models.k2_director import K2Director

logger = logging.getLogger(__name__)


class GeneralDirector:
    """
    Director General que coordina las predicciones entre las 3 misiones.

    Funcionalidad:
    1. Identifica automáticamente la misión del exoplaneta (KOI, TOI, K2)
    2. Delega la predicción al director especializado
    3. Mantiene estadísticas de uso por misión
    4. Usa los datasets limpios de data/clean/
    """

    def __init__(self, data_dir=None):
        """
        Inicializa el Director General

        Args:
            data_dir: Directorio con los datasets. Default: PROJECT_ROOT/data/clean/
        """
        self.koi_director = KOIDirector()
        self.toi_director = TOIDirector()
        self.k2_director = K2Director()

        # Directorio de datos
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / "data" / "clean"
        else:
            self.data_dir = Path(data_dir)

        # Cargar datasets para identificación de features
        self._load_dataset_schemas()

        # Auto-configurar directores si tienen modelos disponibles
        self._auto_configure_directors()

        # Estadísticas
        self.mission_stats = {
            'KOI': 0,
            'TOI': 0,
            'K2': 0,
            'Unknown': 0
        }

        logger.info("Director General inicializado")

    def _load_dataset_schemas(self):
        """Carga los esquemas de columnas de cada misión para identificación"""
        try:
            # KOI: tiene columnas como koi_score, koi_period, koi_fpflag_*
            koi_path = self.data_dir / "koi_clean.csv"
            if koi_path.exists():
                koi_df = pd.read_csv(koi_path, nrows=1)
                self.koi_columns = set(koi_df.columns)
                self.koi_key_features = [col for col in self.koi_columns if col.startswith('koi_')]
                logger.info(f"KOI schema cargado: {len(self.koi_columns)} columnas")
            else:
                logger.warning(f"Dataset KOI no encontrado: {koi_path}")
                self.koi_columns = set()
                self.koi_key_features = []

            # TOI: tiene columnas como toi, toipfx, tid
            toi_path = self.data_dir / "toi_full.csv"
            if toi_path.exists():
                toi_df = pd.read_csv(toi_path, nrows=1)
                self.toi_columns = set(toi_df.columns)
                self.toi_key_features = [col for col in self.toi_columns if 'toi' in col.lower() or col == 'tid']
                logger.info(f"TOI schema cargado: {len(self.toi_columns)} columnas")
            else:
                logger.warning(f"Dataset TOI no encontrado: {toi_path}")
                self.toi_columns = set()
                self.toi_key_features = []

            # K2: tiene columnas como pl_letter, sy_snum, sy_pnum
            k2_path = self.data_dir / "k2_clean.csv"
            if k2_path.exists():
                k2_df = pd.read_csv(k2_path, nrows=1)
                self.k2_columns = set(k2_df.columns)
                self.k2_key_features = [col for col in self.k2_columns if col.startswith('pl_') or col.startswith('sy_')]
                logger.info(f"K2 schema cargado: {len(self.k2_columns)} columnas")
            else:
                logger.warning(f"Dataset K2 no encontrado: {k2_path}")
                self.k2_columns = set()
                self.k2_key_features = []

        except Exception as e:
            logger.error(f"Error cargando schemas: {e}")
            self.koi_columns = set()
            self.toi_columns = set()
            self.k2_columns = set()
            self.koi_key_features = []
            self.toi_key_features = []
            self.k2_key_features = []

    def _auto_configure_directors(self):
        """Auto-configura los directores cargando modelos guardados si existen"""
        import joblib
        import tensorflow as tf

        self.directors_configured = {
            'KOI': False,
            'TOI': False,
            'K2': False
        }

        # Configurar KOI Director
        try:
            koi_rf_path = PROJECT_ROOT / "koi_system" / "saved_models" / "koi_randomforest_model.pkl"
            koi_rf_scaler_path = PROJECT_ROOT / "koi_system" / "saved_models" / "koi_randomforest_scaler.pkl"
            koi_tf_path = PROJECT_ROOT / "koi_system" / "saved_models" / "koi_tensorflow_model.h5"
            koi_tf_scaler_path = PROJECT_ROOT / "koi_system" / "saved_models" / "koi_tensorflow_scaler.pkl"

            if all(p.exists() for p in [koi_rf_path, koi_rf_scaler_path, koi_tf_path, koi_tf_scaler_path]):
                rf_model = joblib.load(koi_rf_path)
                rf_scaler = joblib.load(koi_rf_scaler_path)
                tf_model = tf.keras.models.load_model(koi_tf_path)
                tf_scaler = joblib.load(koi_tf_scaler_path)

                self.koi_director.rf_model = rf_model
                self.koi_director.rf_scaler = rf_scaler
                self.koi_director.tf_model = tf_model
                self.koi_director.tf_scaler = tf_scaler
                self.koi_director.is_trained = True
                self.directors_configured['KOI'] = True
                logger.info("✅ Director KOI auto-configurado")
        except Exception as e:
            logger.warning(f"No se pudo auto-configurar KOI: {e}")

        # Configurar TOI Director
        try:
            toi_rf_path = PROJECT_ROOT / "toi_system" / "saved_models" / "rf_model.pkl"
            toi_rf_scaler_path = PROJECT_ROOT / "toi_system" / "saved_models" / "rf_scaler.pkl"
            toi_tf_path = PROJECT_ROOT / "toi_system" / "saved_models" / "tf_model.h5"
            toi_tf_scaler_path = PROJECT_ROOT / "toi_system" / "saved_models" / "tf_scaler.pkl"

            if all(p.exists() for p in [toi_rf_path, toi_rf_scaler_path, toi_tf_path, toi_tf_scaler_path]):
                rf_model = joblib.load(toi_rf_path)
                rf_scaler = joblib.load(toi_rf_scaler_path)
                tf_model = tf.keras.models.load_model(toi_tf_path)
                tf_scaler = joblib.load(toi_tf_scaler_path)

                self.toi_director.rf_model = rf_model
                self.toi_director.rf_scaler = rf_scaler
                self.toi_director.tf_model = tf_model
                self.toi_director.tf_scaler = tf_scaler
                self.toi_director.is_trained = True
                self.directors_configured['TOI'] = True
                logger.info("✅ Director TOI auto-configurado")
        except Exception as e:
            logger.warning(f"No se pudo auto-configurar TOI: {e}")

        # Configurar K2 Director
        try:
            k2_rf_path = PROJECT_ROOT / "k2_system" / "saved_models" / "k2_rf_model.pkl"
            k2_rf_scaler_path = PROJECT_ROOT / "k2_system" / "saved_models" / "k2_rf_scaler.pkl"
            k2_tf_path = PROJECT_ROOT / "k2_system" / "saved_models" / "k2_tf_model.h5"
            k2_tf_scaler_path = PROJECT_ROOT / "k2_system" / "saved_models" / "k2_tf_scaler.pkl"

            if all(p.exists() for p in [k2_rf_path, k2_rf_scaler_path, k2_tf_path, k2_tf_scaler_path]):
                rf_model = joblib.load(k2_rf_path)
                rf_scaler = joblib.load(k2_rf_scaler_path)
                tf_model = tf.keras.models.load_model(k2_tf_path)
                tf_scaler = joblib.load(k2_tf_scaler_path)

                self.k2_director.rf_model = rf_model
                self.k2_director.rf_scaler = rf_scaler
                self.k2_director.tf_model = tf_model
                self.k2_director.tf_scaler = tf_scaler
                self.k2_director.is_trained = True
                self.directors_configured['K2'] = True
                logger.info("✅ Director K2 auto-configurado")
        except Exception as e:
            logger.warning(f"No se pudo auto-configurar K2: {e}")

        configured_count = sum(self.directors_configured.values())
        logger.info(f"Auto-configuración completa: {configured_count}/3 directores listos")

    def identify_mission(self, X_df):
        """
        Identifica de qué misión proviene un DataFrame

        Args:
            X_df: DataFrame con features del exoplaneta

        Returns:
            str: 'KOI', 'TOI', 'K2' o 'Unknown'
        """
        if not isinstance(X_df, pd.DataFrame):
            logger.warning("Input no es DataFrame, no se puede identificar misión")
            return 'Unknown'

        columns = set(X_df.columns)

        # Features únicos e identificatorios de cada misión
        # KOI: Kepler mission - columnas que empiezan con 'koi_'
        koi_unique = {'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'}

        # TOI: TESS mission - columnas específicas de TESS
        toi_unique = {'toi', 'toipfx', 'tid', 'ctoi_alias', 'tfopwg_disp'}

        # K2: Kepler K2 mission - columnas de planetas conocidos
        k2_unique = {'pl_letter', 'default_flag', 'sy_snum', 'cb_flag', 'disc_year', 'disc_locale'}

        # Contar features únicos presentes
        koi_score = len(columns.intersection(koi_unique))
        toi_score = len(columns.intersection(toi_unique))
        k2_score = len(columns.intersection(k2_unique))

        # Si hay features únicos claros, usar eso
        scores = {'KOI': koi_score, 'TOI': toi_score, 'K2': k2_score}
        max_score = max(scores.values())

        if max_score >= 2:  # Al menos 2 features únicos
            # Retornar la misión con más features únicos
            for mission, score in scores.items():
                if score == max_score:
                    logger.debug(f"Misión identificada por features únicos: {mission} (score={score})")
                    return mission

        # Fallback: usar el método anterior basado en cantidad de columnas
        koi_matches = len(columns.intersection(self.koi_columns))
        toi_matches = len(columns.intersection(self.toi_columns))
        k2_matches = len(columns.intersection(self.k2_columns))

        # Verificar features clave
        koi_key_present = sum(1 for feat in self.koi_key_features if feat in columns)
        toi_key_present = sum(1 for feat in self.toi_key_features if feat in columns)
        k2_key_present = sum(1 for feat in self.k2_key_features if feat in columns)

        # Decisión basada en coincidencias
        matches = {
            'KOI': (koi_matches, koi_key_present),
            'TOI': (toi_matches, toi_key_present),
            'K2': (k2_matches, k2_key_present)
        }

        # Ordenar por: 1) features clave, 2) total de columnas coincidentes
        best_match = max(matches.items(), key=lambda x: (x[1][1], x[1][0]))

        # Requiere al menos 50% de coincidencia o 3+ features clave
        if best_match[1][0] >= len(columns) * 0.5 or best_match[1][1] >= 3:
            logger.debug(f"Misión identificada por coincidencia: {best_match[0]}")
            return best_match[0]

        logger.warning(f"No se pudo identificar misión. Coincidencias: {matches}")
        return 'Unknown'

    def configure(self, mission, rf_model, rf_scaler, tf_model, tf_scaler):
        """
        Configura un director específico de misión

        Args:
            mission: str - 'KOI', 'TOI' o 'K2'
            rf_model: Modelo Random Forest entrenado
            rf_scaler: StandardScaler para RF
            tf_model: Modelo TensorFlow entrenado
            tf_scaler: StandardScaler para TF
        """
        mission = mission.upper()

        if mission == 'KOI':
            director = self.koi_director
        elif mission == 'TOI':
            director = self.toi_director
        elif mission == 'K2':
            director = self.k2_director
        else:
            raise ValueError(f"Misión desconocida: {mission}. Use 'KOI', 'TOI' o 'K2'")

        # Configurar el director
        director.rf_model = rf_model
        director.rf_scaler = rf_scaler
        director.tf_model = tf_model
        director.tf_scaler = tf_scaler
        director.is_trained = True

        self.directors_configured[mission] = True
        logger.info(f"Director {mission} configurado exitosamente")

    def predict(self, X, mission=None):
        """
        Predice si un exoplaneta candidato es genuino

        Args:
            X: Array o DataFrame con features del exoplaneta
            mission: str opcional - 'KOI', 'TOI' o 'K2'.
                    Si no se especifica, se intenta identificar automáticamente

        Returns:
            tuple: (predictions, mission_used)
                - predictions: array con predicciones (0=False Positive, 1=Confirmed)
                - mission_used: misión utilizada para la predicción
        """
        # Convertir a DataFrame si es necesario para identificación
        if mission is None:
            if isinstance(X, pd.DataFrame):
                mission = self.identify_mission(X)
            else:
                raise ValueError(
                    "Si X no es DataFrame, debe especificar el parámetro 'mission'"
                )
        else:
            mission = mission.upper()

        # Verificar que la misión sea válida
        if mission not in ['KOI', 'TOI', 'K2']:
            raise ValueError(f"Misión inválida: {mission}")

        # Verificar que el director esté configurado
        if not self.directors_configured[mission]:
            raise ValueError(
                f"Director {mission} no configurado. "
                f"Use configure('{mission}', rf_model, rf_scaler, tf_model, tf_scaler)"
            )

        # Convertir DataFrame a numpy array para los directores
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Delegar al director correspondiente
        if mission == 'KOI':
            predictions = self.koi_director.predict(X_array)
        elif mission == 'TOI':
            predictions = self.toi_director.predict(X_array)
        elif mission == 'K2':
            predictions = self.k2_director.predict(X_array)

        # Actualizar estadísticas
        self.mission_stats[mission] += len(predictions)

        logger.info(f"Predicción realizada con director {mission}: {len(predictions)} muestras")

        return predictions, mission

    def predict_proba(self, X, mission=None):
        """
        Predice probabilidades de que sea exoplaneta confirmado

        Args:
            X: Array o DataFrame con features
            mission: str opcional - 'KOI', 'TOI' o 'K2'

        Returns:
            tuple: (probabilities, mission_used)
        """
        # Identificar misión si no se especifica
        if mission is None:
            if isinstance(X, pd.DataFrame):
                mission = self.identify_mission(X)
            else:
                raise ValueError(
                    "Si X no es DataFrame, debe especificar el parámetro 'mission'"
                )
        else:
            mission = mission.upper()

        if not self.directors_configured[mission]:
            raise ValueError(f"Director {mission} no configurado")

        # Convertir a array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Obtener probabilidades del director
        if mission == 'KOI':
            director = self.koi_director
        elif mission == 'TOI':
            director = self.toi_director
        elif mission == 'K2':
            director = self.k2_director

        # Asegurar que X es 2D
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)

        # Calcular probabilidades usando Soft Voting
        X_rf = director.rf_scaler.transform(X_array)
        X_tf = director.tf_scaler.transform(X_array)

        rf_proba = director.rf_model.predict_proba(X_rf)[:, 1]
        tf_proba = director.tf_model.predict(X_tf, verbose=0).flatten()

        # Soft Voting: 75% RF + 25% TF
        ensemble_proba = 0.75 * rf_proba + 0.25 * tf_proba

        return ensemble_proba, mission

    def get_stats(self):
        """Retorna estadísticas de uso del Director General"""
        total = sum(self.mission_stats.values())

        stats = {
            'total_predictions': total,
            'predictions_by_mission': self.mission_stats.copy(),
            'directors_configured': self.directors_configured.copy(),
            'configured_count': sum(self.directors_configured.values())
        }

        if total > 0:
            stats['mission_percentages'] = {
                mission: (count / total) * 100
                for mission, count in self.mission_stats.items()
            }

        return stats

    def reset_stats(self):
        """Resetea las estadísticas de uso"""
        self.mission_stats = {mission: 0 for mission in self.mission_stats}

        # También resetear stats de cada director
        if self.directors_configured['KOI']:
            self.koi_director.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        if self.directors_configured['TOI']:
            self.toi_director.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}
        if self.directors_configured['K2']:
            self.k2_director.decision_stats = {'RandomForest': 0, 'TensorFlow': 0}

        logger.info("Estadísticas reseteadas")

    def __repr__(self):
        configured = sum(self.directors_configured.values())
        total_predictions = sum(self.mission_stats.values())
        return (
            f"GeneralDirector("
            f"configured={configured}/3, "
            f"total_predictions={total_predictions})"
        )
