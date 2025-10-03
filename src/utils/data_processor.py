#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROCESADOR DE DATOS
===================
Preparación y limpieza de datos sin data leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class DataProcessor:
    """Procesador de datos para modelos de exoplanetas"""

    def __init__(self):
        self.features = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
            'koi_insol', 'koi_slogg', 'koi_srad', 'koi_steff'
        ]
        self.binary_mapping = {
            'FALSE POSITIVE': 'No exoplaneta',
            'CONFIRMED': 'Exoplaneta',
            'CANDIDATE': 'No exoplaneta'
        }

    def load_clean_data(self, filepath='data/dataset.csv'):
        """Cargar y limpiar datos sin data leakage"""
        df = pd.read_csv(filepath)
        valid_classes = ['FALSE POSITIVE', 'CONFIRMED', 'CANDIDATE']
        df_filtered = df[df['koi_disposition'].isin(valid_classes)].copy()

        # Seleccionar características disponibles
        available_features = [col for col in self.features if col in df_filtered.columns]
        clean_data = df_filtered[available_features + ['koi_disposition']].copy()

        # Limpiar valores faltantes
        for col in available_features:
            if clean_data[col].dtype in ['float64', 'int64']:
                median_val = clean_data[col].median()
                clean_data.loc[:, col] = clean_data[col].fillna(median_val)

        clean_data = clean_data.dropna()
        clean_data['binary_class'] = clean_data['koi_disposition'].map(self.binary_mapping)

        self.features = available_features
        return clean_data

    def identify_extreme_cases(self, data):
        """Identificar casos extremos para modelo parcial"""
        extreme_indices = []

        for idx, row in data.iterrows():
            is_extreme = False

            # Casos claramente exoplanetas (tránsito profundo + características planetarias)
            if all(col in self.features for col in ['koi_depth', 'koi_period', 'koi_duration', 'koi_prad']):
                depth, period, duration, radius = row['koi_depth'], row['koi_period'], row['koi_duration'], row['koi_prad']
                if (depth > 500 and 1 < period < 50 and duration > 1 and
                    1 < radius < 8 and row['binary_class'] == 'Exoplaneta'):
                    is_extreme = True

            # Casos claramente no exoplanetas (señal muy débil o parámetros imposibles)
            if all(col in self.features for col in ['koi_depth', 'koi_period']):
                depth, period = row['koi_depth'], row['koi_period']
                if ((depth < 20 or period < 0.5 or period > 1000) and
                    row['binary_class'] == 'No exoplaneta'):
                    is_extreme = True

            # Radio extremo
            if 'koi_prad' in self.features:
                radius = row['koi_prad']
                if ((radius > 20 or radius < 0.5) and row['binary_class'] == 'No exoplaneta'):
                    is_extreme = True

            # Temperatura extrema
            if 'koi_teq' in self.features:
                temp = row['koi_teq']
                if ((temp > 3000 or temp < 100) and row['binary_class'] == 'No exoplaneta'):
                    is_extreme = True

            if is_extreme:
                extreme_indices.append(idx)

        return extreme_indices

    def is_extreme_case(self, X_row):
        """Determinar si un caso individual es extremo"""
        row_data = pd.DataFrame([X_row], columns=self.features)

        if 'koi_depth' in self.features:
            depth = row_data['koi_depth'].iloc[0]
            if depth > 500 or depth < 20:
                return True

        if 'koi_prad' in self.features:
            radius = row_data['koi_prad'].iloc[0]
            if radius > 20 or radius < 0.5:
                return True

        if 'koi_teq' in self.features:
            temp = row_data['koi_teq'].iloc[0]
            if temp > 3000 or temp < 100:
                return True

        if all(col in self.features for col in ['koi_period', 'koi_duration']):
            period = row_data['koi_period'].iloc[0]
            duration = row_data['koi_duration'].iloc[0]
            if period < 0.5 or period > 1000 or duration < 0.1:
                return True

        return False

    def prepare_train_test_split(self, data, test_size=0.2, random_state=42):
        """Preparar división entrenamiento/test"""
        X = data[self.features].values
        y = data['binary_class'].values

        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def save_scaler(self, scaler, filepath):
        """Guardar escalador"""
        joblib.dump(scaler, filepath)

    def load_scaler(self, filepath):
        """Cargar escalador"""
        return joblib.load(filepath)

    def save_label_encoder(self, le, filepath):
        """Guardar codificador de etiquetas"""
        joblib.dump(le, filepath)

    def load_label_encoder(self, filepath):
        """Cargar codificador de etiquetas"""
        return joblib.load(filepath)