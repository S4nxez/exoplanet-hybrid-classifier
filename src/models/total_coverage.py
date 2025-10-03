#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO DE COBERTURA TOTAL
=========================
Modelo que clasifica todos los casos
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib

class TotalCoverageModel:
    """Modelo de cobertura total con feature importance y calibración"""

    def __init__(self):
        # Modelo base RandomForest
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        # Calibración con validación cruzada
        self.model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=3
        )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importances_ = None
        self.feature_names_ = None
        self.is_trained = False

    def train(self, X, y, feature_names=None):
        """Entrenar modelo con todos los datos"""
        # División entrenamiento/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalado
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Codificar etiquetas
        self.label_encoder.fit(y_train)

        # Entrenar modelo calibrado
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True  # Marcar como entrenado ANTES de evaluaciones

        # Extraer feature importance del modelo base
        try:
            if hasattr(self.model, 'calibrated_classifiers_') and len(self.model.calibrated_classifiers_) > 0:
                # Para CalibratedClassifierCV
                importances = []
                for clf in self.model.calibrated_classifiers_:
                    if hasattr(clf.estimator, 'feature_importances_'):
                        importances.append(clf.estimator.feature_importances_)
                    elif hasattr(clf.base_estimator, 'feature_importances_'):
                        importances.append(clf.base_estimator.feature_importances_)
                if importances:
                    self.feature_importances_ = np.mean(importances, axis=0)
            elif hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
        except Exception as e:
            print(f"   ⚠️ No se pudo extraer feature importance: {e}")
            self.feature_importances_ = None
        
        self.feature_names_ = feature_names

        # Evaluar con métricas extendidas
        predictions = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, predictions)
        }
    
    def get_feature_importance(self, top_n=None):
        """Obtener importancia de características"""
        if not self.is_trained or self.feature_importances_ is None:
            return None
            
        importance_dict = {}
        for i, importance in enumerate(self.feature_importances_):
            feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
            importance_dict[feature_name] = importance
        
        # Ordenar por importancia
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        if top_n:
            sorted_importance = sorted_importance[:top_n]
            
        return sorted_importance
    
    def get_feature_weights(self):
        """Obtener pesos de características para el modelo híbrido"""
        if not self.is_trained or self.feature_importances_ is None:
            return None
        return self.feature_importances_

    def predict(self, X):
        """Predecir casos"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predecir probabilidades"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, model_path, scaler_path, encoder_path):
        """Guardar modelo, escalador y codificador"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Guardar también feature importance y nombres
        import os
        base_path = os.path.dirname(model_path)
        if self.feature_importances_ is not None:
            joblib.dump(self.feature_importances_, os.path.join(base_path, 'total_feature_importance.pkl'))
        if self.feature_names_ is not None:
            joblib.dump(self.feature_names_, os.path.join(base_path, 'total_feature_names.pkl'))

    def load(self, model_path, scaler_path, encoder_path):
        """Cargar modelo, escalador y codificador"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_trained = True
        
        # Cargar también feature importance y nombres
        import os
        base_path = os.path.dirname(model_path)
        try:
            importance_path = os.path.join(base_path, 'total_feature_importance.pkl')
            if os.path.exists(importance_path):
                self.feature_importances_ = joblib.load(importance_path)
        except:
            self.feature_importances_ = None
            
        try:
            names_path = os.path.join(base_path, 'total_feature_names.pkl')
            if os.path.exists(names_path):
                self.feature_names_ = joblib.load(names_path)
        except:
            self.feature_names_ = None