#!/usr/bin/env python3
"""
🔮 K2 PREDICTION SCRIPT
======================
Script para hacer predicciones con el sistema K2 entrenado.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from models.k2_ensemble import K2EnsembleSystem
from utils.data_utils import K2DataLoader, load_k2_sample_data
from utils.visualization_utils import K2Visualizer
from config.k2_config import LogConfig

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class K2Predictor:
    """Predictor para el sistema K2"""

    def __init__(self, model_path="saved_models"):
        self.model_path = Path(model_path)
        self.system = K2EnsembleSystem()
        self.is_loaded = False

    def load_system(self):
        """Carga el sistema entrenado"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {self.model_path}")

        logger.info(f"📁 Cargando sistema desde: {self.model_path}")
        self.system.load_system(self.model_path)
        self.is_loaded = True
        logger.info("✅ Sistema cargado exitosamente")

    def predict_single(self, features):
        """Predicción para una sola muestra"""
        if not self.is_loaded:
            self.load_system()

        # Convertir a formato correcto
        if isinstance(features, dict):
            # Si es diccionario, convertir a DataFrame
            features_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            # Si es Series, convertir a DataFrame con una fila
            features_df = pd.DataFrame([features])
        elif isinstance(features, (list, np.ndarray)):
            # Si es array, asumir orden correcto
            features_df = pd.DataFrame([features])
        else:
            features_df = features

        # Hacer predicción detallada
        results = self.system.predict_with_details(features_df)

        prediction = results['predictions'][0]
        probability = results['probabilities'][0]
        model_used = results['model_used'][0]
        confidence = results['prediction_confidences'][0]

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'model_used': model_used,
            'confidence': float(confidence),
            'is_exoplanet': bool(prediction),
            'confidence_level': self._get_confidence_level(confidence)
        }

    def predict_batch(self, features_df):
        """Predicción para múltiples muestras"""
        if not self.is_loaded:
            self.load_system()

        logger.info(f"🔮 Procesando {len(features_df)} muestras...")

        # Hacer predicciones detalladas
        results = self.system.predict_with_details(features_df)

        # Formatear resultados
        predictions = []
        for i in range(len(features_df)):
            pred_result = {
                'sample_id': i,
                'prediction': int(results['predictions'][i]),
                'probability': float(results['probabilities'][i]),
                'model_used': results['model_used'][i],
                'confidence': float(results['prediction_confidences'][i]),
                'is_exoplanet': bool(results['predictions'][i]),
                'confidence_level': self._get_confidence_level(results['prediction_confidences'][i])
            }
            predictions.append(pred_result)

        # Estadísticas del batch
        batch_stats = {
            'total_samples': len(features_df),
            'exoplanets_detected': int(np.sum(results['predictions'])),
            'detection_rate': float(np.mean(results['predictions'])),
            'rf_cases': int(results['rf_cases']),
            'tf_cases': int(results['tf_cases']),
            'average_confidence': float(np.mean(results['prediction_confidences']))
        }

        logger.info(f"✅ Predicciones completadas:")
        logger.info(f"   Exoplanetas detectados: {batch_stats['exoplanets_detected']}")
        logger.info(f"   Tasa de detección: {batch_stats['detection_rate']:.1%}")
        logger.info(f"   Confianza promedio: {batch_stats['average_confidence']:.3f}")

        return predictions, batch_stats

    def _get_confidence_level(self, confidence):
        """Convierte confianza numérica a nivel cualitativo"""
        if confidence >= 0.9:
            return "Muy Alta"
        elif confidence >= 0.8:
            return "Alta"
        elif confidence >= 0.7:
            return "Media"
        elif confidence >= 0.6:
            return "Baja"
        else:
            return "Muy Baja"

    def explain_prediction(self, features, sample_id=0):
        """Explica una predicción específica"""
        if not self.is_loaded:
            self.load_system()

        # Hacer predicción
        pred_result = self.predict_single(features)

        explanation = {
            'prediction_summary': pred_result,
            'interpretation': self._interpret_prediction(pred_result),
            'recommendations': self._get_recommendations(pred_result)
        }

        return explanation

    def _interpret_prediction(self, pred_result):
        """Interpreta el resultado de predicción"""
        interpretation = []

        if pred_result['is_exoplanet']:
            interpretation.append(f"🪐 EXOPLANETA DETECTADO con {pred_result['probability']:.1%} de probabilidad")
        else:
            interpretation.append(f"❌ No es un exoplaneta ({pred_result['probability']:.1%} probabilidad)")

        interpretation.append(f"🤖 Modelo usado: {pred_result['model_used']}")
        interpretation.append(f"🎯 Confianza: {pred_result['confidence_level']} ({pred_result['confidence']:.3f})")

        # Contexto por modelo
        if pred_result['model_used'] == 'RandomForest':
            interpretation.append("📝 Caso clasificado como 'seguro' - alta precisión esperada")
        elif pred_result['model_used'] == 'TensorFlow':
            interpretation.append("📝 Caso clasificado como 'complejo' - análisis profundo aplicado")

        return interpretation

    def _get_recommendations(self, pred_result):
        """Genera recomendaciones basadas en la predicción"""
        recommendations = []

        if pred_result['is_exoplanet']:
            if pred_result['confidence'] >= 0.8:
                recommendations.append("✅ Candidato fuerte - proceder con análisis detallado")
            elif pred_result['confidence'] >= 0.6:
                recommendations.append("⚠️ Candidato moderado - verificar con datos adicionales")
            else:
                recommendations.append("🔍 Candidato débil - requiere validación externa")
        else:
            if pred_result['confidence'] >= 0.8:
                recommendations.append("❌ Descartado con alta confianza")
            else:
                recommendations.append("🤔 Resultado incierto - considerar re-análisis")

        # Recomendaciones por modelo
        if pred_result['model_used'] == 'RandomForest':
            recommendations.append("📊 Recomendación: Analizar importancia de características")
        elif pred_result['model_used'] == 'TensorFlow':
            recommendations.append("🧠 Recomendación: Examinar patrones complejos detectados")

        return recommendations

def main():
    """Función principal de predicción"""
    print("🔮 K2 PREDICTION SYSTEM")
    print("="*40)

    try:
        # Inicializar predictor
        predictor = K2Predictor()
        predictor.load_system()

        # Generar datos de ejemplo para demostración
        logger.info("📊 Generando datos de ejemplo...")
        sample_data = load_k2_sample_data()

        # Procesar los datos de la misma manera que durante el entrenamiento
        data_loader = K2DataLoader()
        X_processed, y_processed = data_loader.get_processed_data(sample_data, target_column='koi_disposition')

        # Seleccionar algunas muestras para predicción (sin la variable objetivo)
        test_samples = X_processed.head(10)

        # 1. PREDICCIÓN INDIVIDUAL
        print("\n🔮 EJEMPLO: Predicción Individual")
        print("-" * 30)

        single_sample = test_samples.iloc[0]
        single_result = predictor.explain_prediction(single_sample)

        print("📋 Resultado de predicción:")
        for line in single_result['interpretation']:
            print(f"   {line}")

        print("\n💡 Recomendaciones:")
        for rec in single_result['recommendations']:
            print(f"   {rec}")

        # 2. PREDICCIÓN EN LOTE
        print("\n\n🔮 EJEMPLO: Predicción en Lote")
        print("-" * 30)

        predictions, stats = predictor.predict_batch(test_samples)

        print(f"📊 Estadísticas del lote:")
        print(f"   Muestras analizadas: {stats['total_samples']}")
        print(f"   Exoplanetas detectados: {stats['exoplanets_detected']}")
        print(f"   Tasa de detección: {stats['detection_rate']:.1%}")
        print(f"   Casos RandomForest: {stats['rf_cases']}")
        print(f"   Casos TensorFlow: {stats['tf_cases']}")
        print(f"   Confianza promedio: {stats['average_confidence']:.3f}")

        # Mostrar algunos resultados individuales
        print("\n📋 Resultados individuales (primeros 5):")
        for i, pred in enumerate(predictions[:5]):
            status = "🪐 EXOPLANETA" if pred['is_exoplanet'] else "❌ No exoplaneta"
            print(f"   Muestra {i+1}: {status} ({pred['confidence_level']}, {pred['model_used']})")

        # 3. ANÁLISIS DETALLADO
        print("\n\n🔍 ANÁLISIS DETALLADO")
        print("-" * 30)

        exoplanet_candidates = [p for p in predictions if p['is_exoplanet']]
        if exoplanet_candidates:
            print(f"🪐 {len(exoplanet_candidates)} candidatos a exoplaneta encontrados:")

            for i, candidate in enumerate(exoplanet_candidates, 1):
                print(f"   Candidato {i}:")
                print(f"     Probabilidad: {candidate['probability']:.1%}")
                print(f"     Confianza: {candidate['confidence_level']}")
                print(f"     Modelo: {candidate['model_used']}")
        else:
            print("❌ No se detectaron exoplanetas en esta muestra")

        print("\n✅ Análisis de predicción completado")

    except Exception as e:
        logger.error(f"❌ Error durante la predicción: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()