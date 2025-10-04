#!/usr/bin/env python3
"""
TEST SIMPLIFICADO - Verificar que Orkhestra funcione básicamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_processor import DataProcessor
from src.models.partial_coverage import PartialCoverageModel  
from src.models.total_coverage import TotalCoverageModel
import numpy as np
from sklearn.metrics import accuracy_score

def test_orkhestra_simple():
    """Test simplificado de fusión"""
    print("🧪 TEST SIMPLIFICADO DE ORKHESTRA")
    print("="*50)
    
    # 1. Cargar datos
    data_processor = DataProcessor()
    data = data_processor.load_clean_data('data/dataset.csv')
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_split(data)
    
    print(f"📊 Datos: {len(X_test)} muestras de prueba")
    print(f"📊 Etiquetas únicas: {np.unique(y_test)}")
    
    # 2. Entrenar modelos individuales
    print("\n🎯 Entrenando modelo parcial...")
    partial_model = PartialCoverageModel(confidence_threshold=0.95)
    partial_model.train(X_train, y_train)
    
    print("\n🌐 Entrenando modelo total...")
    total_model = TotalCoverageModel()
    total_model.train(X_train, y_train)
    
    # 3. Evaluar modelos individuales
    print("\n📊 EVALUACIÓN INDIVIDUAL:")
    
    # Parcial con confianza
    partial_confident = partial_model.predict_with_confidence(X_test)
    partial_mask = partial_confident != 'Unknown'
    print(f"   🎯 Modelo Parcial coverage: {np.mean(partial_mask)*100:.1f}%")
    
    if np.any(partial_mask):
        partial_acc = accuracy_score(y_test[partial_mask], partial_confident[partial_mask])
        print(f"   🎯 Modelo Parcial accuracy (casos confiados): {partial_acc:.4f} ({partial_acc*100:.1f}%)")
    
    # Total siempre
    total_pred = total_model.predict(X_test)
    total_acc = accuracy_score(y_test, total_pred)
    print(f"   🌐 Modelo Total accuracy: {total_acc:.4f} ({total_acc*100:.1f}%)")
    
    # 4. Fusión manual simple
    print("\n🎼 FUSIÓN MANUAL ORKHESTRA:")
    
    # Usar modelo total como base
    final_predictions = total_pred.copy()
    fusion_info = {
        'used_partial': np.zeros(len(X_test), dtype=bool),
        'used_total': np.ones(len(X_test), dtype=bool)
    }
    
    # Reemplazar con modelo parcial donde sea confiado
    if np.any(partial_mask):
        final_predictions[partial_mask] = partial_confident[partial_mask]
        fusion_info['used_partial'][partial_mask] = True
        fusion_info['used_total'][partial_mask] = False
    
    # Evaluar fusión
    fusion_acc = accuracy_score(y_test, final_predictions)
    
    print(f"   🎼 Orkhestra accuracy: {fusion_acc:.4f} ({fusion_acc*100:.1f}%)")
    print(f"   📊 Uso parcial: {np.mean(fusion_info['used_partial'])*100:.1f}%")
    print(f"   📊 Uso total: {np.mean(fusion_info['used_total'])*100:.1f}%")
    
    # Verificación por partes
    if np.any(fusion_info['used_partial']):
        partial_section_acc = accuracy_score(
            y_test[fusion_info['used_partial']], 
            final_predictions[fusion_info['used_partial']]
        )
        print(f"   🎯 Accuracy sección parcial: {partial_section_acc:.4f} ({partial_section_acc*100:.1f}%)")
    
    if np.any(fusion_info['used_total']):
        total_section_acc = accuracy_score(
            y_test[fusion_info['used_total']], 
            final_predictions[fusion_info['used_total']]
        )
        print(f"   🌐 Accuracy sección total: {total_section_acc:.4f} ({total_section_acc*100:.1f}%)")
    
    print("\n✅ Test completado")
    return fusion_acc

if __name__ == "__main__":
    test_orkhestra_simple()