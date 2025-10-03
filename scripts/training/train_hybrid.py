#!/usr/bin/env python3
"""
SCRIPT DE ENTRENAMIENTO - MODELO HÍBRIDO
=========================================
Script simplificado para entrenar el modelo híbrido
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trainers import HybridModelTrainer

def main():
    trainer = HybridModelTrainer()
    return trainer.run_training_pipeline('saved_models/hybrid_tf_model.keras')

if __name__ == "__main__":
    model, accuracy = main()