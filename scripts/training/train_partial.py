#!/usr/bin/env python3
"""
SCRIPT DE ENTRENAMIENTO - MODELO PARCIAL
=========================================
Script simplificado para entrenar el modelo parcial
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trainers import PartialModelTrainer

def main():
    trainer = PartialModelTrainer()
    return trainer.run_training_pipeline('saved_models/partial_model.pkl')

if __name__ == "__main__":
    model, accuracy = main()