#!/usr/bin/env python3
"""
SCRIPT DE ENTRENAMIENTO - MODELO TOTAL
=======================================
Script simplificado para entrenar el modelo total
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trainers import TotalModelTrainer

def main():
    trainer = TotalModelTrainer()
    return trainer.run_training_pipeline('saved_models/total_model.pkl')

if __name__ == "__main__":
    model, accuracy = main()