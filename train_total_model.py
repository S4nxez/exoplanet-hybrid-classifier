#!/usr/bin/env python3
"""
WRAPPER - TRAIN TOTAL MODEL
===========================
Mantiene compatibilidad con el script original
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.training.train_total import main

if __name__ == "__main__":
    model, accuracy = main()