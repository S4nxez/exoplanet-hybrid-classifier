#!/usr/bin/env python3
"""
SCRIPT DE LIMPIEZA MEJORADO
===========================
Limpia modelos guardados y archivos temporales de forma organizada
"""

import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class ProjectCleaner:
    """Limpiador organizado del proyecto"""

    def __init__(self):
        self.directories_to_clean = [
            '__pycache__',
            'src/__pycache__',
            'src/models/__pycache__',
            'src/utils/__pycache__',
            'src/trainers/__pycache__',
            'src/evaluators/__pycache__',
            'scripts/__pycache__',
            'scripts/training/__pycache__',
            'scripts/prediction/__pycache__',
            'scripts/utils/__pycache__'
        ]

        self.saved_models_dir = 'saved_models'

    def clean_cache(self):
        """Eliminar todos los archivos de caché de Python"""
        print("🧹 Limpiando caché de Python...")
        cleaned_count = 0

        for cache_dir in self.directories_to_clean:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"   ✅ {cache_dir} eliminado")
                cleaned_count += 1

        if cleaned_count == 0:
            print("   ℹ️  No hay archivos de caché para eliminar")
        else:
            print(f"   🎉 {cleaned_count} directorios de caché eliminados")

    def clean_saved_models(self):
        """Eliminar modelos guardados"""
        if os.path.exists(self.saved_models_dir):
            shutil.rmtree(self.saved_models_dir)
            print("✅ Modelos guardados eliminados")
        else:
            print("ℹ️  No hay modelos guardados para eliminar")

    def clean_all(self):
        """Limpiar todo"""
        self.clean_cache()
        self.clean_saved_models()

def main():
    print("🧹 LIMPIADOR DE PROYECTO")
    print("="*30)

    cleaner = ProjectCleaner()

    choice = input("¿Qué deseas limpiar?\n"
                  "1. Solo caché de Python\n"
                  "2. Solo modelos guardados\n"
                  "3. Todo (caché + modelos)\n"
                  "Opción (1-3): ")

    if choice == "1":
        cleaner.clean_cache()
    elif choice == "2":
        cleaner.clean_saved_models()
    elif choice == "3":
        cleaner.clean_all()
    else:
        print("❌ Opción no válida")
        return

    print("\n🎉 Limpieza completada!")

if __name__ == "__main__":
    main()