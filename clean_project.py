"""
🧹 SCRIPT DE LIMPIEZA DEL PROYECTO DIRECTOR GENERAL
Mantiene solo archivos esenciales para el funcionamiento del Director General
que maneja 9 submodelos (3 por cada dataset: KOI, TOI, K2)
"""

import os
import shutil
from pathlib import Path

def clean_project():
    """
    Limpia el proyecto manteniendo solo los archivos esenciales
    """
    project_root = Path(".")

    # Archivos ESENCIALES que deben mantenerse
    essential_files = {
        # 🎯 DIRECTOR GENERAL (core)
        "src/models/general_director.py",

        # 📊 DATASETS LIMPIOS (obligatorios)
        "data/koi.csv",
        "data/TOI.csv",
        "data/k2.csv",
        "data/clean/koi_clean.csv",
        "data/clean/toi_clean.csv",
        "data/clean/k2_clean.csv",

        # 🔭 SISTEMA KOI (3 submodelos)
        "koi_system/__init__.py",
        "koi_system/core/__init__.py",
        "koi_system/core/director.py",
        "koi_system/models/__init__.py",
        "koi_system/models/koi_randomforest.py",
        "koi_system/models/koi_tensorflow.py",
        "koi_system/utils/__init__.py",
        "koi_system/utils/data_utils.py",
        "koi_system/config/__init__.py",
        "koi_system/config/koi_config.py",
        "koi_system/train_koi_system.py",
        "koi_system/predict_koi.py",

        # 🛰️ SISTEMA TOI (3 submodelos)
        "toi_system/__init__.py",
        "toi_system/core/__init__.py",
        "toi_system/core/director.py",
        "toi_system/models/__init__.py",
        "toi_system/models/toi_randomforest.py",
        "toi_system/models/toi_tensorflow.py",
        "toi_system/utils/__init__.py",
        "toi_system/utils/data_utils.py",
        "toi_system/config/__init__.py",
        "toi_system/config/toi_config.py",
        "toi_system/train_complete_toi.py",
        "toi_system/predict_toi.py",

        # 🌍 SISTEMA K2 (3 submodelos)
        "k2_system/__init__.py",
        "k2_system/models/__init__.py",
        "k2_system/models/k2_director.py",
        "k2_system/models/k2_randomforest.py",
        "k2_system/models/k2_tensorflow.py",
        "k2_system/utils/__init__.py",
        "k2_system/utils/data_utils.py",
        "k2_system/config/__init__.py",
        "k2_system/config/k2_config.py",
        "k2_system/train_complete_k2.py",
        "k2_system/predict_k2.py",

        # 💾 MODELOS ENTRENADOS (esenciales)
        "koi_system/saved_models/rf_model.pkl",
        "koi_system/saved_models/tf_model.pkl",
        "koi_system/saved_models/metadata.pkl",
        "koi_system/saved_models/scaler.pkl",
        "toi_system/saved_models/rf_model.pkl",
        "toi_system/saved_models/tf_model.pkl",
        "toi_system/saved_models/metadata.pkl",
        "toi_system/saved_models/scaler.pkl",
        "k2_system/saved_models/rf_model.pkl",
        "k2_system/saved_models/tf_model.pkl",
        "k2_system/saved_models/metadata.pkl",
        "k2_system/saved_models/scaler.pkl",

        # 🐍 ARCHIVOS DE CONFIGURACIÓN
        "requirements.txt",
        "README.md",
        "Makefile",

        # 🎯 SCRIPT PRINCIPAL PARA USAR EL DIRECTOR
        "predict_with_director.py",
        "train_all_systems.py",

        # 📈 GRÁFICAS FINALES
        "graficas/accuracy_comparison.png",
        "graficas/complete_dashboard.png",
        "graficas/confusion_matrix.png",
        "graficas/model_usage.png",
        "graficas/performance_radar.png",
        "graficas/training_evolution.png",
    }

    # Directorios ESENCIALES que deben mantenerse completamente
    essential_dirs = {
        "src",
        "data",
        "graficas",
        "koi_system/saved_models",
        "toi_system/saved_models",
        "k2_system/saved_models",
        "koi_system/core",
        "koi_system/models",
        "koi_system/utils",
        "koi_system/config",
        "toi_system/core",
        "toi_system/models",
        "toi_system/utils",
        "toi_system/config",
        "k2_system/models",
        "k2_system/utils",
        "k2_system/config",
    }

    # Archivos a ELIMINAR (redundantes/experimentales)
    files_to_remove = []

    print("🧹 ANALIZANDO PROYECTO PARA LIMPIEZA...")
    print("="*50)

    # Buscar archivos no esenciales
    for root, dirs, files in os.walk("."):
        # Saltar directorios ocultos
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.startswith('.'):
                continue

            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_root)
            relative_str = str(relative_path).replace("\\", "/")

            # Si no está en la lista de esenciales, marcarlo para eliminación
            if relative_str not in essential_files:
                # Verificar si está en un directorio esencial
                in_essential_dir = any(relative_str.startswith(essential_dir)
                                     for essential_dir in essential_dirs)

                # Si no está en directorio esencial y no es esencial, eliminar
                if not in_essential_dir:
                    files_to_remove.append(relative_path)

    print(f"📊 ARCHIVOS ENCONTRADOS PARA ELIMINAR: {len(files_to_remove)}")

    # Mostrar archivos que se van a eliminar
    if files_to_remove:
        print("\n🗑️ ARCHIVOS QUE SE ELIMINARÁN:")
        for file_path in sorted(files_to_remove):
            print(f"   ❌ {file_path}")

    # Confirmar eliminación
    print(f"\n⚠️  SE ELIMINARÁN {len(files_to_remove)} ARCHIVOS")
    response = input("¿Continuar? (s/N): ").strip().lower()

    if response in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n🗑️ ELIMINANDO ARCHIVOS NO ESENCIALES...")

        removed_count = 0
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
                print(f"   ✅ Eliminado: {file_path}")
            except Exception as e:
                print(f"   ❌ Error eliminando {file_path}: {e}")

        print(f"\n✅ LIMPIEZA COMPLETADA: {removed_count} archivos eliminados")

        # Eliminar directorios vacíos
        print("\n🧹 ELIMINANDO DIRECTORIOS VACÍOS...")
        empty_dirs_removed = 0

        for root, dirs, files in os.walk(".", topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):  # Si está vacío
                        dir_path.rmdir()
                        empty_dirs_removed += 1
                        print(f"   ✅ Directorio vacío eliminado: {dir_path}")
                except:
                    pass  # Ignorar errores

        print(f"\n🎯 RESUMEN FINAL:")
        print(f"   📁 Archivos eliminados: {removed_count}")
        print(f"   📂 Directorios vacíos eliminados: {empty_dirs_removed}")
        print(f"   🎯 Proyecto limpio y listo para usar")

    else:
        print("\n❌ LIMPIEZA CANCELADA")

    print("\n📋 ARCHIVOS ESENCIALES MANTENIDOS:")
    print("   🎯 Director General: src/models/general_director.py")
    print("   🔭 Sistema KOI: 3 submodelos (RF, TF, Director)")
    print("   🛰️ Sistema TOI: 3 submodelos (RF, TF, Director)")
    print("   🌍 Sistema K2: 3 submodelos (RF, TF, Director)")
    print("   📊 Total: 9 submodelos + 1 Director General")


if __name__ == "__main__":
    clean_project()
