"""
🎯 PREDICTOR CON DIRECTOR GENERAL
Script principal para usar el Director General con los 9 submodelos
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.general_director import GeneralDirector

def demo_predictions():
    """
    Demostración del Director General
    """
    print("🎯 DIRECTOR GENERAL MULTI-MISIÓN")
    print("="*50)
    print("🔭 KOI: 3 submodelos (RF + TF + Director)")
    print("🛰️ TOI: 3 submodelos (RF + TF + Director)")
    print("🌍 K2: 3 submodelos (RF + TF + Director)")
    print("📊 Total: 9 submodelos coordinados")
    print("="*50)

    try:
        # Inicializar Director General
        print("\n🚀 Inicializando Director General...")
        director = GeneralDirector()
        print("   ✅ Director General cargado")

        # Verificar sistemas disponibles
        print("\n📊 Verificando sistemas...")
        koi_available = hasattr(director, 'koi_director') and director.koi_director is not None
        toi_available = hasattr(director, 'toi_director') and director.toi_director is not None
        k2_available = hasattr(director, 'k2_director') and director.k2_director is not None

        print(f"   🔭 KOI System: {'✅ Disponible' if koi_available else '❌ No disponible'}")
        print(f"   🛰️ TOI System: {'✅ Disponible' if toi_available else '❌ No disponible'}")
        print(f"   🌍 K2 System: {'✅ Disponible' if k2_available else '❌ No disponible'}")

        # Verificar datasets
        print("\n📂 Verificando datasets...")
        data_dir = Path("data/clean")
        koi_data = data_dir / "koi_clean.csv"
        toi_data = data_dir / "toi_full.csv"  # Nombre correcto del archivo TOI
        k2_data = data_dir / "k2_clean.csv"

        print(f"   📊 KOI Dataset: {'✅ Disponible' if koi_data.exists() else '❌ No encontrado'}")
        print(f"   📊 TOI Dataset: {'✅ Disponible' if toi_data.exists() else '❌ No encontrado'}")
        print(f"   📊 K2 Dataset: {'✅ Disponible' if k2_data.exists() else '❌ No encontrado'}")

        # Test con datos reales si están disponibles
        if koi_available and koi_data.exists():
            print("\n🔭 PROBANDO SISTEMA KOI...")
            df_koi = pd.read_csv(koi_data)

            # Tomar una muestra pequeña
            sample_size = min(10, len(df_koi))
            sample_data = df_koi.sample(n=sample_size)

            print(f"   📊 Muestra de {sample_size} candidatos")

            # Hacer predicciones
            try:
                predictions = director.predict(sample_data, mission='KOI')
                print(f"   ✅ Predicciones completadas")
                print(f"   🎯 Planetas detectados: {sum(predictions)}/{len(predictions)}")

                # Mostrar algunas predicciones individuales
                for i, pred in enumerate(predictions[:3]):
                    planeta_str = "🪐 PLANETA" if pred == 1 else "⭕ NO PLANETA"
                    print(f"   📋 Candidato {i+1}: {planeta_str}")

            except Exception as e:
                print(f"   ❌ Error en predicciones KOI: {e}")

        # Test con TOI si está disponible
        if toi_available and toi_data.exists():
            print("\n🛰️ PROBANDO SISTEMA TOI...")
            try:
                df_toi = pd.read_csv(toi_data)
                sample_size = min(5, len(df_toi))
                sample_data = df_toi.sample(n=sample_size)

                predictions = director.predict(sample_data, mission='TOI')
                print(f"   ✅ TOI: {sum(predictions)}/{len(predictions)} planetas detectados")
            except Exception as e:
                print(f"   ❌ Error en predicciones TOI: {e}")

        # Test con K2 si está disponible
        if k2_available and k2_data.exists():
            print("\n🌍 PROBANDO SISTEMA K2...")
            try:
                df_k2 = pd.read_csv(k2_data)
                sample_size = min(5, len(df_k2))
                sample_data = df_k2.sample(n=sample_size)

                predictions = director.predict(sample_data, mission='K2')
                print(f"   ✅ K2: {sum(predictions)}/{len(predictions)} planetas detectados")
            except Exception as e:
                print(f"   ❌ Error en predicciones K2: {e}")

        print("\n🎯 RESUMEN DEL DIRECTOR GENERAL:")
        available_systems = sum([koi_available, toi_available, k2_available])
        print(f"   📊 Sistemas operativos: {available_systems}/3")
        print(f"   🧠 Submodelos activos: {available_systems * 3}/9")

        if available_systems == 3:
            print(f"   🏆 ¡DIRECTOR GENERAL COMPLETAMENTE OPERATIVO!")
        elif available_systems > 0:
            print(f"   ⚠️  Director parcialmente operativo")
        else:
            print(f"   ❌ Director no operativo - entrenar sistemas primero")

    except Exception as e:
        print(f"\n❌ ERROR GENERAL: {e}")
        print("💡 Sugerencias:")
        print("   1. Ejecuta 'python train_all_systems.py' primero")
        print("   2. Verifica que existan los datasets en data/clean/")
        print("   3. Revisa los logs de entrenamiento")

def interactive_prediction():
    """
    Modo interactivo para predicciones
    """
    print("\n🎮 MODO INTERACTIVO")
    print("="*30)

    try:
        director = GeneralDirector()

        while True:
            print("\n🎯 Opciones:")
            print("1. 🔭 Predecir KOI")
            print("2. 🛰️ Predecir TOI")
            print("3. 🌍 Predecir K2")
            print("4. 📊 Ver estadísticas")
            print("5. ❌ Salir")

            choice = input("\nSelecciona opción (1-5): ").strip()

            if choice == '1':
                print("🔭 Modo KOI seleccionado")
                # Aquí iría la lógica específica para KOI

            elif choice == '2':
                print("🛰️ Modo TOI seleccionado")
                # Aquí iría la lógica específica para TOI

            elif choice == '3':
                print("🌍 Modo K2 seleccionado")
                # Aquí iría la lógica específica para K2

            elif choice == '4':
                print("📊 Estadísticas del Director General")
                # Mostrar estadísticas generales

            elif choice == '5':
                print("👋 ¡Hasta luego!")
                break

            else:
                print("❌ Opción no válida")

    except Exception as e:
        print(f"❌ Error en modo interactivo: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Director General - Predictor Multi-Misión")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Modo interactivo")
    parser.add_argument("--demo", "-d", action="store_true",
                       help="Ejecutar demostración")

    args = parser.parse_args()

    if args.interactive:
        interactive_prediction()
    elif args.demo:
        demo_predictions()
    else:
        # Por defecto, ejecutar demo
        demo_predictions()