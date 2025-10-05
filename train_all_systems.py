"""
🎯 SCRIPT PRINCIPAL - ENTRENAR TODOS LOS SISTEMAS
Entrena los 9 submodelos del Director General (3 por dataset)
"""

import sys
import subprocess
from pathlib import Path

def train_all_systems():
    """
    Entrena todos los sistemas del Director General
    """
    print("🎯 INICIANDO ENTRENAMIENTO COMPLETO DEL DIRECTOR GENERAL")
    print("="*60)
    print("📊 Total de submodelos: 9 (3 por cada dataset)")
    print("🔭 KOI: RF + TF + Director")
    print("🛰️ TOI: RF + TF + Director")
    print("🌍 K2: RF + TF + Director")
    print("="*60)

    systems = [
        ("🔭 KOI System", "koi_system", "train_koi_system.py"),
        ("🛰️ TOI System", "toi_system", "train_complete_toi.py"),
        ("🌍 K2 System", "k2_system", "train_complete_k2.py")
    ]

    trained_systems = 0
    total_systems = len(systems)

    for system_name, system_dir, train_script in systems:
        print(f"\n📈 ENTRENANDO {system_name}...")
        print(f"   📁 Directorio: {system_dir}")
        print(f"   🐍 Script: {train_script}")

        try:
            # Cambiar al directorio del sistema
            original_dir = Path.cwd()
            system_path = Path(system_dir)

            if not system_path.exists():
                print(f"   ❌ ERROR: Directorio {system_dir} no existe")
                continue

            if not (system_path / train_script).exists():
                print(f"   ❌ ERROR: Script {train_script} no existe")
                continue

            # Ejecutar entrenamiento
            result = subprocess.run([
                sys.executable, train_script
            ], cwd=system_path, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ {system_name} entrenado exitosamente")
                trained_systems += 1
            else:
                print(f"   ❌ ERROR entrenando {system_name}")
                print(f"   📝 STDOUT: {result.stdout}")
                print(f"   📝 STDERR: {result.stderr}")

        except Exception as e:
            print(f"   ❌ EXCEPCIÓN entrenando {system_name}: {e}")

    print(f"\n🎯 RESUMEN FINAL:")
    print(f"   ✅ Sistemas entrenados: {trained_systems}/{total_systems}")
    print(f"   📊 Submodelos totales: {trained_systems * 3}")

    if trained_systems == total_systems:
        print(f"   🏆 ¡DIRECTOR GENERAL COMPLETAMENTE OPERATIVO!")
        print(f"   🚀 Usa 'python predict_with_director.py' para hacer predicciones")
    else:
        print(f"   ⚠️  Algunos sistemas fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    train_all_systems()