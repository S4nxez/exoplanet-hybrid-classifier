import sys
import os

# Agregar la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_processor import DataProcessor
from models.partial_coverage import PartialCoverageModel
from models.total_coverage import TotalCoverageModel

def main():
    print("Entrenamiento de Modelos de Exoplanetas")
    print("=" * 50)

    # Inicializar procesador de datos
    data_processor = DataProcessor()

    # Cargar y procesar datos
    print("Cargando datos...")
    data = data_processor.load_clean_data('data/dataset.csv')
    print(f"Datos cargados: {len(data)} registros")
    print(f"Caracteristicas: {len(data_processor.features)}")

    # Mostrar distribucion
    class_counts = data['binary_class'].value_counts()
    print("Distribucion de clases:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(data)*100:.1f}%)")

    # Preparar datos
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_split(data)
    print(f"\nDatos de entrenamiento: {len(X_train)}")
    print(f"Datos de test: {len(X_test)}")

    # === MODELO TOTAL ===
    print("\n=== MODELO DE COBERTURA TOTAL ===")
    total_model = TotalCoverageModel()
    total_acc_train = total_model.train(X_train, y_train)
    print(f"Precision en entrenamiento: {total_acc_train:.4f} ({total_acc_train*100:.2f}%)")

    # Evaluar modelo total
    total_preds = total_model.predict(X_test)
    from sklearn.metrics import accuracy_score
    total_acc_test = accuracy_score(y_test, total_preds)
    print(f"Precision en test: {total_acc_test:.4f} ({total_acc_test*100:.2f}%)")
    print(f"Cobertura: 100.0%")

    # === MODELO PARCIAL ===
    print("\n=== MODELO DE COBERTURA PARCIAL ===")
    # Identificar casos extremos
    extreme_indices = data_processor.identify_extreme_cases(data)
    extreme_data = data.loc[extreme_indices].copy()
    print(f"Casos extremos identificados: {len(extreme_data)} ({len(extreme_data)/len(data)*100:.1f}%)")

    if len(extreme_data) > 0:
        partial_model = PartialCoverageModel()
        X_extreme = extreme_data[data_processor.features].values
        y_extreme = extreme_data['binary_class'].values
        partial_acc_train = partial_model.train(X_extreme, y_extreme)
        print(f"Precision en entrenamiento: {partial_acc_train:.4f} ({partial_acc_train*100:.2f}%)")

        # Evaluar en casos extremos del test
        extreme_test_indices = [i for i in range(len(X_test)) if data_processor.is_extreme_case(X_test[i])]
        if len(extreme_test_indices) > 0:
            X_extreme_test = X_test[extreme_test_indices]
            y_extreme_test = y_test[extreme_test_indices]
            partial_preds = partial_model.predict(X_extreme_test)
            partial_acc_test = accuracy_score(y_extreme_test, partial_preds)
            partial_coverage = len(extreme_test_indices) / len(y_test) * 100
            print(f"Precision en test: {partial_acc_test:.4f} ({partial_acc_test*100:.2f}%)")
            print(f"Cobertura: {partial_coverage:.1f}%")
        else:
            print("No hay casos extremos en el conjunto de test")
            partial_acc_test = 0
            partial_coverage = 0
    else:
        print("No se encontraron casos extremos para entrenar")
        partial_acc_test = 0
        partial_coverage = 0

    # === RESULTADOS FINALES ===
    print("\n" + "=" * 50)
    print("RESULTADOS FINALES")
    print("=" * 50)
    print(f"MODELO TOTAL:")
    print(f"  Precision: {total_acc_test:.4f} ({total_acc_test*100:.2f}%)")
    print(f"  Cobertura: 100.0%")

    if partial_coverage > 0:
        print(f"\nMODELO PARCIAL:")
        print(f"  Precision: {partial_acc_test:.4f} ({partial_acc_test*100:.2f}%)")
        print(f"  Cobertura: {partial_coverage:.1f}%")

        if partial_acc_test > total_acc_test:
            print(f"\n🎉 EXITO! Modelo parcial supera al total")
            print(f"   {partial_acc_test*100:.2f}% > {total_acc_test*100:.2f}%")
        else:
            print(f"\n📊 Modelo total es superior")
            print(f"   {total_acc_test*100:.2f}% >= {partial_acc_test*100:.2f}%")

    # === GUARDAR MODELOS ===
    print("\n💾 Guardando modelos...")
    total_model.save('saved_models/total_model.pkl', 'saved_models/total_scaler.pkl', 'saved_models/total_encoder.pkl')
    if 'partial_model' in locals():
        partial_model.save('saved_models/partial_model.pkl', 'saved_models/partial_scaler.pkl')
    print("✅ Modelos guardados exitosamente")

    return {
        'total_model': total_model,
        'total_accuracy': total_acc_test,
        'partial_accuracy': partial_acc_test if partial_coverage > 0 else None,
        'partial_coverage': partial_coverage
    }

if __name__ == "__main__":
    main()
