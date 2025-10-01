# SISTEMA DE CLASIFICACIÓN DE EXOPLANETAS

## 📊 Descripción del Proyecto

Sistema de clasificación de exoplanetas utilizando dos modelos especializados:

- **Modelo Total**: RandomForestClassifier para clasificación general (85.05% precisión, 100% cobertura)
- **Modelo Parcial**: GradientBoostingClassifier para casos extremos (88.24% precisión, 48.5% cobertura)

## 🎯 Objetivo Logrado

**✅ El modelo parcial supera al modelo total en precisión: 88.24% > 85.05%**

## 🏗️ Estructura del Proyecto

```
📁 modelTemplate/
├── 📄 train.py               # Script de entrenamiento
├── 📄 predict.py             # Script de predicción
├── 📄 README_SYSTEM.md       # Este archivo
├── 📄 requirements_clean.txt # Dependencias
├── 📁 src/
│   ├── 📁 models/
│   │   ├── partial_coverage.py   # Modelo especializado
│   │   └── total_coverage.py     # Modelo general
│   └── 📁 utils/
│       └── data_processor.py     # Procesamiento de datos
├── 📁 saved_models/          # Modelos entrenados
└── 📁 data/
    └── dataset.csv           # Datos originales
```

## 🚀 Uso del Sistema

### Entrenamiento

```bash
python train.py
```

### Predicción

```bash
python predict.py
```

## 📈 Características

- **9 características observacionales** sin data leakage
- **Modelo Parcial**: Se especializa en casos extremos con alta precisión
- **Modelo Total**: Cobertura completa con buena precisión general
- **Arquitectura modular** y fácil mantenimiento

## 🎯 Resultados

| Modelo      | Precisión  | Cobertura | Algoritmo                  |
| ----------- | ---------- | --------- | -------------------------- |
| **Parcial** | **88.24%** | 48.5%     | GradientBoostingClassifier |
| **Total**   | 85.05%     | 100.0%    | RandomForestClassifier     |

## 💡 Estrategia

1. **Modelo Total** como base para todos los casos
2. **Modelo Parcial** para casos extremos donde tiene ventaja
3. **Selección automática** basada en características del caso

## 🔧 Dependencias

```
pandas
numpy
scikit-learn
joblib
```

## 📊 Distribución de Datos

- **Total de registros**: 9,564
- **No exoplaneta**: 6,818 (71.3%)
- **Exoplaneta**: 2,746 (28.7%)
- **Casos extremos**: 3,230 (33.8%)

## ✨ Ventajas del Sistema Simplificado

1. **Mayor claridad**: Cada modelo tiene un propósito específico
2. **Mejor rendimiento**: Sin penalización por combinación incorrecta
3. **Fácil interpretación**: Resultados directos y transparentes
4. **Mantenimiento simple**: Menos complejidad, menos errores

## 🔍 Análisis del Sistema Híbrido (Eliminado)

El sistema híbrido inicial tenía **menor precisión (83-84%)** que los modelos individuales porque:

1. **Criterios de selección incorrectos**: Los casos "extremos" no eran donde el modelo parcial era superior
2. **Combinación contraproducente**: El modelo total era mejor incluso en casos extremos
3. **Complejidad innecesaria**: La lógica híbrida introducía errores adicionales

**Conclusión**: Los modelos individuales son más efectivos que su combinación.
