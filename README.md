# 🚀 Sistema de Clasificación de Exoplanetas

Sistema optimizado de machine learning para clasificación de exoplanetas del catálogo Kepler con **2 modelos especializados** para diferentes necesidades.

## 📊 Modelos Disponibles

### 🎯 Modelo Ultra-Preciso

- **Precisión**: 99.7% en decisiones definitivas
- **Casos inciertos**: ~52%
- **Ideal para**: Aplicaciones críticas, investigación científica
- **Uso**: Cuando los errores son muy costosos

### 🎯 Modelo de Cobertura Binaria

- **Precisión**: 98.6% en decisiones definitivas
- **Casos inciertos**: ~0% (cobertura completa)
- **Clases**: Solo 2 - Exoplaneta vs No Exoplaneta
- **Ideal para**: Screening automático, decisiones simples
- **Uso**: Clasificación binaria rápida, pipelines automatizados

## 🏗️ Estructura del Proyecto

```
├── dataset.csv                         # Dataset de Kepler
├── main_system.py                      # Sistema principal
├── max_precision_optimizer.py          # Modelo ultra-preciso
├── train_coverage_model.py             # Entrenamiento cobertura completa
├── clean_project.py                    # Script de limpieza
├── README.md                           # Esta documentación
├── requirements.txt                     # Dependencias
├── models/                              # Arquitecturas de red
├── data_loader/                         # Cargadores de datos
├── base/                               # Clases base
├── utils/                              # Utilidades
└── configs/                            # Configuraciones
```

## Estructura del Dataset

El proyecto utiliza el catálogo Kepler Object of Interest (KOI) con características como:

- `koi_disposition`: Variable objetivo (CONFIRMED, CANDIDATE, FALSE POSITIVE)
- `koi_period`: Período orbital
- `koi_prad`: Radio del planeta
- `koi_teq`: Temperatura de equilibrio
- `koi_depth`: Profundidad del tránsito
- Y 59 características astronómicas seleccionadas automáticamente

## 🚀 Instalación y Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar modelos

**Modelo Ultra-Preciso** (si no existe):

```bash
python max_precision_optimizer.py
```

**Modelo de Cobertura Binaria**:

```bash
python train_coverage_model.py
```

### 3. Usar el sistema

**Comparar modelos**:

```bash
python main_system.py --compare
```

**Demo con modelo binario**:

```bash
python main_system.py --demo --model cobertura_completa
```

**Demo con modelo ultra-preciso**:

```bash
python main_system.py --demo --model ultra_preciso
```

**Uso automático** (elige el mejor modelo):

```bash
python main_system.py --demo
```

## 📋 Uso Programático

### Cargar y usar modelos

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Modelo de Cobertura Completa
model_coverage = load_model('modelo_cobertura_completa.h5')
scaler_coverage = joblib.load('scaler_cobertura_completa.pkl')
config_coverage = joblib.load('config_cobertura_completa.pkl')

# Modelo Ultra-Preciso
model_precision = load_model('modelo_softmax_final.h5')
scaler_precision = joblib.load('scaler_softmax_final.pkl')
config_precision = joblib.load('max_precision_config.pkl')

# Hacer predicciones
X_new = scaler_coverage.transform(your_data)
predictions = model_coverage.predict(X_new)
```

### Sistema completo

```python
from main_system import ExoplanetClassificationSystem

# Inicializar sistema
system = ExoplanetClassificationSystem()

# Hacer predicciones
results = system.predict_with_model(your_data, model_type='cobertura_completa')

# Analizar resultados
system.analyze_predictions(results)
```

## 🎯 Guía de Selección de Modelo

### Usa **Modelo Ultra-Preciso** cuando:

- ❌ Los errores falsos positivos/negativos son muy costosos
- 🔬 Investigación científica que requiere alta confiabilidad
- ⚠️ Puedes manejar muchos casos "inciertos"
- 🎯 Precisión > Cobertura

### Usa **Modelo de Cobertura Binaria** cuando:

- ⚡ Necesitas decisiones simples: ¿Es exoplaneta o no?
- 🏭 Pipelines de screening automático
- 📊 Clasificación binaria rápida de grandes datasets
- 🎯 Simplicidad > Granularidad (solo 2 clases)

## 📊 Resultados de Rendimiento

| Modelo                | Precisión | Casos Inciertos | Clases   | Uso Recomendado                  |
| --------------------- | --------- | --------------- | -------- | -------------------------------- |
| **Ultra-Preciso**     | 99.7%     | 52%             | 3 clases | Investigación, validación        |
| **Cobertura Binaria** | 98.6%     | 0%              | 2 clases | Producción, screening automático |

## 📁 Archivos de Modelos

### Modelo Ultra-Preciso

- `modelo_softmax_final.h5` - Red neuronal entrenada
- `scaler_softmax_final.pkl` - Normalizador de datos
- `max_precision_config.pkl` - Configuración de umbrales

### Modelo de Cobertura Binaria

- `modelo_cobertura_completa.h5` - Red neuronal entrenada (binaria)
- `scaler_cobertura_completa.pkl` - Normalizador de datos
- `config_cobertura_completa.pkl` - Configuración de umbrales

## 🎉 Características del Sistema

✅ **2 modelos especializados** para diferentes necesidades
✅ **Sistema unificado** de predicción y análisis
✅ **Código limpio** sin archivos basura
✅ **Fácil de usar** con scripts principales
✅ **Bien documentado** con ejemplos prácticos
✅ **Modular** y extensible
✅ **Optimizado** para producción

---

**Desarrollado para clasificación robusta y eficiente de exoplanetas del catálogo Kepler** 🌟
