# 🚀 Sistema de Clasificación de Exoplanetas

Sistema optimizado de machine learning para clasificación de exoplanetas del catálogo Kepler con **arquitectura modular** y **3 modelos especializados**.

## 📊 Modelos Disponibles

### 🎯 Modelo Parcial (Ultra-Preciso)
- **Precisión**: 99.2% en decisiones definitivas
- **Coverage**: ~47% de casos
- **Ideal para**: Aplicaciones críticas donde los errores son costosos

### 🌐 Modelo Total (Cobertura Completa)  
- **Precisión**: 85.6% en todos los casos
- **Coverage**: 100% de casos
- **Ideal para**: Aplicaciones generales, análisis exploratorio

### 🤖 Modelo Híbrido (Mejor de ambos mundos)
- **Precisión**: 85.8% con sistema de cascada inteligente
- **Sistema**: Cascada (30%) + Stacking (70%)
- **Ideal para**: Máximo rendimiento, aplicaciones de producción

## 📁 Estructura del Proyecto Organizada

```
├── data/
│   └── dataset.csv              # Dataset de exoplanetas
├── src/                         # 🔧 Código fuente modular
│   ├── models/                  # Definiciones de modelos ML
│   │   ├── partial_coverage.py  # Modelo parcial
│   │   ├── total_coverage.py    # Modelo total
│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow
│   ├── trainers/                # 🏗️ Entrenadores con herencia
│   │   ├── base_trainer.py      # Clase base abstracta
│   │   ├── total_trainer.py     # Entrenador modelo total
│   │   ├── partial_trainer.py   # Entrenador modelo parcial
│   │   └── hybrid_trainer.py    # Entrenador modelo híbrido
│   ├── evaluators/              # 📊 Evaluadores especializados
│   │   ├── base_evaluator.py    # Clase base abstracta
│   │   └── model_evaluators.py  # Evaluadores específicos
│   └── utils/                   # 🛠️ Utilidades compartidas
│       └── data_processor.py    # Procesamiento de datos
├── scripts/                     # 📜 Scripts organizados
│   ├── training/                # Scripts de entrenamiento
│   │   ├── train_total.py       # Entrenar modelo total
│   │   ├── train_partial.py     # Entrenar modelo parcial
│   │   └── train_hybrid.py      # Entrenar modelo híbrido
│   ├── prediction/              # Scripts de predicción
│   │   └── predict_models.py    # Predicciones con argumentos
│   ├── utils/                   # Utilidades
│   │   └── clean_project.py     # Limpieza de proyecto
│   ├── main_training.py         # 🚀 Script principal de entrenamiento
│   ├── main_prediction.py       # 🔍 Script principal de predicción
│   └── project_info.py          # ℹ️ Información del proyecto
├── saved_models/                # 💾 Modelos entrenados
├── train_*.py                   # 🔗 Wrappers de compatibilidad
├── predict_model.py             # 🔗 Wrapper de compatibilidad
└── clean_project.py             # 🔗 Wrapper de compatibilidad

│   │   └── model_evaluators.py  # Evaluadores específicos

│   └── utils/├── src/### 🎯 Modelo de Cobertura Binaria

│       └── data_processor.py    # Procesador de datos

├── scripts/                     # Scripts organizados│   ├── models/

│   ├── training/                # Scripts de entrenamiento

│   │   ├── train_total.py       # Entrenar modelo total│   │   ├── partial_coverage.py  # Modelo parcial- **Precisión**: 98.6% en decisiones definitivas

│   │   ├── train_partial.py     # Entrenar modelo parcial

│   │   └── train_hybrid.py      # Entrenar modelo híbrido│   │   ├── total_coverage.py    # Modelo total- **Casos inciertos**: ~0% (cobertura completa)

│   ├── prediction/              # Scripts de predicción

│   │   └── predict_models.py    # Predictor universal│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow- **Clases**: Solo 2 - Exoplaneta vs No Exoplaneta

│   └── utils/                   # Utilidades

│       └── clean_project.py     # Limpiador de proyecto│   └── utils/- **Ideal para**: Screening automático, decisiones simples

├── saved_models/               # Modelos entrenados

├── train_partial_model.py      # Wrapper compatibilidad│       └── data_processor.py    # Procesador de datos- **Uso**: Clasificación binaria rápida, pipelines automatizados

├── train_total_model.py        # Wrapper compatibilidad  

├── train_hybrid_model.py       # Wrapper compatibilidad├── saved_models/               # Modelos entrenados

├── predict_model.py            # Wrapper compatibilidad

└── clean_project.py            # Wrapper compatibilidad├── train_partial_model.py      # Entrenador modelo parcial## 🏗️ Estructura del Proyecto

```

├── train_total_model.py        # Entrenador modelo total

## 🛠️ Instalación

├── train_hybrid_model.py       # Entrenador modelo híbrido```

1. Instalar dependencias:

```bash└── predict_model.py            # Predictor universal├── dataset.csv                         # Dataset de Kepler

pip install -r requirements.txt

``````├── main_system.py                      # Sistema principal



2. Verificar que existe el dataset:├── max_precision_optimizer.py          # Modelo ultra-preciso

```bash

ls data/dataset.csv## 🛠️ Instalación├── train_coverage_model.py             # Entrenamiento cobertura completa

```

├── clean_project.py                    # Script de limpieza

## 🎯 Entrenar Modelos

1. Instalar dependencias:├── README.md                           # Esta documentación

### Opción 1: Scripts Organizados (Recomendado)

```bash├── requirements.txt                     # Dependencias

```bash

# Modelo totalpip install -r requirements.txt├── models/                              # Arquitecturas de red

python scripts/training/train_total.py

```├── data_loader/                         # Cargadores de datos

# Modelo parcial  

python scripts/training/train_partial.py├── base/                               # Clases base



# Modelo híbrido2. Verificar que existe el dataset:├── utils/                              # Utilidades

python scripts/training/train_hybrid.py

``````bash└── configs/                            # Configuraciones



### Opción 2: Wrappers de Compatibilidadls data/dataset.csv```



```bash```

python train_total_model.py    # Modelo total

python train_partial_model.py  # Modelo parcial## Estructura del Dataset

python train_hybrid_model.py   # Modelo híbrido

```## 🎯 Entrenar Modelos



## 🔮 Realizar PrediccionesEl proyecto utiliza el catálogo Kepler Object of Interest (KOI) con características como:



### Opción 1: Script Organizado (Recomendado)### Modelo Parcial (Casos Extremos)



```bash```bash- `koi_disposition`: Variable objetivo (CONFIRMED, CANDIDATE, FALSE POSITIVE)

# Usar modelo específico

python scripts/prediction/predict_models.py --model totalpython train_partial_model.py- `koi_period`: Período orbital

python scripts/prediction/predict_models.py --model partial

python scripts/prediction/predict_models.py --model hybrid```- `koi_prad`: Radio del planeta

```

- `koi_teq`: Temperatura de equilibrio

### Opción 2: Wrapper de Compatibilidad

### Modelo Total (Baseline)- `koi_depth`: Profundidad del tránsito

```bash

python predict_model.py --model total```bash- Y 59 características astronómicas seleccionadas automáticamente

python predict_model.py --model partial

python predict_model.py --model hybridpython train_total_model.py

```

```## 🚀 Instalación y Uso

## 🧹 Limpieza del Proyecto



```bash

# Script organizado### Modelo Híbrido (TensorFlow)### 1. Instalar dependencias

python scripts/utils/clean_project.py

```bash

# O wrapper

python clean_project.pypython train_hybrid_model.py```bash

```

```pip install -r requirements.txt

## 📊 Resultados Esperados

```

- **Modelo Parcial**: ~85.8% accuracy con alta cobertura en casos extremos

- **Modelo Total**: ~85.6% accuracy como baseline sólido## 🔮 Realizar Predicciones

- **Modelo Híbrido**: ~85.7% accuracy superando al baseline mediante sistema de cascada

### 2. Entrenar modelos

## 🧠 Arquitectura del Sistema Híbrido

### Usar Modelo Parcial

1. **🔹 Cascada**: Usa modelo parcial para casos extremos de alta confianza (30% de casos, 99.3% accuracy)

2. **🔹 Stacking Inteligente**: Combina modelo total con TensorFlow para casos restantes```bash**Modelo Ultra-Preciso** (si no existe):

3. **🧠 Backbone Fuerte**: Prioriza modelo total como base confiable

python predict_model.py --model partial

## 🏗️ Refactorización Aplicada

``````bash

### ✅ Beneficios de la Nueva Estructura:

python max_precision_optimizer.py

1. **📁 Organización**: Código separado por responsabilidades

2. **🔧 Reutilización**: Clases base para entrenadores y evaluadores### Usar Modelo Total```

3. **📦 Modularidad**: Cada componente en su módulo específico

4. **🔄 Compatibilidad**: Wrappers mantienen interfaz original```bash

5. **🧹 Limpieza**: Scripts más cortos y enfocados

python predict_model.py --model total**Modelo de Cobertura Binaria**:

### 🎯 Arquitectura de Clases:

```

- **BaseTrainer**: Clase base para todos los entrenadores

- **BaseEvaluator**: Clase base para todos los evaluadores```bash

- **ModelTrainers**: Entrenadores específicos que heredan funcionalidad común

- **ModelEvaluators**: Evaluadores específicos con análisis personalizado### Usar Modelo Híbridopython train_coverage_model.py



## 🚀 Ejemplo de Uso Rápido```bash```



```bashpython predict_model.py --model hybrid

# Entrenar todos los modelos (método organizado)

python scripts/training/train_total.py```### 3. Usar el sistema

python scripts/training/train_partial.py  

python scripts/training/train_hybrid.py



# Comparar resultados## 📊 Resultados Esperados**Comparar modelos**:

python scripts/prediction/predict_models.py --model total

python scripts/prediction/predict_models.py --model hybrid

```

- **Modelo Parcial**: ~85.8% accuracy con alta cobertura en casos extremos```bash

## 🎯 Características Técnicas

- **Modelo Total**: ~85.6% accuracy como baseline sólidopython main_system.py --compare

- **Datos**: 9 características numéricas de exoplanetas

- **Clases**: Binario (Exoplaneta / No exoplaneta)- **Modelo Híbrido**: ~85.7% accuracy superando al baseline mediante sistema de cascada```

- **Split**: 80% entrenamiento, 20% prueba (estratificado)

- **Modelos Base**: scikit-learn (GradientBoosting, RandomForest)

- **Modelo Híbrido**: TensorFlow con arquitectura optimizada

- **Refactorización**: Herencia, composición y separación de responsabilidades## 🧠 Arquitectura del Sistema Híbrido**Demo con modelo binario**:



1. **🔹 Cascada**: Usa modelo parcial para casos extremos de alta confianza (30% de casos, 99.3% accuracy)```bash

2. **🔹 Stacking Inteligente**: Combina modelo total con TensorFlow para casos restantespython main_system.py --demo --model cobertura_completa

3. **🧠 Backbone Fuerte**: Prioriza modelo total como base confiable```



## 🎯 Características Técnicas**Demo con modelo ultra-preciso**:



- **Datos**: 9 características numéricas de exoplanetas```bash

- **Clases**: Binario (Exoplaneta / No exoplaneta)python main_system.py --demo --model ultra_preciso

- **Split**: 80% entrenamiento, 20% prueba (estratificado)```

- **Modelos Base**: scikit-learn (GradientBoosting, RandomForest)

- **Modelo Híbrido**: TensorFlow con arquitectura optimizada**Uso automático** (elige el mejor modelo):



## 📈 Métricas```bash

python main_system.py --demo

- **Accuracy**: Precisión general del modelo```

- **F1-Score**: Balance entre precisión y recall

- **Coverage**: % de casos manejados con alta confianza## 📋 Uso Programático

- **Balance Cascada-Stacking**: Distribución de predicciones

### Cargar y usar modelos

## 🚀 Ejemplo de Uso Rápido

```python

```bashimport joblib

# Entrenar todos los modelosimport numpy as np

python train_partial_model.pyfrom tensorflow.keras.models import load_model

python train_total_model.py  

python train_hybrid_model.py# Modelo de Cobertura Completa

model_coverage = load_model('modelo_cobertura_completa.h5')

# Comparar resultadosscaler_coverage = joblib.load('scaler_cobertura_completa.pkl')

python predict_model.py --model totalconfig_coverage = joblib.load('config_cobertura_completa.pkl')

python predict_model.py --model hybrid

```# Modelo Ultra-Preciso
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
