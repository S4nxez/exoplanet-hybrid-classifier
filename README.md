# 🚀 Sistema de Clasificación de Exoplanetas# 🚀 Sistema de Clasificación de Exoplanetas



Sistema optimizado de machine learning para clasificación de exoplanetas del catálogo Kepler con **arquitectura modular** y **3 modelos especializados**.Sistema optimizado de machine learning para clasificación de exoplanetas del catálogo Kepler con **arquitectura modular** y **3 modelos especializados**.



## 📊 Modelos Disponibles## 📊 Modelos Disponibles



### 🎯 Modelo Parcial (Ultra-Preciso)### 🎯 Modelo Parcial (Ultra-Preciso)

- **Precisión**: 99.2% en decisiones definitivas- **Precisión**: 99.2% en decisiones definitivas

- **Coverage**: ~47% de casos- **Coverage**: ~47% de casos

- **Ideal para**: Aplicaciones críticas donde los errores son costosos- **Ideal para**: Aplicaciones críticas donde los errores son costosos



### 🌐 Modelo Total (Cobertura Completa)  ### 🌐 Modelo Total (Cobertura Completa)  

- **Precisión**: 85.6% en todos los casos- **Precisión**: 85.6% en todos los casos

- **Coverage**: 100% de casos- **Coverage**: 100% de casos

- **Ideal para**: Aplicaciones generales, análisis exploratorio- **Ideal para**: Aplicaciones generales, análisis exploratorio



### 🤖 Modelo Híbrido (Mejor de ambos mundos)### 🤖 Modelo Híbrido (Mejor de ambos mundos)

- **Precisión**: 85.8% con sistema de cascada inteligente- **Precisión**: 85.8% con sistema de cascada inteligente

- **Sistema**: Cascada (30%) + Stacking (70%)- **Sistema**: Cascada (30%) + Stacking (70%)

- **Ideal para**: Máximo rendimiento, aplicaciones de producción- **Ideal para**: Máximo rendimiento, aplicaciones de producción



## 📁 Estructura del Proyecto Organizada## 📁 Estructura del Proyecto Organizada



``````

├── data/├── data/

│   └── dataset.csv              # Dataset de exoplanetas│   └── dataset.csv              # Dataset de exoplanetas

├── src/                         # 🔧 Código fuente modular├── src/                         # 🔧 Código fuente modular

│   ├── models/                  # Definiciones de modelos ML│   ├── models/                  # Definiciones de modelos ML

│   │   ├── partial_coverage.py  # Modelo parcial│   │   ├── partial_coverage.py  # Modelo parcial

│   │   ├── total_coverage.py    # Modelo total│   │   ├── total_coverage.py    # Modelo total

│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow

│   ├── trainers/                # 🏗️ Entrenadores con herencia│   ├── trainers/                # 🏗️ Entrenadores con herencia

│   │   ├── base_trainer.py      # Clase base abstracta│   │   ├── base_trainer.py      # Clase base abstracta

│   │   ├── total_trainer.py     # Entrenador modelo total│   │   ├── total_trainer.py     # Entrenador modelo total

│   │   ├── partial_trainer.py   # Entrenador modelo parcial│   │   ├── partial_trainer.py   # Entrenador modelo parcial

│   │   └── hybrid_trainer.py    # Entrenador modelo híbrido│   │   └── hybrid_trainer.py    # Entrenador modelo híbrido

│   ├── evaluators/              # 📊 Evaluadores especializados│   ├── evaluators/              # 📊 Evaluadores especializados

│   │   ├── base_evaluator.py    # Clase base abstracta│   │   ├── base_evaluator.py    # Clase base abstracta

│   │   └── model_evaluators.py  # Evaluadores específicos│   │   └── model_evaluators.py  # Evaluadores específicos

│   └── utils/                   # 🛠️ Utilidades compartidas│   └── utils/                   # 🛠️ Utilidades compartidas

│       └── data_processor.py    # Procesamiento de datos│       └── data_processor.py    # Procesamiento de datos

├── scripts/                     # 📜 Scripts organizados├── scripts/                     # 📜 Scripts organizados

│   ├── training/                # Scripts de entrenamiento│   ├── training/                # Scripts de entrenamiento

│   │   ├── train_total.py       # Entrenar modelo total│   │   ├── train_total.py       # Entrenar modelo total

│   │   ├── train_partial.py     # Entrenar modelo parcial│   │   ├── train_partial.py     # Entrenar modelo parcial

│   │   └── train_hybrid.py      # Entrenar modelo híbrido│   │   └── train_hybrid.py      # Entrenar modelo híbrido

│   ├── prediction/              # Scripts de predicción│   ├── prediction/              # Scripts de predicción

│   │   └── predict_models.py    # Predicciones con argumentos│   │   └── predict_models.py    # Predicciones con argumentos

│   ├── utils/                   # Utilidades│   ├── utils/                   # Utilidades

│   │   └── clean_project.py     # Limpieza de proyecto│   │   └── clean_project.py     # Limpieza de proyecto

│   ├── main_training.py         # 🚀 Script principal de entrenamiento│   ├── main_training.py         # 🚀 Script principal de entrenamiento

│   ├── main_prediction.py       # 🔍 Script principal de predicción│   ├── main_prediction.py       # 🔍 Script principal de predicción

│   └── project_info.py          # ℹ️ Información del proyecto│   └── project_info.py          # ℹ️ Información del proyecto

├── saved_models/                # 💾 Modelos entrenados├── saved_models/                # 💾 Modelos entrenados

├── train_*.py                   # 🔗 Wrappers de compatibilidad├── train_*.py                   # 🔗 Wrappers de compatibilidad

├── predict_model.py             # 🔗 Wrapper de compatibilidad├── predict_model.py             # 🔗 Wrapper de compatibilidad

└── clean_project.py             # 🔗 Wrapper de compatibilidad└── clean_project.py             # 🔗 Wrapper de compatibilidad

```

│   │   └── model_evaluators.py  # Evaluadores específicos

## 🛠️ Instalación

│   └── utils/├── src/### 🎯 Modelo de Cobertura Binaria

1. **Instalar dependencias:**

```bash│       └── data_processor.py    # Procesador de datos

pip install -r requirements.txt

```├── scripts/                     # Scripts organizados│   ├── models/



2. **Verificar dataset:**│   ├── training/                # Scripts de entrenamiento

```bash

ls data/dataset.csv│   │   ├── train_total.py       # Entrenar modelo total│   │   ├── partial_coverage.py  # Modelo parcial- **Precisión**: 98.6% en decisiones definitivas

```

│   │   ├── train_partial.py     # Entrenar modelo parcial

## 🚀 Uso Rápido

│   │   └── train_hybrid.py      # Entrenar modelo híbrido│   │   ├── total_coverage.py    # Modelo total- **Casos inciertos**: ~0% (cobertura completa)

### Scripts Principales Unificados (Recomendado)

│   ├── prediction/              # Scripts de predicción

```bash

# Entrenar todos los modelos│   │   └── predict_models.py    # Predictor universal│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow- **Clases**: Solo 2 - Exoplaneta vs No Exoplaneta

python scripts/main_training.py --model all

│   └── utils/                   # Utilidades

# Entrenar modelo específico

python scripts/main_training.py --model total│       └── clean_project.py     # Limpiador de proyecto│   └── utils/- **Ideal para**: Screening automático, decisiones simples

python scripts/main_training.py --model partial  

python scripts/main_training.py --model hybrid├── saved_models/               # Modelos entrenados



# Evaluar y comparar modelos├── train_partial_model.py      # Wrapper compatibilidad│       └── data_processor.py    # Procesador de datos- **Uso**: Clasificación binaria rápida, pipelines automatizados

python scripts/main_prediction.py --compare

├── train_total_model.py        # Wrapper compatibilidad  

# Ver información del proyecto

python scripts/project_info.py├── train_hybrid_model.py       # Wrapper compatibilidad├── saved_models/               # Modelos entrenados

```

├── predict_model.py            # Wrapper compatibilidad

### Scripts Individuales Organizados

└── clean_project.py            # Wrapper compatibilidad├── train_partial_model.py      # Entrenador modelo parcial## 🏗️ Estructura del Proyecto

```bash

# Entrenamiento individual```

python scripts/training/train_total.py

python scripts/training/train_partial.py├── train_total_model.py        # Entrenador modelo total

python scripts/training/train_hybrid.py

## 🛠️ Instalación

# Predicción individual

python scripts/prediction/predict_models.py --model total├── train_hybrid_model.py       # Entrenador modelo híbrido```

python scripts/prediction/predict_models.py --model partial

python scripts/prediction/predict_models.py --model hybrid1. Instalar dependencias:

```

```bash└── predict_model.py            # Predictor universal├── dataset.csv                         # Dataset de Kepler

### Wrappers de Compatibilidad (Retrocompatibilidad)

pip install -r requirements.txt

```bash

# Mantienen la API original``````├── main_system.py                      # Sistema principal

python train_total_model.py

python train_partial_model.py

python train_hybrid_model.py

python predict_model.py --model total2. Verificar que existe el dataset:├── max_precision_optimizer.py          # Modelo ultra-preciso

```

```bash

## 🏗️ Arquitectura del Sistema

ls data/dataset.csv## 🛠️ Instalación├── train_coverage_model.py             # Entrenamiento cobertura completa

### Diseño Modular con Herencia

```

- **BaseTrainer**: Clase abstracta con pipeline común de entrenamiento

- **BaseEvaluator**: Clase abstracta para evaluación consistente  ├── clean_project.py                    # Script de limpieza

- **Trainers específicos**: Heredan funcionalidad común, implementan lógica específica

- **Separación de responsabilidades**: Entrenamiento vs. Evaluación vs. Scripts## 🎯 Entrenar Modelos



### Patrón de Diseño1. Instalar dependencias:├── README.md                           # Esta documentación



```python### Opción 1: Scripts Organizados (Recomendado)

# Ejemplo de uso de la arquitectura modular

from src.trainers import TotalModelTrainer```bash├── requirements.txt                     # Dependencias



trainer = TotalModelTrainer()```bash

model, accuracy = trainer.run_training_pipeline()

```# Modelo totalpip install -r requirements.txt├── models/                              # Arquitecturas de red



## 📈 Resultados de Rendimientopython scripts/training/train_total.py



| Modelo | Accuracy | F1-Score | Coverage | Características |```├── data_loader/                         # Cargadores de datos

|--------|----------|----------|----------|-----------------|

| 🎯 Parcial | 85.83% | 0.749 | 47.26% | Alta confianza |# Modelo parcial  

| 🌐 Total | 85.57% | 0.740 | 100% | Cobertura completa |

| 🤖 Híbrido | 85.78% | 0.745 | 100% | Sistema inteligente |python scripts/training/train_partial.py├── base/                               # Clases base



## 🔧 Utilidades



```bash# Modelo híbrido2. Verificar que existe el dataset:├── utils/                              # Utilidades

# Limpiar archivos temporales

python scripts/utils/clean_project.pypython scripts/training/train_hybrid.py



# Ver información completa del proyecto``````bash└── configs/                            # Configuraciones

python scripts/project_info.py



# Verificar modelos guardados

ls saved_models/### Opción 2: Wrappers de Compatibilidadls data/dataset.csv```

```



## 🎯 Casos de Uso

```bash```

### Para Investigación Científica

- Usar **Modelo Parcial** para análisis críticospython train_total_model.py    # Modelo total

- Alta precisión en casos definitivos

- Ideal cuando los falsos positivos son costosospython train_partial_model.py  # Modelo parcial## Estructura del Dataset



### Para Aplicaciones Generalespython train_hybrid_model.py   # Modelo híbrido

- Usar **Modelo Total** para screening completo

- Cobertura del 100% de casos```## 🎯 Entrenar Modelos

- Buena precisión general



### Para Aplicaciones de Producción

- Usar **Modelo Híbrido** para máximo rendimiento## 🔮 Realizar PrediccionesEl proyecto utiliza el catálogo Kepler Object of Interest (KOI) con características como:

- Combina lo mejor de ambos enfoques

- Sistema de cascada optimizado



## 📋 Características Técnicas### Opción 1: Script Organizado (Recomendado)### Modelo Parcial (Casos Extremos)



- **Lenguaje**: Python 3.13+

- **ML Frameworks**: Scikit-learn, TensorFlow/Keras

- **Arquitectura**: Modular con herencia y composición```bash```bash- `koi_disposition`: Variable objetivo (CONFIRMED, CANDIDATE, FALSE POSITIVE)

- **Patrones**: Strategy, Template Method, Abstract Factory

- **Compatibilidad**: Wrappers para retrocompatibilidad# Usar modelo específico

- **Organización**: Scripts organizados por funcionalidad

python scripts/prediction/predict_models.py --model totalpython train_partial_model.py- `koi_period`: Período orbital

## 🔄 Flujo de Trabajo

python scripts/prediction/predict_models.py --model partial

1. **Entrenamiento**: `scripts/main_training.py --model all`

2. **Evaluación**: `scripts/main_prediction.py --compare`python scripts/prediction/predict_models.py --model hybrid```- `koi_prad`: Radio del planeta

3. **Limpieza**: `scripts/utils/clean_project.py`

4. **Información**: `scripts/project_info.py````



## 🏆 Ventajas de la Arquitectura- `koi_teq`: Temperatura de equilibrio



✅ **Código modular**: Fácil mantenimiento y extensión  ### Opción 2: Wrapper de Compatibilidad

✅ **Herencia bien diseñada**: Reutilización sin duplicación  

✅ **Scripts organizados**: Separación clara de responsabilidades  ### Modelo Total (Baseline)- `koi_depth`: Profundidad del tránsito

✅ **Compatibilidad**: Wrappers mantienen API original  

✅ **Escalabilidad**: Fácil agregar nuevos modelos  ```bash

✅ **Testing**: Estructura facilita pruebas unitarias  

python predict_model.py --model total```bash- Y 59 características astronómicas seleccionadas automáticamente

---

python predict_model.py --model partial

**Proyecto organizado con arquitectura modular y herencia. Compatibilidad hacia atrás mantenida. Scripts principales unificados disponibles.**
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
