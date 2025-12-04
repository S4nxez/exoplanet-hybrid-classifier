# 🌟 Director General Multi-Mission - Sistema de Clasificación de Exoplanetas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![Status](https://img.shields.io/badge/Status-Operativo-brightgreen)

**Sistema inteligente para la clasificación automática de candidatos a exoplanetas**
_Combinando Random Forest y TensorFlow mediante Soft Voting adaptativo_

</div>

## 🚀 Descripción General

El **Director General** es un sistema avanzado de Machine Learning diseñado para la clasificación automática de candidatos a exoplanetas provenientes de múltiples misiones espaciales (KOI, TOI, K2). Utiliza una estrategia de ensemble inteligente que combina Random Forest y TensorFlow para maximizar la precisión en la detección.

### 🝆 Resultados Principales

- 🎯 **Accuracy del Director:** 85.39%
- 🌲 **Random Forest individual:** 85.33%
- 🧠 **TensorFlow individual:** 82.24%
- ⚡ **Tiempo de entrenamiento:** < 5 minutos
- 📊 **Evaluado en:** 1,841 candidatos reales de Kepler

## 🔬 Arquitectura del Sistema

### 🎯 Director General

El componente central que toma decisiones inteligentes sobre qué modelo usar para cada predicción:

```
Input Data → Feature Analysis → Model Selection → Prediction
                    ↓
            [RF: 72.46%]  [TF: 27.54%]
                    ↓
            Soft Voting (Weighted Ensemble)
                    ↓
              Final Prediction
```

### 🌝 Sistemas Multi-Misión

1. **🔭 KOI System (Kepler Objects of Interest)**

   - Dataset: 9,564 muestras
   - Features: 64 parámetros físicos y orbitales
   - Estado: ✅ Completamente entrenado

2. **🛰︝ TOI System (TESS Objects of Interest)**

   - Modelos: RF + TF wrapper
   - Estado: ✅ Modelos base entrenados

3. **🌝 K2 System (Kepler Extended Mission)**
   - Modelo: Random Forest especializado
   - Estado: ✅ Modelo disponible

## 📝 Estructura del Proyecto

```
modelTemplate/
├── 📂 src/                          # Código fuente principal
│   ├── models/
│   │   ├── general_director.py      # 🎯 Director General
│   │   ├── koi_randomforest.py     # 🌲 RF para KOI
│   │   └── koi_tensorflow.py       # 🧠 TF para KOI
│   └── utils/                       # Utilidades
├── 📂 koi_system/                   # Sistema KOI completo
│   ├── saved_models/               # ✅ Modelos entrenados
│   ├── config/                     # Configuraciones
│   └── utils/                      # Utilidades específicas
├── 📂 toi_system/                   # Sistema TOI
├── 📂 k2_system/                    # Sistema K2
├── 📂 data/                         # Datasets
│   └── clean/                      # Datos procesados
└── 📂 scripts/                      # Scripts de entrenamiento
```

## 🚀 Instalación y Uso

### 1. Requisitos

```bash
pip install -r requirements.txt
```

**Dependencias principales:**

- Python 3.8+
- TensorFlow 2.x
- scikit-learn 1.x
- pandas, numpy
- joblib

### 2. Entrenamiento Rápido

#### 🔭 Sistema KOI (Principal)

```bash
cd koi_system
python train_koi_system.py
```

#### 🛰︝ Sistema TOI

```bash
cd toi_system
python train_toi_system.py
```

#### 🌝 Sistema K2

```bash
cd k2_system
python train_k2_system.py
```

### 3. Predicción con Director General

```python
from src.models.general_director import GeneralDirector

# Inicializar Director
director = GeneralDirector()

# Predicción automática
predictions, mission_used = director.predict(data, mission='KOI')

# Estadísticas del ensemble
stats = director.get_stats()
print(f"RF usado: {stats['rf_usage']:.1%}")
print(f"TF usado: {stats['tf_usage']:.1%}")
```

## 📊 Métricas de Rendimiento

### 🝆 Comparación de Modelos (Sistema KOI)

| Modelo                  | Accuracy   | Precision | Recall    | F1-Score  |
| ----------------------- | ---------- | --------- | --------- | --------- |
| 🌲 Random Forest        | 85.33%     | 85.1%     | 85.6%     | 85.3%     |
| 🧠 TensorFlow           | 82.24%     | 82.8%     | 81.7%     | 82.2%     |
| 🎯 **Director General** | **85.39%** | **85.2%** | **85.8%** | **85.5%** |

### 🔀 Estrategia del Director

- **Selección inteligente:** El Director elige automáticamente el mejor modelo para cada caso
- **RF predominante:** Usado en 72.46% de casos (alta confianza)
- **TF especializado:** Usado en 27.54% de casos (casos complejos)
- **Accuracy cuando RF elegido:** 86.96%
- **Accuracy cuando TF elegido:** 81.26%

## 🎯 Características Principales

### 🧠 Soft Voting Inteligente

- Combina probabilidades de RF y TF de forma adaptativa
- Pesos dinámicos basados en confianza de predicción
- Optimización automática de umbrales

### ⚡ Rendimiento

- **Entrenamiento rápido:** < 5 minutos para dataset completo
- **Predicción eficiente:** Miles de candidatos por segundo
- **Escalabilidad:** Fácil extensión a nuevas misiones

### 🔧 Robustez

- **Early stopping:** Previene overfitting automáticamente
- **Validación cruzada:** Evaluación rigurosa en datos no vistos
- **Manejo de datos faltantes:** Preprocessamiento robusto

## 🌟 Casos de Uso

### 🔬 Investigación Científica

- Análisis automatizado de datos de Kepler/TESS
- Clasificación de candidatos a exoplanetas
- Reducción de falsos positivos

### 🝭 Aplicaciones Operacionales

- Pipeline automatizado para nuevas observaciones
- Soporte a decisiones en misiones espaciales
- Análisis en tiempo real de datos astronómicos

## Mantenimiento y limpieza segura

- `clean_project.py` ahora funciona como auditor: lista placeholders conocidos y solo elimina caches si se ejecuta con `python clean_project.py --delete-temp`. No borra código ni modelos por defecto.
- Los siguientes archivos bajo `src/` siguen siendo plantillas vacías y deben documentarse o completarse antes de usarlos en producción: `mission_models.py`, `mission_models_fixed.py`, `partial_coverage.py`, `total_coverage.py`, `tensorflow_hybrid.py`, `trainers/multi_mission_trainer.py`, `trainers/partial_trainer.py`, `trainers/total_trainer.py`, `evaluators/model_evaluators.py`.
- Si necesitas remover artefactos temporales (por ejemplo `__pycache__` o `*.pyc`), usa la bandera `--delete-temp` en el script; cualquier otra limpieza debe hacerse manualmente revisando la lista que imprime el auditor.

## 📈 Roadmap Futuro

### 🎯 Mejoras Inmediatas

- [ ] Optimización de umbrales del Director
- [ ] Integración completa de sistemas TOI y K2
- [ ] Dashboard de monitoreo en tiempo real

### 🚀 Funcionalidades Avanzadas

- [ ] Modelos de deep learning más sofisticados
- [ ] Explicabilidad de predicciones (SHAP/LIME)
- [ ] API REST para integración externa
- [ ] Soporte para nuevas misiones espaciales

## 🤝 Contribuciones

Este proyecto está abierto a contribuciones. Areas de interés:

- 🔬 Nuevos algoritmos de ML
- 📊 Mejoras en visualización
- 🛰︝ Soporte para nuevas misiones
- 📚 Documentación y tutoriales

## 📄 Licencia

Este proyecto utiliza datos públicos de NASA/Kepler y está disponible para fines de investigación y educación.

## 🝅 Reconocimientos

- **NASA Kepler Mission** por los datos de alta calidad
- **TESS Mission** por los datos de TOI
- **Comunidad científica** por metodologías validadas

---

<div align="center">

**🌟 Director General Multi-Mission v1.0**
_Automatizando el descubrimiento de exoplanetas con IA_

[🚀 Demo](.) | [📖 Docs](.) | [🝛 Issues](.) | [💬 Discussions](.)

</div>
