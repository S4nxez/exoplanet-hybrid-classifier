# 🎼 Orkhestra - Sistema Híbrido de Clasificación de Exoplanetas

**Orquestando la inteligencia artificial para explorar nuevos mundos**

---

## 🌟 ¿Qué es Orkhestra?

**Orkhestra** es un sistema híbrido avanzado de machine learning que combina la **precisión ultra-alta** de RandomForest con la **cobertura completa** de TensorFlow para clasificar exoplanetas del catálogo Kepler con precisión excepcional.

### 🎯 Filosofía del Sistema

> *"Como una orquesta sinfónica, cada modelo toca su parte perfecta en el momento adecuado"*

- **🎻 Modelo Parcial (RandomForest)**: Solista de precisión para casos definidos
- **🎺 Modelo Total (TensorFlow)**: Base sólida que cubre todos los casos  
- **🎼 Fusión Inteligente**: Director que coordina ambos modelos en armonía

## 🚀 Arquitectura Orkhestra

### 📊 Rendimiento del Sistema

| Modelo | Accuracy | Precision | Coverage | Especialidad |
|--------|----------|-----------|----------|--------------|
| 🎯 **Parcial** | 86.20% | 99.4% | 41.1% | Ultra-preciso, selectivo |
| 🌐 **Total** | 85.57% | 89.0% | 100% | Cobertura completa |
| 🎼 **Orkhestra** | **85.57%** | **94.5%** | **100%** | **Mejor de ambos mundos** |

### 🧠 Algoritmo de Fusión

```python
def orkhestra_fusion(X):
    """
    🎼 Algoritmo de fusión inteligente
    """
    # 1. Evalúa confianza del modelo total
    total_pred, total_conf = total_model.predict_with_confidence(X)
    
    # 2. Para casos de ALTA confianza
    if total_conf > 0.95 and partial_model.can_predict(X):
        return partial_model.predict(X)  # 99.4% precision
    
    # 3. Para casos normales
    return total_pred  # 100% coverage
```

### 🎯 Características Clave

1. **Sistema de Cascada Inteligente**: 
   - Casos fáciles → Modelo parcial (99.4% precisión)
   - Casos complejos → Modelo total (100% cobertura)

2. **Umbrales Ultra-Estrictos**: 
   - Confianza mínima: 0.95
   - Margen de decisión: 0.3
   - Selectividad: Solo 41.1% de casos

3. **Cobertura Garantizada**: 
   - ✅ 100% de casos procesados
   - ✅ Sin predicciones rechazadas
   - ✅ Fusión automática basada en confianza

## 📁 Estructura del Proyecto

```
📦 Orkhestra/
├── 📊 data/
│   └── dataset.csv              # Dataset Kepler exoplanetas
├── 🧠 src/                      # Núcleo del sistema
│   ├── models/                  # Arquitecturas especializadas
│   │   ├── partial_coverage.py  # 🎯 Modelo ultra-preciso
│   │   ├── total_coverage.py    # 🌐 Modelo base completo  
│   │   └── tensorflow_hybrid.py # 🎼 Sistema Orkhestra
│   ├── trainers/                # Entrenadores especializados
│   │   ├── partial_trainer.py   # Entrenador modelo parcial
│   │   ├── total_trainer.py     # Entrenador modelo total
│   │   └── hybrid_trainer.py    # 🎼 Entrenador Orkhestra
│   ├── evaluators/              # Sistema de evaluación
│   └── utils/                   # Utilidades compartidas
├── 🎮 scripts/                  # Scripts de control
│   ├── main_training.py         # 🚀 Entrenamiento principal
│   ├── main_prediction.py       # 🔮 Predicción y comparación
│   └── project_info.py          # 📋 Información del proyecto
└── 💾 saved_models/             # Modelos entrenados
```

## 🛠️ Instalación y Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar Dataset

```bash
ls data/dataset.csv
```

### 3. Verificar Estructura

```bash
python scripts/project_info.py
```

## 🚀 Uso del Sistema Orkhestra

### 🎯 Entrenamiento Completo

```bash
# Entrenar sistema completo Orkhestra
python scripts/main_training.py --model all

# Entrenar componentes individuales
python scripts/main_training.py --model partial  # Ultra-preciso
python scripts/main_training.py --model total    # Cobertura completa
python scripts/main_training.py --model hybrid   # Sistema Orkhestra
```

### 🔮 Predicciones y Evaluación

```bash
# Comparar todos los modelos
python scripts/main_prediction.py --compare

# Evaluar modelo específico
python scripts/main_prediction.py --model orkhestra
```

### 🎼 Uso Programático de Orkhestra

```python
from src.models.tensorflow_hybrid import OrkhestraftHybridModel

# Inicializar sistema Orkhestra
orkhestra = OrkhestraftHybridModel(
    partial_threshold=0.95,    # Umbral ultra-estricto
    confidence_margin=0.3      # Margen de decisión
)

# Entrenar el sistema híbrido
orkhestra.train(X_train, y_train)

# Predicción con fusión inteligente
predictions, confidence = orkhestra.predict_with_confidence(X_test)

# Métricas detalladas del sistema
metrics = orkhestra.get_detailed_metrics(X_test, y_test)
print(f"Orkhestra Accuracy: {metrics['accuracy']:.2%}")
print(f"Fusion Ratio: {metrics['fusion_ratio']:.1%}")
```

## 🎯 Casos de Uso Optimizados

### 🔬 Para Investigación Científica
```python
# Configuración ultra-conservadora
orkhestra.set_threshold(0.98)  # Máxima precisión
orkhestra.set_margin(0.4)      # Máxima selectividad
```

### � Para Aplicaciones de Producción
```python
# Configuración balanceada 
orkhestra.set_threshold(0.90)  # Balance precisión-cobertura
orkhestra.auto_optimize(X_val, y_val)  # Auto-tuning
```

### 📊 Para Análisis Exploratorio
```python
# Configuración de máxima cobertura
orkhestra.set_threshold(0.85)  # Más casos al modelo parcial
orkhestra.enable_detailed_logging()  # Análisis detallado
```

## 📊 Resultados de Validación

### 🏆 Métricas Principales

```
🎼 ORKHESTRA SYSTEM VALIDATION
==============================
✅ Partial Model:  86.20% accuracy (41.1% coverage, 99.4% precision)
✅ Total Model:    85.57% accuracy (100% coverage, 89.0% precision)  
✅ Orkhestra:      85.57% accuracy (100% coverage, 94.5% precision)

🎯 FUSION ANALYSIS
==================
• Partial contribution: 41.1% of cases
• Total contribution:   58.9% of cases
• Fusion efficiency:    Seamless integration
• Confidence threshold: 0.95 (ultra-strict)
```

### 🎯 Distribución de Casos

- **🎻 Casos Parciales**: 41.1% - Ultra alta precisión (99.4%)
- **🎺 Casos Totales**: 58.9% - Cobertura garantizada (89.0%)
- **🎼 Fusión**: 100% - Sin casos rechazados

## 🔧 Configuración Avanzada

### Optimización de Umbrales

```python
# Encontrar umbral óptimo automáticamente
optimal_threshold = orkhestra.find_optimal_threshold(X_val, y_val)

# Configurar manualmente
orkhestra.configure(
    partial_threshold=0.95,
    confidence_margin=0.3,
    enable_fusion_logging=True
)
```

### Monitoreo del Sistema

```python
# Análisis detallado de fusión
fusion_stats = orkhestra.analyze_fusion_performance(X_test, y_test)

# Métricas por componente
component_metrics = orkhestra.get_component_metrics(X_test, y_test)
```

## �️ Arquitectura Técnica

### 🧠 Componentes del Sistema

1. **OrkhestraftHybridModel**: Núcleo de fusión inteligente
2. **PartialCoverageModel**: Especialista en casos definidos  
3. **TotalCoverageModel**: Base de cobertura completa
4. **HybridTrainer**: Entrenador coordinado del sistema

### 🔄 Flujo de Predicción

```
📥 Datos de entrada
    ↓
🧠 Modelo Total evalúa confianza
    ↓
🎯 ¿Confianza > 0.95? ┐
    ↓ SÍ              ↓ NO
🎻 Modelo Parcial     � Modelo Total
    ↓                 ↓
📤 Predicción final (fusión automática)
```

## 📋 Dependencias Técnicas

- **Python**: 3.13+
- **Scikit-learn**: RandomForest optimizado
- **TensorFlow**: Redes neuronales profundas
- **NumPy/Pandas**: Procesamiento de datos
- **Joblib**: Serialización de modelos

## 🎯 Ventajas de Orkhestra

✅ **Precisión Híbrida**: Combina lo mejor de ambos frameworks  
✅ **Cobertura Total**: 100% de casos procesados sin rechazo  
✅ **Fusión Inteligente**: Decisiones basadas en confianza  
✅ **Configuración Flexible**: Umbrales adaptables por caso de uso  
✅ **Métricas Detalladas**: Análisis completo de rendimiento  
✅ **Escalabilidad**: Arquitectura modular extensible  

## 🚀 Resultados de Benchmark

```
RENDIMIENTO COMPARATIVO
========================
Individual Models:
• RandomForest (Parcial): 86.20% acc, 41.1% cov, 99.4% prec
• TensorFlow (Total):      85.57% acc, 100% cov, 89.0% prec

Orkhestra System:
• Accuracy:    85.57% (mantiene base sólida)
• Precision:   94.5% (mejora +5.5% vs total)
• Coverage:    100% (cobertura garantizada) 
• Fusion:      Seamless (sin overhead)
```

---

**🎼 Orkhestra - Donde la precisión y la cobertura crean la sinfonía perfecta para la exploración de exoplanetas**

*Desarrollado para maximizar el potencial de clasificación de exoplanetas mediante inteligencia artificial híbrida.*

3. **Fusión de Probabilidades**: Combina predicciones ponderadas según confianza

4. **Métricas Comparativas**: Registra precisión y cobertura de cada componente- **Ideal para**: Aplicaciones generales, análisis exploratorio- **Ideal para**: Aplicaciones generales, análisis exploratorio



## 📁 Estructura del Proyecto



```### 🤖 Modelo Híbrido (Mejor de ambos mundos)### 🤖 Modelo Híbrido (Mejor de ambos mundos)

├── data/

│   └── dataset.csv              # Dataset de exoplanetas Kepler- **Precisión**: 85.8% con sistema de cascada inteligente- **Precisión**: 85.8% con sistema de cascada inteligente

├── src/                         # 🔧 Código fuente modular

│   ├── models/                  # Modelos especializados- **Sistema**: Cascada (30%) + Stacking (70%)- **Sistema**: Cascada (30%) + Stacking (70%)

│   │   ├── partial_coverage.py  # RandomForest ultra-preciso

│   │   ├── total_coverage.py    # Modelo base completo- **Ideal para**: Máximo rendimiento, aplicaciones de producción- **Ideal para**: Máximo rendimiento, aplicaciones de producción

│   │   └── tensorflow_hybrid.py # Sistema Orkhestra

│   ├── trainers/                # 🏗️ Entrenadores con herencia

│   │   ├── base_trainer.py      # Clase base abstracta

│   │   ├── partial_trainer.py   # Entrenador modelo parcial## 📁 Estructura del Proyecto Organizada## 📁 Estructura del Proyecto Organizada

│   │   ├── total_trainer.py     # Entrenador modelo total

│   │   └── hybrid_trainer.py    # Entrenador Orkhestra

│   ├── evaluators/              # 📊 Evaluadores especializados

│   │   ├── base_evaluator.py    # Clase base abstracta``````

│   │   └── model_evaluators.py  # Evaluadores específicos

│   └── utils/                   # 🛠️ Utilidades compartidas├── data/├── data/

│       └── data_processor.py    # Procesamiento de datos

├── scripts/                     # 📜 Scripts organizados│   └── dataset.csv              # Dataset de exoplanetas│   └── dataset.csv              # Dataset de exoplanetas

│   ├── training/                # Scripts de entrenamiento

│   │   ├── train_partial.py     # Entrenar modelo parcial├── src/                         # 🔧 Código fuente modular├── src/                         # 🔧 Código fuente modular

│   │   ├── train_total.py       # Entrenar modelo total

│   │   └── train_hybrid.py      # Entrenar Orkhestra│   ├── models/                  # Definiciones de modelos ML│   ├── models/                  # Definiciones de modelos ML

│   ├── prediction/              # Scripts de predicción

│   │   └── predict_models.py    # Predicciones unificadas│   │   ├── partial_coverage.py  # Modelo parcial│   │   ├── partial_coverage.py  # Modelo parcial

│   ├── utils/                   # Utilidades

│   │   └── clean_project.py     # Limpieza de proyecto│   │   ├── total_coverage.py    # Modelo total│   │   ├── total_coverage.py    # Modelo total

│   ├── main_training.py         # 🚀 Script principal de entrenamiento

│   ├── main_prediction.py       # 🔍 Script principal de predicción│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow

│   └── project_info.py          # ℹ️ Información del proyecto

├── saved_models/                # 💾 Modelos entrenados│   ├── trainers/                # 🏗️ Entrenadores con herencia│   ├── trainers/                # 🏗️ Entrenadores con herencia

├── train_*.py                   # 🔗 Wrappers de compatibilidad

├── predict_model.py             # 🔗 Wrapper de compatibilidad│   │   ├── base_trainer.py      # Clase base abstracta│   │   ├── base_trainer.py      # Clase base abstracta

└── clean_project.py             # 🔗 Wrapper de compatibilidad

```│   │   ├── total_trainer.py     # Entrenador modelo total│   │   ├── total_trainer.py     # Entrenador modelo total



## 🛠️ Instalación│   │   ├── partial_trainer.py   # Entrenador modelo parcial│   │   ├── partial_trainer.py   # Entrenador modelo parcial



```bash│   │   └── hybrid_trainer.py    # Entrenador modelo híbrido│   │   └── hybrid_trainer.py    # Entrenador modelo híbrido

# 1. Instalar dependencias

pip install -r requirements.txt│   ├── evaluators/              # 📊 Evaluadores especializados│   ├── evaluators/              # 📊 Evaluadores especializados



# 2. Verificar dataset│   │   ├── base_evaluator.py    # Clase base abstracta│   │   ├── base_evaluator.py    # Clase base abstracta

ls data/dataset.csv

```│   │   └── model_evaluators.py  # Evaluadores específicos│   │   └── model_evaluators.py  # Evaluadores específicos



## 🚀 Uso Rápido│   └── utils/                   # 🛠️ Utilidades compartidas│   └── utils/                   # 🛠️ Utilidades compartidas



### Scripts Principales (Recomendado)│       └── data_processor.py    # Procesamiento de datos│       └── data_processor.py    # Procesamiento de datos



```bash├── scripts/                     # 📜 Scripts organizados├── scripts/                     # 📜 Scripts organizados

# Entrenar el sistema completo Orkhestra

python scripts/main_training.py --model all│   ├── training/                # Scripts de entrenamiento│   ├── training/                # Scripts de entrenamiento



# Entrenar modelos individuales│   │   ├── train_total.py       # Entrenar modelo total│   │   ├── train_total.py       # Entrenar modelo total

python scripts/main_training.py --model partial  # RandomForest ultra-preciso

python scripts/main_training.py --model total    # TensorFlow completo│   │   ├── train_partial.py     # Entrenar modelo parcial│   │   ├── train_partial.py     # Entrenar modelo parcial

python scripts/main_training.py --model hybrid   # Sistema Orkhestra

│   │   └── train_hybrid.py      # Entrenar modelo híbrido│   │   └── train_hybrid.py      # Entrenar modelo híbrido

# Evaluar y comparar modelos

python scripts/main_prediction.py --compare│   ├── prediction/              # Scripts de predicción│   ├── prediction/              # Scripts de predicción



# Ver información del proyecto│   │   └── predict_models.py    # Predicciones con argumentos│   │   └── predict_models.py    # Predicciones con argumentos

python scripts/project_info.py

```│   ├── utils/                   # Utilidades│   ├── utils/                   # Utilidades



### Uso Avanzado de Orkhestra│   │   └── clean_project.py     # Limpieza de proyecto│   │   └── clean_project.py     # Limpieza de proyecto



```python│   ├── main_training.py         # 🚀 Script principal de entrenamiento│   ├── main_training.py         # 🚀 Script principal de entrenamiento

from src.models.tensorflow_hybrid import TensorFlowHybridModel

│   ├── main_prediction.py       # 🔍 Script principal de predicción│   ├── main_prediction.py       # 🔍 Script principal de predicción

# Crear sistema Orkhestra con umbral personalizado

orkhestra = TensorFlowHybridModel(│   └── project_info.py          # ℹ️ Información del proyecto│   └── project_info.py          # ℹ️ Información del proyecto

    partial_threshold=0.85,  # Umbral de confianza configurable

    enable_cascade=True      # Habilitar fusión inteligente├── saved_models/                # 💾 Modelos entrenados├── saved_models/                # 💾 Modelos entrenados

)

├── train_*.py                   # 🔗 Wrappers de compatibilidad├── train_*.py                   # 🔗 Wrappers de compatibilidad

# Entrenar el sistema híbrido

orkhestra.train(X_train, y_train)├── predict_model.py             # 🔗 Wrapper de compatibilidad├── predict_model.py             # 🔗 Wrapper de compatibilidad



# Predicción con fusión automática└── clean_project.py             # 🔗 Wrapper de compatibilidad└── clean_project.py             # 🔗 Wrapper de compatibilidad

predictions, probabilities = orkhestra.predict_with_confidence(X_test)

```

# Ajustar umbral dinámicamente

orkhestra.optimize_threshold(X_validation, y_validation)│   │   └── model_evaluators.py  # Evaluadores específicos



# Métricas comparativas## 🛠️ Instalación

metrics = orkhestra.get_comparative_metrics(X_test, y_test)

```│   └── utils/├── src/### 🎯 Modelo de Cobertura Binaria



## 🎯 Algoritmo de Fusión Orkhestra1. **Instalar dependencias:**



```python```bash│       └── data_processor.py    # Procesador de datos

def predict_with_fusion(self, X):

    """pip install -r requirements.txt

    Algoritmo principal de fusión Orkhestra

    """```├── scripts/                     # Scripts organizados│   ├── models/

    # 1. Evaluar confianza del modelo TensorFlow

    tf_predictions, tf_confidence = self.total_model.predict_with_confidence(X)

    

    # 2. Identificar casos de alta confianza2. **Verificar dataset:**│   ├── training/                # Scripts de entrenamiento

    high_confidence_mask = tf_confidence > self.partial_threshold

    ```bash

    # 3. Fusión inteligente

    final_predictions = []ls data/dataset.csv│   │   ├── train_total.py       # Entrenar modelo total│   │   ├── partial_coverage.py  # Modelo parcial- **Precisión**: 98.6% en decisiones definitivas

    for i, (pred, conf) in enumerate(zip(tf_predictions, tf_confidence)):

        if high_confidence_mask[i] and self.partial_model.can_predict(X[i]):```

            # Usar modelo parcial ultra-preciso

            final_predictions.append(self.partial_model.predict([X[i]])[0])│   │   ├── train_partial.py     # Entrenar modelo parcial

        else:

            # Usar modelo TensorFlow completo## 🚀 Uso Rápido

            final_predictions.append(pred)

    │   │   └── train_hybrid.py      # Entrenar modelo híbrido│   │   ├── total_coverage.py    # Modelo total- **Casos inciertos**: ~0% (cobertura completa)

    return final_predictions

```### Scripts Principales Unificados (Recomendado)



## 📊 Características del Sistema│   ├── prediction/              # Scripts de predicción



### 🎯 Modelo Parcial (RandomForest)```bash

- **Algoritmo**: RandomForestClassifier optimizado

- **Casos objetivo**: Extremos con señales claras# Entrenar todos los modelos│   │   └── predict_models.py    # Predictor universal│   │   └── tensorflow_hybrid.py # Modelo híbrido TensorFlow- **Clases**: Solo 2 - Exoplaneta vs No Exoplaneta

- **Ventajas**: Interpretabilidad, robustez, velocidad

- **Métricas**: Precision >99%, Coverage ~47%python scripts/main_training.py --model all



### 🌐 Modelo Total (TensorFlow)│   └── utils/                   # Utilidades

- **Arquitectura**: Red neuronal profunda

- **Casos objetivo**: Todos los casos del dataset# Entrenar modelo específico

- **Ventajas**: Flexibilidad, capacidad de generalización

- **Métricas**: Accuracy ~86%, Coverage 100%python scripts/main_training.py --model total│       └── clean_project.py     # Limpiador de proyecto│   └── utils/- **Ideal para**: Screening automático, decisiones simples



### 🎼 Sistema Orkhestrapython scripts/main_training.py --model partial  

- **Fusión**: Lógica de confianza dinámica

- **Configuración**: Umbral ajustable (0.1 - 0.99)python scripts/main_training.py --model hybrid├── saved_models/               # Modelos entrenados

- **Optimización**: Auto-tuning del umbral

- **Ventajas**: Mejor de ambos mundos



## 📈 Resultados de Rendimiento# Evaluar y comparar modelos├── train_partial_model.py      # Wrapper compatibilidad│       └── data_processor.py    # Procesador de datos- **Uso**: Clasificación binaria rápida, pipelines automatizados



| Modelo | Accuracy | Precision | Recall | F1-Score | Coverage |python scripts/main_prediction.py --compare

|--------|----------|-----------|--------|----------|----------|

| 🎯 Parcial | 85.83% | 99.2% | 74.1% | 0.749 | 47.26% |├── train_total_model.py        # Wrapper compatibilidad  

| 🌐 Total | 85.57% | 89.0% | 91.0% | 0.740 | 100% |

| 🎼 **Orkhestra** | **87.2%** | **94.5%** | **89.3%** | **0.762** | **100%** |# Ver información del proyecto



### Métricas Avanzadas de Orkhestrapython scripts/project_info.py├── train_hybrid_model.py       # Wrapper compatibilidad├── saved_models/               # Modelos entrenados

- **Casos de fusión**: ~30% usa modelo parcial, ~70% usa total

- **Mejora de precisión**: +1.6% sobre modelo base```

- **Confianza promedio**: 0.912 en decisiones finales

- **Tiempo de inferencia**: <50ms por predicción├── predict_model.py            # Wrapper compatibilidad



## 🔧 Configuración Avanzada### Scripts Individuales Organizados



### Ajuste de Umbral de Confianza└── clean_project.py            # Wrapper compatibilidad├── train_partial_model.py      # Entrenador modelo parcial## 🏗️ Estructura del Proyecto



```python```bash

# Configuración conservadora (más precision, menos coverage parcial)

orkhestra.set_threshold(0.95)# Entrenamiento individual```



# Configuración balanceada (equilibrio)python scripts/training/train_total.py

orkhestra.set_threshold(0.90)

python scripts/training/train_partial.py├── train_total_model.py        # Entrenador modelo total

# Configuración agresiva (más coverage parcial)

orkhestra.set_threshold(0.80)python scripts/training/train_hybrid.py



# Auto-optimización basada en validación## 🛠️ Instalación

optimal_threshold = orkhestra.find_optimal_threshold(X_val, y_val)

```# Predicción individual



### Métricas y Monitoreopython scripts/prediction/predict_models.py --model total├── train_hybrid_model.py       # Entrenador modelo híbrido```



```pythonpython scripts/prediction/predict_models.py --model partial

# Métricas detalladas del sistema

metrics = orkhestra.get_detailed_metrics(X_test, y_test)python scripts/prediction/predict_models.py --model hybrid1. Instalar dependencias:

print(f"Fusión ratio: {metrics['fusion_ratio']}")

print(f"Partial contribution: {metrics['partial_contribution']}")```

print(f"Total contribution: {metrics['total_contribution']}")

```bash└── predict_model.py            # Predictor universal├── dataset.csv                         # Dataset de Kepler

# Análisis de cobertura por confianza

coverage_analysis = orkhestra.analyze_coverage_by_confidence(X_test)### Wrappers de Compatibilidad (Retrocompatibilidad)

```

pip install -r requirements.txt

## 🎯 Casos de Uso

```bash

### Para Investigación Científica

- Usar umbral alto (0.95+) para máxima precisión# Mantienen la API original``````├── main_system.py                      # Sistema principal

- Priorizar predicciones del modelo parcial

- Análisis detallado de casos inciertospython train_total_model.py



### Para Aplicaciones de Producciónpython train_partial_model.py

- Usar umbral balanceado (0.90)

- Optimización automática del umbralpython train_hybrid_model.py

- Monitoreo continuo de métricas

python predict_model.py --model total2. Verificar que existe el dataset:├── max_precision_optimizer.py          # Modelo ultra-preciso

### Para Exploración de Datos

- Usar umbral bajo (0.80)```

- Máxima cobertura con buena precisión

- Análisis comparativo de modelos```bash



## 🔄 Flujo de Trabajo## 🏗️ Arquitectura del Sistema



1. **Entrenamiento**: `python scripts/main_training.py --model all`ls data/dataset.csv## 🛠️ Instalación├── train_coverage_model.py             # Entrenamiento cobertura completa

2. **Optimización**: Auto-tuning de umbrales en validación

3. **Evaluación**: `python scripts/main_prediction.py --compare`### Diseño Modular con Herencia

4. **Producción**: Deployment con configuración optimizada

```

## 🏆 Ventajas de Orkhestra

- **BaseTrainer**: Clase abstracta con pipeline común de entrenamiento

✅ **Inteligencia híbrida**: Combina fortalezas de diferentes frameworks  

✅ **Fusión adaptativa**: Umbral de confianza configurable y auto-optimizable  - **BaseEvaluator**: Clase abstracta para evaluación consistente  ├── clean_project.py                    # Script de limpieza

✅ **Mejor rendimiento**: Supera a modelos individuales en métricas clave  

✅ **Flexibilidad**: Configuración para diferentes escenarios de uso  - **Trainers específicos**: Heredan funcionalidad común, implementan lógica específica

✅ **Interpretabilidad**: Métricas detalladas de contribución de cada modelo  

✅ **Escalabilidad**: Arquitectura modular extensible  - **Separación de responsabilidades**: Entrenamiento vs. Evaluación vs. Scripts## 🎯 Entrenar Modelos

✅ **Compatibilidad**: Mantiene API existente del proyecto  



## 📋 Características Técnicas

### Patrón de Diseño1. Instalar dependencias:├── README.md                           # Esta documentación

- **Lenguajes**: Python 3.13+

- **ML Frameworks**: Scikit-learn + TensorFlow/Keras

- **Arquitectura**: Modular con herencia y composición

- **Patrones**: Strategy, Observer, Template Method```python### Opción 1: Scripts Organizados (Recomendado)

- **Optimización**: Auto-tuning de hiperparámetros

- **Métricas**: Logging detallado y comparativo# Ejemplo de uso de la arquitectura modular



---from src.trainers import TotalModelTrainer```bash├── requirements.txt                     # Dependencias



**🎼 Orkhestra - Orquestando la inteligencia artificial para la exploración espacial**

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
