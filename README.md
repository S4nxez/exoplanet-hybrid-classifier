# 🎼 Orkhestra - Sistema Híbrido de Clasificación de Exoplanetas

**Orquestando la inteligencia artificial para explorar nuevos mundos**

---

## 🌟 ¿Qué es Orkhestra?

**Orkhestra** es un sistema híbrido avanzado de machine learning que combina la **precisión ultra-alta** de RandomForest con la **cobertura completa** de TensorFlow para clasificar exoplanetas del catálogo Kepler.

### 🎯 Filosofía del Sistema

> *"Como una orquesta sinfónica, cada modelo toca su parte perfecta en el momento adecuado"*

- **🎻 Modelo Parcial (RandomForest)**: Solista de precisión para casos definidos
- **🎺 Modelo Total (TensorFlow)**: Base sólida que cubre todos los casos  
- **🎼 Fusión Inteligente**: Director que coordina ambos modelos en armonía

## 🚀 Rendimiento del Sistema

| Modelo | Accuracy | Precision | Coverage | Especialidad |
|--------|----------|-----------|----------|--------------|
| 🎯 **Parcial** | 86.20% | 99.4% | 41.1% | Ultra-preciso, selectivo |
| 🌐 **Total** | 85.57% | 89.0% | 100% | Cobertura completa |
| 🎼 **Orkhestra** | **85.57%** | **94.5%** | **100%** | **Mejor de ambos mundos** |

## 🧠 Algoritmo de Fusión

```python
def orkhestra_fusion(X):
    """🎼 Algoritmo de fusión inteligente"""
    # 1. Evalúa confianza del modelo total
    total_pred, total_conf = total_model.predict_with_confidence(X)
    
    # 2. Para casos de ALTA confianza (>0.95)
    if total_conf > 0.95 and partial_model.can_predict(X):
        return partial_model.predict(X)  # 99.4% precision
    
    # 3. Para casos normales
    return total_pred  # 100% coverage
```

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
│   ├── evaluators/              # Sistema de evaluación
│   └── utils/                   # Utilidades compartidas
├── 🎮 scripts/                  # Scripts de control
│   ├── main_training.py         # 🚀 Entrenamiento principal
│   ├── main_prediction.py       # 🔮 Predicción y comparación
│   └── project_info.py          # 📋 Información del proyecto
└── 💾 saved_models/             # Modelos entrenados
```

## 🛠️ Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar dataset
ls data/dataset.csv

# 3. Verificar estructura
python scripts/project_info.py
```

## 🚀 Uso del Sistema Orkhestra

### 🎯 Entrenamiento

```bash
# Entrenar sistema completo Orkhestra
python scripts/main_training.py --model all

# Entrenar componentes individuales
python scripts/main_training.py --model partial  # Ultra-preciso
python scripts/main_training.py --model total    # Cobertura completa
python scripts/main_training.py --model hybrid   # Sistema Orkhestra
```

### 🔮 Evaluación

```bash
# Comparar todos los modelos
python scripts/main_prediction.py --compare

# Evaluar modelo específico
python scripts/main_prediction.py --model orkhestra
```

### 🎼 Uso Programático

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
```

## 🎯 Casos de Uso

### 🔬 Para Investigación Científica
```python
orkhestra.set_threshold(0.98)  # Máxima precisión
```

### 🏭 Para Aplicaciones de Producción
```python
orkhestra.set_threshold(0.90)  # Balance precisión-cobertura
orkhestra.auto_optimize(X_val, y_val)  # Auto-tuning
```

### 📊 Para Análisis Exploratorio
```python
orkhestra.set_threshold(0.85)  # Más casos al modelo parcial
```

## 📊 Resultados de Validación

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
• Confidence threshold: 0.95 (ultra-strict)
```

## 🔧 Arquitectura Técnica

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
🎻 Modelo Parcial     🎺 Modelo Total
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