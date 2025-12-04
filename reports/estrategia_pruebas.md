# Estrategia Integral de Pruebas

## 1. Propósito y Alcance
- Garantizar que los tres sistemas de misión (`koi_system`, `k2_system`, `toi_system`) y el stack compartido en `src/` mantengan calidad consistente al evolucionar.
- Cubrir desde utilitarios de datos hasta directores híbridos, entrenadores y CLI de predicción, incluyendo scripts auxiliares (`clean_project.py`, `train_all_systems.py`).
- Establecer un roadmap de automatización que permita detectar regresiones científicas (métrica/modelo) y regresiones de ingeniería (API, CLI, datos).

## 2. Principios Rectores
1. **Pirámide balanceada**: priorizar pruebas unitarias rápidas, respaldadas por integraciones selectivas y ejecuciones end-to-end (E2E) ligeras.
2. **Datos deterministas**: fixtures sintéticos controlados (por misión) + snapshots reducidos de `data/clean/*.csv` para validar pipelines reales.
3. **Paridad entre misiones**: cualquier regla añadida a KOI/K2/TOI debe tener pruebas homólogas para impedir divergencias silenciosas.
4. **Documentar supuestos**: cada suite debe describir en fixtures o helpers las hipótesis físicas/estadísticas para facilitar mantenimiento.
5. **Automatizar evidencia**: integración continua obligatoria (pytest + coverage + informes de métricas modelo).

## 3. Inventario de Superficies Críticas
| Dominio | Componentes clave | Riesgos detectados | Tipo de prueba prioritario |
| --- | --- | --- | --- |
| Ingesta de datos | `*_system/utils/data_utils.py`, `src/utils/data_processor.py` | columnas faltantes, imputación inconsistente | Unit (validación de columnas), property-based con `hypothesis` |
| Modelos expertos | `koi_randomforest.py`, `k2_tensorflow.py`, `toi_system/models/train_models.py` | drift de hiperparámetros, serialización rota | Unit + smoke de entrenamiento (marca `slow`)
| Director híbrido | `koi_system/core/director.py`, `k2_system/models/k2_director.py`, `toi_system/core/director.py` | rutas mal asignadas, umbrales sin cobertura | Unit (gating) + contract tests (mismo dataset)
| Orquestadores | `train_all_systems.py`, `*_system/train_*` | pipelines cortados, rutas de salida erróneas | Integration con fakes + CLI tests
| Predictores CLI | `predict_k2.py`, `predict_koi.py`, `predict_toi.py` | argumentos inválidos, modelos faltantes | CLI/E2E con `pytest` + `typer.CliRunner`
| Scripts de limpieza | `clean_project.py` | borrado destructivo | Unit + snapshot FS (pyfakefs)
| Reportes/analítica | `reports/*.md` generados desde código | divergencias con realidad | Lint + comparación de checkpoints

## 4. Niveles de Prueba y Cobertura Objetivo
1. **Unitarias (70% esfuerzo)**
   - Meta: ?85% de cobertura en módulos utilitarios y directores.
   - Herramientas: `pytest`, `pytest-mock`, `hypothesis` para inputs tabulares.
2. **Componentes / Integración (20%)**
   - Validar interacción entre data loaders, scaler, modelos y serialización.
   - Ejecutar con subconjuntos reales (`nrows<=2000`) para KOI/K2/TOI.
3. **End-to-End (10%)**
   - Scripts `train_complete_*` y `predict_*` usando recursos mockeados.
   - Se marcan `@pytest.mark.slow` y se ejecutan nocturnamente.

## 5. Estrategias Específicas por Dominio
### 5.1 Utilitarios de Datos (KOI/K2/TOI)
- **Cobertura mínima**: parsing de CSV/PKL, limpieza (`clean_data`), creación de features derivadas, imputación.
- **Casos negativos**: columnas faltantes, valores fuera de rango, tipos erróneos.
- **Ejemplo existente**: `tests/test_toi_data_utils.py` valida la construcción de labels; replicar patrón para KOI/K2 usando `build_physical_target`.

### 5.2 Ingeniería de Características Compartida (`src/utils/data_processor.py`)
- Validar pipelines multi-misión (parámetros `mission_name`).
- Property tests para aserciones de escala (mean?0 tras `StandardScaler`).

### 5.3 Directores y Gating
- Extender `tests/test_director_gating.py` con datasets específicos por misión.
- Contract tests: mismos vectores de entrada deben producir ruteos coherentes entre misiones cuando las condiciones son equivalentes.
- Añadir fixtures para umbrales (`confidence_threshold`) y escenarios borde (probabilidad = umbral).

### 5.4 Entrenadores y Modelos
- **RandomForest**: smoke tests que verifiquen `feature_importances_`, balanceo y serialización (`joblib`).
- **TensorFlow**: usar `tf.keras.backend.clear_session()` en fixtures; validar que checkpoints generados existen y contienen firmas esperadas.
- **Director (NN)**: tests de entrenamiento corto (5 epochs) con dataset en memoria para asegurar convergencia básica.

### 5.5 Pipelines de Entrenamiento
- Simular ejecuciones de `train_complete_*` con `TemporaryDirectory` para salidas.
- Mockear escritura de modelos pesados para acelerar.
- Verificar logs y métricas mínimas (>0.5 accuracy) para detectar regresiones graves.

### 5.6 CLI / Predicción
- Usar `CliRunner` o `subprocess.run` en modo sandbox.
- Cubrir: argumentos obligatorios, archivos inexistentes, carga de modelos corruptos, modo batch.
- Validar JSON/CSV output schema.

### 5.7 Scripts Auxiliares y Limpieza
- `clean_project.py`: reproducir hallazgos de recomendación 5; crear pruebas con `pyfakefs` para asegurar que sólo elimine rutas permitidas.
- `Makefile` targets: tests que verifiquen comandos críticos (`make train_all`).

### 5.8 Reportes y Artefactos Persistentes
- Tests que garanticen que `reports/analisis_profundidad.md` y `reports/estrategia_pruebas.md` se actualizan cuando cambian recomendaciones (usar snapshots de texto + hooks).

## 6. Datos, Fixtures y Herramientas de Soporte
- **Fixtures reales reducidos**: scripts para crear `data/samples/{koi,k2,toi}_mini.csv` (?1 MB) referenciados por todas las suites.
- **Generadores sintéticos**: helpers compartidos en `tests/fixtures/factories.py` para crear filas con combinaciones físicas válidas.
- **Pytest plugins**: `conftest.py` central con fixtures para paths, configuraciones (`K2Config`, etc.), y clientes CLI.

## 7. Automatización y Pipeline CI
1. **Pre-commit**: ejecutar `pytest -m "not slow"` y `ruff`/`black` (si aplica).
2. **CI por push/PR**:
   - Job 1: lint + unit tests (Linux, Python 3.10/3.11).
   - Job 2: integration + slow tests (cron nocturno o workflow manual) con GPU opcional para TensorFlow.
   - Publicar cobertura con `coverage.py` (umbral 80%) y subir a artefactos.
3. **Notebooks/Reports**: job que valide que archivos `.md` se mantienen sincronizados (por ejemplo, verificar enlaces.

## 8. Métricas de Éxito
- Cobertura por módulo (`pytest --cov=k2_system --cov=koi_system --cov=toi_system --cov=src`).
- Tiempo máximo suite rápida: <6 minutos.
- Regressions detectadas antes de merge: ?95%.
- Zero tolerancia a flakes (tests inestables); registrar en tablero.

## 9. Roadmap de Implementación
1. **Semana 1**: normalizar fixtures y completar cobertura de utilitarios KOI/K2/TOI.
2. **Semana 2**: suites de directores + smoke de entrenamiento.
3. **Semana 3**: CLI/E2E, scripts auxiliares y documentación viva.
4. **Semana 4**: integrar CI completa, ajustar umbrales de cobertura y preparar reporte de métricas.

## 10. Riesgos y Mitigaciones
- **Dependencia de TensorFlow**: usar marcadores `slow` y contenedores con GPU opcional.
- **Datos pesados**: generar subconjuntos deterministas y cachearlos en artefactos CI.
- **Inestabilidad numérica**: fijar semillas (`K2Config.RANDOM_SEED`, etc.) en todos los fixtures.
- **Scripts destructivos**: aislar en pruebas con sistemas de archivos falsos + validaciones previas.

---
Esta estrategia debe revisarse trimestralmente junto con `reports/analisis_profundidad.md` para asegurar que la cobertura de pruebas evoluciona al ritmo de las recomendaciones del proyecto.
