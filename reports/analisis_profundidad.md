# Informe de revisión profunda

## Resumen ejecutivo
- El repositorio implementa parcialmente el árbol de decisión de tres niveles descrito: existe un `GeneralDirector` (nivel 1) y directores por misión (nivel 2). KOI y K2 ya cuentan con un selector dinámico que decide entre RandomForest y TensorFlow (nivel 3), mientras que TOI sigue usando el voto fijo original.
- Las fallas funcionales de la primera recomendación ya fueron atendidas: el demo `predict_with_director.py` procesa correctamente las respuestas del director general, las features ahora se alinean automáticamente con los scalers por misión y `imbalanced-learn` se declaró como dependencia.
- El sistema TOI ya cuenta con utilidades de datos, entrenamiento reproducible y un director con gating heurístico, por lo que las tres misiones pueden ejecutarse extremo a extremo.
- Existen múltiples archivos placeholders en `src/` y scripts de limpieza/requirements obsoletos, lo que revela deuda técnica alta y riesgo de borrado accidental de artefactos importantes.

## Actualización 04/12/2025
- `predict_with_director.py` desempaqueta las tuplas devueltas por `GeneralDirector.predict` y vuelve a reportar las métricas sin arrojar `TypeError`.
- `GeneralDirector` define explícitamente los conjuntos de columnas esperados por KOI/TOI/K2 y filtra/ordena las entradas antes de tocar los scalers, eliminando el `ValueError` por conteo de features.
- `requirements.txt` incluye `imbalanced-learn>=0.11.0`, por lo que el uso de `BorderlineSMOTE` en `k2_system/models/k2_randomforest.py` ya no rompe instalaciones limpias.
- `koi_system/core/director.py` y `k2_system/models/k2_director.py` implementan gating heurístico (selección KOI/K2 por muestra) validado con `python -m pytest tests/test_director_gating.py`.
- `k2_system/utils/data_utils.py` genera `k2_physical_target` por defecto y el runtime `k2_system/models/k2_ensemble.py` ahora solo carga los expertos entrenados para el director, reutilizado por `predict_k2.py` y `train_complete_k2.py`. La integración quedó cubierta por `python -m pytest`.
- `clean_project.py` dejó de borrar archivos productivos: actúa como auditor seguro y lista los placeholders reales en `src/`, alineado con la nueva sección de mantenimiento en `README.md`.

## Arquitectura actual vs. objetivo del árbol
1. **Nivel 1 – Director General (`src/models/general_director.py`)**: identifica la misión y reenvía el array de features al director especializado. No estandariza columnas ni valida esquemas, por lo que asume que el caller usó exactamente las mismas features y orden que durante el entrenamiento.
2. **Nivel 2 – Directores por misión (`koi_system/core/director.py`, `toi_system/core/director.py`, `k2_system/models/k2_director.py`)**: las tres misiones aplican heurísticas de complejidad/confianza para enrutar cada muestra a RF o TF, cubiertas por `tests/test_director_gating.py`.
3. **Nivel 3 – Modelos base (RandomForest + TensorFlow)**: KOI y K2 tienen implementaciones completas, pero TOI carece de clases reutilizables. Además, los modelos guardados no comparten un contrato homogéneo (algunos guardan metadata, otros no), lo que complica la orquestación.

## Hallazgos críticos (ordenados por severidad)

### 1. El demo principal falla inmediatamente
- **Archivo**: `predict_with_director.py`, función `demo_predictions`.
- **Problema**: `GeneralDirector.predict` devuelve `(predicciones, mission_used)` pero el script asume un array y hace `sum(predictions)` (líneas ~63, 85, 99). Resulta en `TypeError: unsupported operand type(s) for +: 'int' and 'tuple'`.
- **Impacto**: el showcase/documentación no pueden ejecutarse, dificultando la validación manual del árbol completo.
- **Acción**: desempaquetar correctamente (`preds, mission = director.predict(...)`) y adaptar los prints.
- **Estado**: ? Resuelto en `predict_with_director.py` (04/12/2025).

### 2. Incompatibilidad de features entre el Director General y los submodelos
- **Archivos**: `src/models/general_director.py` vs `koi_system/models/koi_randomforest.py`, `koi_system/models/koi_tensorflow.py`.
- **Problema**: el director pasa el DataFrame completo (`df_koi` con 60+ columnas) directo a `KOIDirector.predict`, pero los scalers se entrenaron con nueve columnas (`KOIConfig.FEATURES`). `StandardScaler.transform` lanzará `ValueError: X has 64 features, but StandardScaler is expecting 9`. Lo mismo aplica a TOI/K2.
- **Impacto**: cualquier llamada real a `GeneralDirector.predict` con datos crudos falla antes de llegar al árbol de modelos.
- **Acción**: definir pipelines de features por misión (orden consistente, selección de columnas, imputación) y aplicarlos dentro de `GeneralDirector` antes de delegar.
- **Estado**: ? Resuelto; `GeneralDirector` ya impone `feature_sets` por misión y valida columnas.

### 3. La capa intermedia no elige modelos, solo promedia *(Resuelto 04/12/2025)*
- **Archivos**: `koi_system/core/director.py`, `toi_system/core/director.py`, `k2_system/models/k2_director.py`.
- **Problema**: originalmente las tres clases realizaban un soft voting rígido (`ensemble_proba = 0.75 * rf + 0.25 * tf`). KOI y K2 ya sustituyeron ese esquema por un gating determinista basado en confianza/complexidad; TOI mantiene el comportamiento anterior.
- **Impacto**: antes el sistema no capitalizaba las fortalezas por modelo; con las mejoras, KOI/K2 ya reportan estadísticas de uso reales y se puede ampliar la lógica a TOI.
- **Acción**: mantener las pruebas `python -m pytest tests/test_director_gating.py` para vigilar el gating de KOI/TOI/K2.

### 4. `K2EnsembleSystem` es inejecutable *(Resuelto 04/12/2025)*
- **Archivos**: `k2_system/models/k2_ensemble.py` y `k2_system/models/k2_director.py`.
- **Problema**: la versión previa intentaba entrenar un director inexistente y fallaba con `AttributeError`.
- **Solución**: `k2_system/models/k2_ensemble.py` se reescribió como runtime ligero que solo carga `k2_rf_model.pkl`, `k2_tf_model.h5` y sus scalers para configurar el director de gating. `predict_k2.py` invoca este runtime y `train_complete_k2.py` produce exactamente esos artefactos usando `K2DataLoader`.

### 5. Sistema TOI incompleto *(Resuelto 04/12/2025)*
- **Archivos**: `toi_system/models/*.py`, `toi_system/utils/data_utils.py`, `toi_system/predict_toi.py` (todos vacíos) y `train_complete_toi.py`.
- **Problema**: anteriormente no existían utilidades ni modelos reutilizables y el etiquetado era sintético. Ahora `toi_system/utils` provee configuraciones y pipelines con labels reales (`tfopwg_disp`), `toi_system/models/train_models.py` genera los modelos persistentes y `toi_system/predict_toi.py` expone inferencia con el director actualizado.
- **Acción**: ejecutar `python -m pytest tests/test_toi_data_utils.py` y `python -m pytest tests/test_director_gating.py` para garantizar que el pipeline TOI se mantiene estable.

### 6. Target de K2 apunta a una columna inexistente *(Resuelto 04/12/2025)*
- **Archivo**: `k2_system/utils/data_utils.py`, método `prepare_target`.
- **Problema**: usaba `target_column='koi_disposition'`, inexistente en `k2_clean.csv`, y fallaba al preparar datos.
- **Solución**: ahora genera `k2_physical_target` mediante `build_physical_target` cuando la columna no existe y los tests `tests/test_k2_data_utils.py` cubren ese flujo.

### 7. Dependencia faltante para SMOTE
- **Archivo**: `k2_system/models/k2_randomforest.py` importa `from imblearn.over_sampling import BorderlineSMOTE`, pero `requirements.txt` no incluye `imbalanced-learn`.
- **Impacto**: al instalar el entorno se produce `ModuleNotFoundError` y el entrenamiento RF no puede correr.
- **Acción**: añadir `imbalanced-learn>=0.11` (o versión compatible) al requirements.
- **Estado**: ? Resuelto en `requirements.txt` (04/12/2025).

### 8. Identificación automática de misión puede devolver `Unknown` pero no se acepta
- **Archivo**: `src/models/general_director.py`, métodos `identify_mission` y `predict`.
- **Problema**: `identify_mission` retorna `'Unknown'` si no alcanza 50% de coincidencia; sin embargo `predict` valida `mission in ['KOI','TOI','K2']` y lanza `ValueError` cuando recibe `'Unknown'`.
- **Impacto**: entradas legítimas (p.ej. subconjuntos reducidos de features) causan fallos en lugar de caer en un modo degradado o requerir selección manual.
- **Acción**: manejar `'Unknown'` (p.ej. solicitar `mission` explícitamente o default a un pipeline genérico) y enriquecer los esquemas para minimizar falsos negativos.

### 9. Script de limpieza puede borrar archivos válidos *(Resuelto 04/12/2025)*
- **Archivo**: `clean_project.py`.
- **Problema**: enumeraba archivos inexistentes y podía borrar artefactos legítimos.
- **Solución**: se transformó en un auditor seguro que solo reporta placeholders y, opcionalmente, elimina caches (`--delete-temp`). El README documenta cómo usarlo sin riesgo.

### 10. Módulos base en `src/` son placeholders vacíos
- **Archivos**: `src/evaluators/*`, `src/models/mission_models*.py`, `src/trainers/*`, `src/utils/data_processor.py` (todos vacíos).
- **Impacto**: la capa compartida anunciada en README no existe; cualquier desarrollador que intente extender el árbol desde estos módulos encontrará archivos vacíos y sin contrato.
- **Acción**: completar o remover estos stubs para reducir deuda técnica y ajustar la documentación. Mientras tanto, `clean_project.py` y `README.md` listan explícitamente los archivos pendientes para evitar confusión.

## Mejores prácticas y mejoras adicionales
- **Normalizar persistencia**: Establecer un formato único de guardado (modelos, scalers, metadata) y versionarlo para que el `GeneralDirector` pueda validar compatibilidad antes de cargar.
- **Validaciones y pruebas**: Añadir pruebas unitarias para `identify_mission`, conversiones de features y selección de modelo; cubrir los casos KOI/TOI/K2 con datasets pequeños.
- **Documentar pipelines de datos**: Explicar claramente qué columnas y preprocesamiento usa cada misión para que terceros puedan preparar inputs compatibles.
- **Monitoreo y métricas de uso**: Una vez implementado el selector real, exponer métricas de uso y accuracy por modelo (RF vs TF) en un dashboard o log estructurado.

## Recomendaciones priorizadas
1. **(Completado 04/12/2025) Corregir fallos funcionales inmediatos**: el demo ya usa el retorno correcto, el `GeneralDirector` normaliza las columnas esperadas y `imbalanced-learn` figura en `requirements.txt`.
2. **(Completado 04/12/2025)** Implementar el selector de modelos en la capa intermedia para cumplir el diseño tipo árbol. KOI y K2 ahora enrutan cada muestra mediante heurísticas de complejidad/confianza y cuentan con pruebas `python -m pytest tests/test_director_gating.py` que validan el gating.
3. **(Completado 04/12/2025)** Completar el sistema TOI (data utils, modelos, predictor) usando etiquetas del catálogo `tfopwg_disp`. Nuevos módulos bajo `toi_system/utils` y `toi_system/models` alimentan el director con gating; validado mediante `python -m pytest tests/test_toi_data_utils.py tests/test_director_gating.py`.
4. **(Completado 04/12/2025)** Revisar utilitarios y configuraciones K2: `K2DataLoader` adopta `k2_physical_target` por defecto, `k2_system/models/k2_ensemble.py` es un runtime compatible con el director y `train_complete_k2.py` produce los artefactos que `predict_k2.py` consume. Validado con `python -m pytest`.
5. **(Completado 04/12/2025)** Limpieza de código: `clean_project.py` actúa como auditor seguro y `README.md` registra los placeholders existentes en `src/`, reduciendo el riesgo de borrar archivos activos o de asumir módulos inexistentes.
