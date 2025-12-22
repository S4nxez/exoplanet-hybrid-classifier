# Informe de revisi�n profunda

## Resumen ejecutivo
- El repositorio implementa parcialmente el �rbol de decisi�n de tres niveles descrito: existe un `GeneralDirector` (nivel 1) y directores por misi�n (nivel 2). KOI y K2 ya cuentan con un selector din�mico que decide entre RandomForest y TensorFlow (nivel 3), mientras que TOI sigue usando el voto fijo original.
- Las fallas funcionales de la primera recomendaci�n ya fueron atendidas: el demo `predict_with_director.py` procesa correctamente las respuestas del director general, las features ahora se alinean autom�ticamente con los scalers por misi�n y `imbalanced-learn` se declar� como dependencia.
- El sistema TOI ya cuenta con utilidades de datos, entrenamiento reproducible y un director con gating heur�stico, por lo que las tres misiones pueden ejecutarse extremo a extremo.
- Existen m�ltiples archivos placeholders en `src/` y scripts de limpieza/requirements obsoletos, lo que revela deuda t�cnica alta y riesgo de borrado accidental de artefactos importantes.

## Actualización 04/12/2025 - ESTADO REAL
**CRÍTICO: Verificación completa realizada - múltiples problemas persistentes**

### ❌ PROBLEMAS NO RESUELTOS:
1. **TOI System**: Rendimiento extremadamente bajo (31.86% RF, 16.81% TF, 16.94% Director)
   - El target parece estar MAL ETIQUETADO
   - 83.2% de positivos en el conjunto de prueba indica desbalance extremo
   - Director NO mejora sobre modelos individuales (-14.93%)

2. **K2 System**: ERROR CRÍTICO de incompatibilidad de features
   - Scalers esperan 8 features pero se envían 14 features
   - Los modelos entrenados NO coinciden con la configuración actual
   - Sistema completamente NO FUNCIONAL para evaluación

3. **KOI System**: Target sintético defectuoso
   - 0% de positivos en test set (distribución completamente inválida)
   - Métricas artificialmente altas pero sin significado real
   - Director ligeramente PEOR que TensorFlow individual

### ⚠️ PROBLEMAS ESTRUCTURALES:
- **Imports rotos**: Los sistemas no pueden importar sus propios módulos (ModuleNotFoundError: 'koi_system', 'toi_system', 'k2_system')
- **train_all_systems.py**: FALLA completamente - ningún sistema entrena exitosamente
- **Incompatibilidad de configuraciones**: Las features en código NO coinciden con modelos guardados
- **Targets sintéticos mal definidos**: KOI genera 0% positivos, TOI tiene distribución sospechosa

### ❌ FUNCIONALIDAD DIRECTOR:
- KOIDirector NO tiene método `load_models()` - AttributeError
- TOIDirector usa soft voting pero con métricas inválidas
- K2Director no puede cargar por incompatibilidad de features

### ✓ Lo que SÍ funciona parcialmente:
- `requirements.txt` incluye `imbalanced-learn>=0.11.0`
- `clean_project.py` actúa como auditor seguro
- Estructura de archivos está organizada
- Modelos .h5 y .pkl existen físicamente

## Arquitectura actual vs. objetivo del �rbol
1. **Nivel 1 � Director General (`src/models/general_director.py`)**: identifica la misi�n y reenv�a el array de features al director especializado. No estandariza columnas ni valida esquemas, por lo que asume que el caller us� exactamente las mismas features y orden que durante el entrenamiento.
2. **Nivel 2 � Directores por misi�n (`koi_system/core/director.py`, `toi_system/core/director.py`, `k2_system/models/k2_director.py`)**: las tres misiones aplican heur�sticas de complejidad/confianza para enrutar cada muestra a RF o TF, cubiertas por `tests/test_director_gating.py`.
3. **Nivel 3 � Modelos base (RandomForest + TensorFlow)**: KOI y K2 tienen implementaciones completas, pero TOI carece de clases reutilizables. Adem�s, los modelos guardados no comparten un contrato homog�neo (algunos guardan metadata, otros no), lo que complica la orquestaci�n.

## Hallazgos cr�ticos (ordenados por severidad)

### 1. El demo principal falla inmediatamente
- **Archivo**: `predict_with_director.py`, funci�n `demo_predictions`.
- **Problema**: `GeneralDirector.predict` devuelve `(predicciones, mission_used)` pero el script asume un array y hace `sum(predictions)` (l�neas ~63, 85, 99). Resulta en `TypeError: unsupported operand type(s) for +: 'int' and 'tuple'`.
- **Impacto**: el showcase/documentaci�n no pueden ejecutarse, dificultando la validaci�n manual del �rbol completo.
- **Acci�n**: desempaquetar correctamente (`preds, mission = director.predict(...)`) y adaptar los prints.
- **Estado**: ? Resuelto en `predict_with_director.py` (04/12/2025).

### 2. Incompatibilidad de features entre el Director General y los submodelos
- **Archivos**: `src/models/general_director.py` vs `koi_system/models/koi_randomforest.py`, `koi_system/models/koi_tensorflow.py`.
- **Problema**: el director pasa el DataFrame completo (`df_koi` con 60+ columnas) directo a `KOIDirector.predict`, pero los scalers se entrenaron con nueve columnas (`KOIConfig.FEATURES`). `StandardScaler.transform` lanzar� `ValueError: X has 64 features, but StandardScaler is expecting 9`. Lo mismo aplica a TOI/K2.
- **Impacto**: cualquier llamada real a `GeneralDirector.predict` con datos crudos falla antes de llegar al �rbol de modelos.
- **Acci�n**: definir pipelines de features por misi�n (orden consistente, selecci�n de columnas, imputaci�n) y aplicarlos dentro de `GeneralDirector` antes de delegar.
- **Estado**: ? Resuelto; `GeneralDirector` ya impone `feature_sets` por misi�n y valida columnas.

### 3. La capa intermedia no elige modelos, solo promedia *(Resuelto 04/12/2025)*
- **Archivos**: `koi_system/core/director.py`, `toi_system/core/director.py`, `k2_system/models/k2_director.py`.
- **Problema**: originalmente las tres clases realizaban un soft voting r�gido (`ensemble_proba = 0.75 * rf + 0.25 * tf`). KOI y K2 ya sustituyeron ese esquema por un gating determinista basado en confianza/complexidad; TOI mantiene el comportamiento anterior.
- **Impacto**: antes el sistema no capitalizaba las fortalezas por modelo; con las mejoras, KOI/K2 ya reportan estad�sticas de uso reales y se puede ampliar la l�gica a TOI.
- **Acci�n**: mantener las pruebas `python -m pytest tests/test_director_gating.py` para vigilar el gating de KOI/TOI/K2.

### 4. `K2EnsembleSystem` es inejecutable *(Resuelto 04/12/2025)*
- **Archivos**: `k2_system/models/k2_ensemble.py` y `k2_system/models/k2_director.py`.
- **Problema**: la versi�n previa intentaba entrenar un director inexistente y fallaba con `AttributeError`.
- **Soluci�n**: `k2_system/models/k2_ensemble.py` se reescribi� como runtime ligero que solo carga `k2_rf_model.pkl`, `k2_tf_model.h5` y sus scalers para configurar el director de gating. `predict_k2.py` invoca este runtime y `train_complete_k2.py` produce exactamente esos artefactos usando `K2DataLoader`.

### 5. Sistema TOI incompleto *(Resuelto 04/12/2025)*
- **Archivos**: `toi_system/models/*.py`, `toi_system/utils/data_utils.py`, `toi_system/predict_toi.py` (todos vac�os) y `train_complete_toi.py`.
- **Problema**: anteriormente no exist�an utilidades ni modelos reutilizables y el etiquetado era sint�tico. Ahora `toi_system/utils` provee configuraciones y pipelines con labels reales (`tfopwg_disp`), `toi_system/models/train_models.py` genera los modelos persistentes y `toi_system/predict_toi.py` expone inferencia con el director actualizado.
- **Acci�n**: ejecutar `python -m pytest tests/test_toi_data_utils.py` y `python -m pytest tests/test_director_gating.py` para garantizar que el pipeline TOI se mantiene estable.

### 6. Target de K2 apunta a una columna inexistente *(Resuelto 04/12/2025)*
- **Archivo**: `k2_system/utils/data_utils.py`, m�todo `prepare_target`.
- **Problema**: usaba `target_column='koi_disposition'`, inexistente en `k2_clean.csv`, y fallaba al preparar datos.
- **Soluci�n**: ahora genera `k2_physical_target` mediante `build_physical_target` cuando la columna no existe y los tests `tests/test_k2_data_utils.py` cubren ese flujo.

### 7. Dependencia faltante para SMOTE
- **Archivo**: `k2_system/models/k2_randomforest.py` importa `from imblearn.over_sampling import BorderlineSMOTE`, pero `requirements.txt` no incluye `imbalanced-learn`.
- **Impacto**: al instalar el entorno se produce `ModuleNotFoundError` y el entrenamiento RF no puede correr.
- **Acci�n**: a�adir `imbalanced-learn>=0.11` (o versi�n compatible) al requirements.
- **Estado**: ? Resuelto en `requirements.txt` (04/12/2025).

### 8. Identificaci�n autom�tica de misi�n puede devolver `Unknown` pero no se acepta
- **Archivo**: `src/models/general_director.py`, m�todos `identify_mission` y `predict`.
- **Problema**: `identify_mission` retorna `'Unknown'` si no alcanza 50% de coincidencia; sin embargo `predict` valida `mission in ['KOI','TOI','K2']` y lanza `ValueError` cuando recibe `'Unknown'`.
- **Impacto**: entradas leg�timas (p.ej. subconjuntos reducidos de features) causan fallos en lugar de caer en un modo degradado o requerir selecci�n manual.
- **Acci�n**: manejar `'Unknown'` (p.ej. solicitar `mission` expl�citamente o default a un pipeline gen�rico) y enriquecer los esquemas para minimizar falsos negativos.

### 9. Script de limpieza puede borrar archivos v�lidos *(Resuelto 04/12/2025)*
- **Archivo**: `clean_project.py`.
- **Problema**: enumeraba archivos inexistentes y pod�a borrar artefactos leg�timos.
- **Soluci�n**: se transform� en un auditor seguro que solo reporta placeholders y, opcionalmente, elimina caches (`--delete-temp`). El README documenta c�mo usarlo sin riesgo.

### 10. M�dulos base en `src/` son placeholders vac�os
- **Archivos**: `src/evaluators/*`, `src/models/mission_models*.py`, `src/trainers/*`, `src/utils/data_processor.py` (todos vac�os).
- **Impacto**: la capa compartida anunciada en README no existe; cualquier desarrollador que intente extender el �rbol desde estos m�dulos encontrar� archivos vac�os y sin contrato.
- **Acci�n**: completar o remover estos stubs para reducir deuda t�cnica y ajustar la documentaci�n. Mientras tanto, `clean_project.py` y `README.md` listan expl�citamente los archivos pendientes para evitar confusi�n.

## Mejores pr�cticas y mejoras adicionales
- **Normalizar persistencia**: Establecer un formato �nico de guardado (modelos, scalers, metadata) y versionarlo para que el `GeneralDirector` pueda validar compatibilidad antes de cargar.
- **Validaciones y pruebas**: A�adir pruebas unitarias para `identify_mission`, conversiones de features y selecci�n de modelo; cubrir los casos KOI/TOI/K2 con datasets peque�os.
- **Documentar pipelines de datos**: Explicar claramente qu� columnas y preprocesamiento usa cada misi�n para que terceros puedan preparar inputs compatibles.
- **Monitoreo y m�tricas de uso**: Una vez implementado el selector real, exponer m�tricas de uso y accuracy por modelo (RF vs TF) en un dashboard o log estructurado.

## Recomendaciones priorizadas (ACTUALIZADO 04/12/2025)
1. **🔴 URGENTE - Reentrenar todos los sistemas correctamente**:
   - Definir targets reales (no sintéticos) para KOI y K2
   - Corregir etiquetado de TOI para mejorar balance y rendimiento
   - Asegurar que features en código coincidan con modelos guardados
   - Validar que train/test split genere distribuciones válidas

2. **🔴 URGENTE - Arreglar sistema de imports**:
   - Convertir koi_system, toi_system, k2_system en paquetes instalables
   - O usar rutas relativas consistentes
   - `train_all_systems.py` debe poder ejecutar todos los entrenamientos

3. **🟡 CRÍTICO - Implementar método `load_models()` faltante**:
   - KOIDirector necesita `load_models()` para cargar RF y TF
   - Estandarizar interfaz de carga en todos los directores
   - Validar compatibilidad de features al cargar

4. **🟡 CRÍTICO - Validar y corregir configuraciones**:
   - K2: Scalers esperan 8 features, código envía 14
   - TOI: 83% positivos sugiere problema de etiquetado
   - KOI: 0% positivos indica target completamente roto

5. **🟢 MEJORABLE - Completar funcionalidad Director**:
   - Implementar gating real (no solo soft voting) en TOI
   - Validar que directores mejoren sobre modelos individuales
   - Agregar métricas de uso de modelos (% RF vs TF)

**ESTADO ACTUAL: ❌ SISTEMA NO PRODUCTIVO - REQUIERE REENTRENAMIENTO COMPLETO**
