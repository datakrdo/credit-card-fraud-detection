🇬🇧 [English](README.md) · 🇪🇸 [Español](README_es.md)

# Detección de fraude — transacciones con tarjeta de crédito

Detección de fraude sobre el dataset sintético Sparkov de transacciones con tarjeta de
crédito: ~1,85 millones de transacciones de 983 tarjetas entre enero de 2019 y diciembre
de 2020, con prevalencia de fraude de 0,58 % en el período de entrenamiento y 0,39 % en el
holdout.

Dos entregables:

- **[`fraud-portfolio_es.ipynb`](fraud-portfolio_es.ipynb)** — notebook autocontenida
  (no importa el paquete `fraud`), narrativa en español, ejecutada de punta a punta. Es el recorrido
  completo: problema, EDA, feature engineering causal, escalera de modelos, la decisión de
  cómo tratar el desbalance, calibración, y la sección de umbral y dinero.
- **`src/fraud/`** — paquete productivo instalable con `uv`: carga y split temporal,
  feature engineering, modelos GPU, evaluación, umbral, orquestador y CLI de entrenamiento
  y scoring batch.

## Decisiones de diseño 🧭

- **PR-AUC contra prevalencia, nunca accuracy.** Un clasificador constante que nunca marca
  fraude saca 99,4 % de accuracy en este dataset. `src/fraud/evaluation.py::binary_metrics`
  reporta PR-AUC y su lift sobre la prevalencia real (0,58 % train / 0,39 % holdout), no
  contra el 0,5 de un ranking aleatorio.
- **Split temporal, no aleatorio.** El modelo puntúa transacciones futuras con datos del
  pasado; partir al azar mezclaría transacciones futuras de una tarjeta en el conjunto de
  entrenamiento de esa misma tarjeta. `src/fraud/data.py::temporal_split` corta por fecha, y
  `fraudTest.csv` se trata como el holdout natural del dataset, intocado hasta la evaluación
  final.
- **Features de velocidad causales.** La hora 22–23 h muestra un lift de 5× sobre la
  prevalencia base, y la mediana de segundos desde la transacción previa de la tarjeta es
  4.908 en fraude contra 16.623 en legítimas — la ráfaga es señal real. Toda esa familia de
  features (`src/fraud/features.py`) se calcula con
  `groupby("cc_num").rolling(window, on=..., closed="left")`, que excluye la fila actual: un
  agregado que se filtra a sí mismo infla el PR-AUC de validación sin que el modelo haya
  aprendido nada generalizable.
- **`distance_km` se construye y se deja fuera del modelo.** `merch_lat`/`merch_long` en este dataset se
  generan de forma uniforme alrededor del domicilio del titular, así que la distancia
  titular↔comercio no discrimina (≈76 km en ambas clases). Una ablación sobre el modelo
  final lo confirma: PR-AUC de validación 0,9770 sin ella, 0,9763 con ella. Se sigue
  calculando en `build_features` para que el resultado negativo sea reproducible, pero no
  es entrada del modelo.
- **Ponderación de la pérdida, no SMOTE.** `scale_pos_weight` (boosteados) /
  `class_weight="balanced"` (LogReg) en vez de resamplear. Motivos: (1) 9.651 fraudes
  absolutos no es un problema de escasez de etiquetas; (2) SMOTE interpola en un espacio con
  categóricas de alta cardinalidad (`merchant`, 693 niveles) donde la distancia euclídea no
  significa nada; (3) ponderar no toca los datos, por lo que no puede filtrar validación;
  (4) resamplear distorsiona las probabilidades, y este proyecto necesita probabilidades
  calibradas para el cálculo de ahorro en dólares. La notebook mide, no sólo afirma: compara
  PR-AUC de validación entre no-tratar / ponderar / SMOTENC / undersampling antes de fijar la
  decisión.
- **Umbral por ahorro neto en dólares, congelado en validación.** `src/fraud/threshold.py`
  maximiza `Σ amt[TP] − FP_COST·|FP| − Σ amt[FN]` sobre la validación, nunca sobre el
  holdout — elegirlo ahí sería la misma fuga que tunear cualquier hiperparámetro contra el
  conjunto de evaluación final.

## Herramientas 🛠️

- Python, pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `IsotonicRegression`, `DummyClassifier`)
- XGBoost y LightGBM sobre GPU (`device="cuda"` / `"gpu"`)
- imbalanced-learn (`SMOTENC`, `RandomUnderSampler`, dentro de `imblearn.Pipeline`)
- SHAP
- matplotlib, seaborn
- uv, ruff, pytest, GitHub Actions

## Habilidades 🧠

- Ingeniería de features causal (rolling windows y agregados sin fuga temporal)
- Clasificación con desbalance extremo: PR-AUC, calibración de probabilidades, costo-beneficio
- Validación temporal para series de eventos (split por fecha, holdout de producción)
- Optimización de umbral de decisión por valor esperado en dólares
- Explicabilidad de modelos (SHAP) y comunicación de resultados a una audiencia no técnica
- Empaquetado de un proyecto de ML como paquete instalable con CLI, tests y CI

## Estructura

- `src/fraud/data.py` — carga, valida esquema, castea `cc_num` a string, ordena
  causalmente, y corta `fraudTrain.csv` en train/validación por fecha (`TemporalSplit`).
- `src/fraud/features.py` — `build_features`, pura y sin estado: features temporales,
  monto, velocidad causal por tarjeta, novedad de comercio/categoría, y `distance_km`
  (calculada, no es entrada del modelo).
- `src/fraud/modeling.py` — registro `MODEL_SPECS` (`dummy`, `logreg`, `lightgbm`,
  `xgboost`), con los dos últimos usando GPU (`device="cuda"`/`"gpu"`) y degradación
  elegante si los extras `[gpu]` no están instalados.
- `src/fraud/evaluation.py` — PR-AUC contra prevalencia, precision@k, matriz de confusión.
- `src/fraud/threshold.py` — barrido de umbral vectorizado, ahorro neto, análisis marginal
  y sensibilidad a `FP_COST`.
- `src/fraud/pipeline.py` — orquestador único: split → features → fit → calibración
  isotónica sobre validación → umbral congelado. `FraudModel` empaqueta las tres cosas
  juntas, porque un umbral sin su modelo no es una decisión reproducible.
- `src/fraud/cli.py` — subcomandos `train` / `score` / `evaluate` / `simulate`.
- `src/fraud/monitoring.py`, `simulation.py`, `state.py`, `serving.py` — ver MLOps abajo. `jobs/`, `resources/`, `databricks.yml`: el bundle de Databricks.
- `src/fraud/explainability.py` — SHAP sobre el modelo final, detrás del extra `[shap]`.

## Inicio rápido 🚀

`data/` está en `.gitignore`: descargar el dataset Sparkov desde
[Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) y colocar
`fraudTrain.csv`/`fraudTest.csv` en `data/raw/` antes de correr lo siguiente.

```bash
uv sync --all-extras
uv run ruff check src/ tests/ jobs/ && uv run ruff format --check src/ tests/ jobs/
uv run pytest -q

uv run fraud train --data data/raw/fraudTrain.csv --model xgboost --out models/
uv run fraud evaluate --model models/xgboost.joblib --data data/raw/fraudTest.csv
uv run fraud score transacciones.csv --model models/xgboost.joblib --history data/raw/fraudTrain.csv -o scores.csv
```

`uv run jupyter nbconvert --execute --inplace fraud-portfolio_es.ipynb` reproduce la
notebook desde cero.

## MLOps: desplegar, monitorear, simular 🛰️

```
Volume (CSVs) ─► train_register ─► run MLflow ─► modelo UC fraud.ml.fraud_model
                                      │              @challenger ─gate─► @champion
                                      ▼
 job daily_replay (1 ejecución = 1 día simulado, con drift inyectado opcional)
   simulate_day ─► raw.transactions / ml.predictions / raw.labels (retraso de 7 días)
   refresh_features ─► features.card_state ─► online store ─► endpoint `fraud-scoring`
   monitor ─► monitoring.daily_metrics / alerts ─► dashboard AI/BI + alerta SQL
   retention ─► elimina predicciones/etiquetas de más de 12 meses
```

- **Registry y linaje.** `FraudModel` lleva `metadata` (versión = timestamp + git sha, SHA-256 de los CSV,
  versiones de librerías) y `reference` (distribuciones de validación) dentro del artefacto.
  `jobs/train_register.py` lo registra en MLflow como `fraud.ml.fraud_model`, asigna `@challenger` y lo promueve
  a `@champion` solo si el PR-AUC de **validación** está a menos de 0.005 del campeón actual (nunca el holdout:
  `fraudTest` es solo para reportar).
- **Reentrenamiento.** Corre `train_register` con `through_date=AAAA-MM-DD` para reajustar desde `raw.transactions` +
  `raw.labels` (toda fila hasta esa fecha debe tener etiqueta). Justo después de un drift fija `test_days`, `valid_days`
  (validación = últimos N días) y `half_life_days` (pesos por recencia); los valores por defecto son incorrectos en ese
  caso, ver "Reentrenar tras un drift". Nada lo dispara solo: el runbook de abajo dice cuándo debe hacerlo una persona.
- **Camino online.** Las features necesitan todo el historial de la tarjeta, así que `features.card_state` (una
  fila por tarjeta, historial como arrays, `src/fraud/state.py`) se publica a un online store de Databricks y se
  consulta por `cc_num` al servir. El endpoint reutiliza `build_features(history=...)`: una sola implementación
  para batch y online, con test de paridad (`tests/test_state.py`). Límite: el estado crece con la actividad;
  en producción se guardarían ventanas o sketches aceptando features acumuladas aproximadas.
- **Monitoreo** (`src/fraud/monitoring.py`, solo numpy/pandas). Mismo día, sin etiquetas: PSI por feature y del
  score (ventana de 7 días, para que el volumen por día de semana no parezca drift), tasa de alertas vs
  validación, categorías no vistas, nulos. Con etiquetas a 7 días: recall y ahorro neto como proporción del
  dinero defraudado (no por transacción: eso cambia con la prevalencia, 0.59% en validación vs 0.39% en holdout)
  sobre una ventana madura de 14 días. Bandas: PSI > 0.10 warning, > 0.25 crítico; tasa de alertas fuera de 0.5-2x;
  recall -10 pp; ahorro -20%. Solo las filas `critical` alimentan la alerta SQL; los warnings son solo del dashboard.
- **Simulación** (`src/fraud/simulation.py`). Reproduce `fraudTest.csv` día a día; desde `onset_day` se inyecta
  un drift. Medido con 45 días simulados y inicio el día 21 (`uv run fraud simulate --scenario ...`):

| Escenario | Qué cambia | Primera alerta crítica |
|---|---|---|
| `none` | nada | **una falsa alarma**: `alert_rate_ratio` día 27 (0.495 vs el piso de 0.5); más un warning persistente de PSI (0.11-0.12 en `tx_count_card_7D`) |
| `category_shift` | 40% del tráfico salta a `shopping_net` | día 23, `psi_category` |
| `amount_inflation` | todos los montos x1.8 | día 23, `alert_rate_ratio`; PSI en features de monto desde el día 25 |
| `new_fraud_pattern` | el fraude pasa a ser pequeño y diurno | día 23, tasa de alertas (ver abajo); **alerta de recall día 28** (0.86 vs piso 0.88, tras el retraso de 7 días); ahorro día 37 |

  La última fila se lee junto con la primera: la regla de tasa de alertas no es específica (también dispara en
  `none`, el día 27), así que el concept drift solo se *confirma* cuando llegan las etiquetas. Las bandas no se
  ajustaron sobre el holdout; un replay del período de validación sin drift (54 días) no generó alertas (PSI máximo
  0.036, cociente de tasa de alertas entre 0.51 y 1.55, o sea que el piso es justo incluso en muestra). La menor
  prevalencia del holdout lleva el cociente a ~0.7, por eso `none` roza el piso ahí.

```bash
uv run fraud simulate --model models/xgboost.joblib --scenario new_fraud_pattern   # local, sin Databricks
V="--profile datakrdo --var warehouse_id=<id> --var catalog=workspace"              # Free Edition: usar el catálogo `workspace` existente
databricks bundle deploy $V                                                         # schemas, jobs, dashboard, alerta
databricks fs cp data/raw/fraudTrain.csv dbfs:/Volumes/workspace/raw/landing/ --profile datakrdo   # y fraudTest.csv
databricks bundle run train_register $V
scripts/deploy_endpoint.sh datakrdo fraud <grupo-equipo>                            # endpoint sobre @champion + CAN_QUERY al grupo
databricks bundle run daily_replay $V --params scenario=new_fraud_pattern,onset_day=5   # un día simulado por ejecución
```

> Desplegado y ejecutado en un workspace Databricks Free Edition: `bundle validate --strict` pasa, `train_register`
> registra `@champion`, `daily_replay` (replay, monitor, retención, publicación al online store) corre de punta a
> punta y el endpoint `fraud-scoring` responde (`serving-endpoints query`). Solo un workspace real muestra lo que
> hubo que corregir: el experimento MLflow en serverless, la llave primaria en la tabla de features, la búsqueda de
> la versión del modelo, el wrapper del Feature Store que oculta el modelo, `get_online_store` que devuelve `None`
> en vez de fallar, la forma del JSON del endpoint y **la sincronización del online store que fallaba con
> `DELTA_SCHEMA_CHANGED`** cuando la tabla de features se sobrescribe o altera en cada refresco (lee su change
> feed; el job ahora hace MERGE). **No disponible acá:** la inference table del endpoint (AI Gateway la rechaza en
> este workspace). **Sin ejercitar:** paridad de scores entre el endpoint y el camino batch, y el dashboard con datos.

**Runbook.** `psi_*` crítico en entradas: revisa primero los datos de origen (esquema, unidades, nuevo mix de
comercios); reentrena solo si el recall también cae. `alert_rate_ratio` fuera de banda con entradas estables:
sospecha concept drift, extrae casos etiquetados y espera el chequeo de recall. `recall`/`savings_ratio`: reentrena
con `through_date`; el gate sobre validación decide la promoción. Retención: `ml.predictions` y `raw.labels` se
cortan a 12 meses (`jobs/retention.py`); `raw.transactions` no, porque el estado de tarjeta (features acumuladas)
se construye desde ahí.

## Reentrenar tras un drift 🔁

Ejecutado en el workspace Databricks tras `new_fraud_pattern` (el fraude pasa a ser pequeño y diurno, inicio 2020-06-26),
evaluado en los mismos 4 días reservados (2020-07-12..15, 37 fraudes):

| Modelo | PR-AUC | Recall | Qué es |
|---|---|---|---|
| Campeón v4 (patrón viejo) | 0.003 | 0.00 (0 de 37) | el modelo que levantó las alertas de drift |
| v6 / v7 (primeros reentrenos) | 0.003 | 0.00 | **no entrenaron con los días de drift** |
| **v8** | **0.103** | **0.62** (23 de 37) | validación = últimos 4 días, vida media de recencia 14 días |

Dos valores por defecto de un reentreno ingenuo ocultaron el patrón nuevo, y ambos se descubrieron con este ejercicio:

1. **El split de validación se tragó el drift.** El 20% más reciente de un historial de 18 meses empieza en marzo, así
   que 15 días de drift cayeron enteros en validación y ninguno en entrenamiento (`split_by_recency(valid_days=...)`).
2. **El patrón viejo ahogó al nuevo.** ~7.500 fraudes del patrón viejo contra ~180 nuevos. En local, con los mismos
   datos: historial completo 0.07 de PR-AUC, últimos 60 días 0.28, pesos por recencia (vida media 14 d) 0.38, solo datos
   recientes 0.50. Una vida media más corta se adapta más rápido pero olvida más; la simulación no puede medir ese
   olvido, porque en ella el fraude viejo desaparece por completo.

Cuántos datos con drift hacen falta (curva de aprendizaje local, mismo test): 3 días 0.017, 6 días 0.07, 15 días 0.18,
20+ días ~0.20 de PR-AUC. Con 7 días de retraso en las etiquetas, el modelo queda casi ciego al patrón nuevo unas tres
semanas, así que el disparador del reentreno debe ser un mínimo de fraudes nuevos etiquetados (~150-200), y otra cosa
(una regla) debe cubrir ese hueco. Los números de v8 salen de 37 fraudes: tómalos como prueba de existencia, no como
estimación.

## ¿Está listo para producción? ⚖️

No: es una implementación de referencia del ciclo MLOps. Los datos son sintéticos (fraude por reglas, 983 tarjetas, sin
validación externa), así que las métricas principales (PR-AUC de test ~0.955) miden qué tan bien el modelo aprende esas
reglas. El diseño de serving no es viable tal como está (un estado refrescado a diario pierde gran parte de la
precisión; ver abajo), la paridad de scores entre endpoint y batch no está probada, el endpoint no registra requests en
este tipo de workspace, no hay pruebas de latencia ni de carga, y el costo de revisión (USD 4) es una suposición. Lo que
sí muestra: validación temporal, features causales, linaje, monitoreo de drift que disparó, un ciclo de reentrenamiento
con gate y una lista honesta de lo que sigue abierto.

## Puntos ciegos conocidos 🔦

**El camino online puede perder la mayor parte de la precisión del modelo.** El estado se refresca una vez al día,
así que una transacción no ve las anteriores de la misma tarjeta en el mismo día, y las features de ráfaga
(`tx_count_card_1h`, `seconds_since_prev_tx`) son en las que más se apoya el modelo. Medido puntuando 14 días del
holdout de las dos formas (las 983 tarjetas, 122 fraudes; `simulation.build_features_frozen`): PR-AUC **0.939 con
historial en vivo vs 0.739 con estado refrescado a diario**, recall 0.951 -> 0.844, alertas 200 -> 331 con el mismo
umbral. Un refresco diario **no** es un diseño de serving aceptable para este modelo: producción necesita el estado
actualizado por transacción (streaming) o reentrenar con features calculadas como se sirven. El job `daily_replay`
puntúa en batch, así que no muestra este problema.

| Punto ciego | Estado |
|---|---|
| Validación/holdout/scoring arrancaban cada tarjeta "en frío" (sin historial): skew train/serve | Corregido con `build_features(history=)`. Medido antes de tokenizar: validación 0.9749 -> 0.9758, test 0.9528 -> 0.9545. Reentreno final: 0.9758 / 0.9563 (la notebook principal, con su propio loader, da 0.9545) |
| `fraud score` exigía `is_fraud`; la salida llevaba PII | Corregido (`require_label=False`); la salida son ids, monto, score y versión del modelo |
| Número de tarjeta completo (PAN) como llave en todas partes | Corregido: `cc_num` se tokeniza con HMAC al cargar (`FRAUD_CARD_KEY`; la llave por defecto existe solo para la demo). Tokenizar también antes del endpoint |
| Nombre, calle, zip y job guardados en las tablas / enviados al endpoint | Corregido: solo las columnas de entrada del modelo (`state.TX_COLUMNS`). `dob` y `lat/long` del domicilio siguen siendo cuasi-identificadores |
| El artefacto no tenía versión, hash de datos ni distribución de referencia | Corregido (`FraudModel.metadata`, `.reference`) |
| El gate de promoción usaba el holdout | Corregido: el gate usa validación. Sigue abierto: tras un reentreno compara PR-AUC de validación medido en conjuntos *distintos* (v8: 0.48 vs campeón 0.98), así que no promoverá solo a un modelo mejor; v8 se promovió a mano tras compararlos en los mismos días de test |
| Boosters fijos a GPU | Override `--device cpu` / `device=` (serverless, CI) |
| Reejecutar un job duplicaba filas; el monitor fallaba antes de que existieran etiquetas | Corregido: delete+append por fecha, tablas creadas de antemano (sin probar en Databricks) |
| CI: actions sin fijar, sin auditoría de dependencias | Actions fijadas por SHA, Dependabot, `pip-audit` (encontró y corrigió 6 paquetes vulnerables) |
| El monitor vigila el camino batch, no el endpoint; el endpoint no tiene inference table (no soportada en este tipo de workspace) | Abierto: pasar una muestra diaria por el endpoint y comparar scores con el batch |
| Un reentreno ingenuo no aprende un drift reciente (split de validación, dilución) | Corregido con `valid_days` y `half_life_days`; ver "Reentrenar tras un drift" |
| Umbral y calibración se ajustan sobre el mismo slice de validación; el recall de referencia (0.98) es en muestra | Abierto: un split anidado sería más limpio; efecto no medido |
| `age_years` es input del modelo | Abierto: sin revisión de equidad; quitar o justificar antes de un despliegue real |
| El retraso de etiquetas (~7 días) no se modela en el entrenamiento; datos sintéticos, 983 tarjetas | Abierto: documentado, simulado solo en el monitoreo |
| La ventana PSI de 7 días no ve una anomalía de un solo día; `toPandas()` de todo el historial cada día | Abierto: sirve para ~2M filas, no para escala de producción |

## Limitaciones ⚠️

Dataset sintético (Sparkov), sin validación externa. Sin datos de dispositivo, sesión ni
canal de pago. `FP_COST` (costo de revisión manual, USD 4) es una estimación externa al
dataset — la sección de sensibilidad de la notebook muestra cuánto depende de ella la
recomendación. 983 tarjetas es una población chica. El modelo no tiene forma de manejar una
tarjeta sin historial (arranque en frío): ahí las features de velocidad valen cero por
construcción, no por un error del modelo.

## Licencia

MIT — ver el archivo `LICENSE`.
