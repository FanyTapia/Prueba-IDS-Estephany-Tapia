# Resumen técnico — Sistema de pronóstico de demanda para reposición

**Propósito de este documento:** descripción de nivel avanzado del flujo completo del sistema tal como está implementado hoy. Sirve como base para la refactorización hacia un flujo serializado por etapas (preprocesado → entrenamiento → inferencia → postproceso) con mejor trazabilidad. Refleja el estado del código a la fecha; no propone cambios.

---

## 1 · Problema y solución en una página

**Contexto.** Cadena retail MX: 80 tiendas × 6 categorías × 425 días (2023-01-01 → 2024-02-29). El sistema de reposición vigente (`replenishment_signal`) es reactivo: corr 0.87 con la media móvil de 7 días de la demanda *pasada*. Consecuencia medida: sub-surte ~34% en picos de demanda (Buen Fin) y sobre-surte ~2.3× en los días posteriores.

**Solución.** Pronóstico de `units_sold` a **horizonte de 14 días** para las 480 series tienda × categoría, con un **modelo global** de gradient boosting + **priors de eventos** desde fuentes públicas, validado por backtest walk-forward contra el sistema vigente.

**Resultado (backtest, 6 cortes, ~39k predicciones evaluadas por modelo):**

| Segmento | Modelo propuesto | Sistema actual | Naïve estacional |
|---|---|---|---|
| Total | **0.275** | 0.325 | 0.416 |
| Pico (Buen Fin/Navidad) | **0.316** | 0.369 | 0.519 |
| Días normales | **0.253** | 0.301 | 0.360 |

(WAPE; el modelo propuesto gana los 6 folds contra los 3 competidores. Sesgo en picos: +0.09 vs −0.23 del sistema actual.)

---

## 2 · Mapa de módulos y dependencias

> **Actualización (refactorización por etapas):** los módulos se reorganizaron en
> paquetes que nombran su etapa del proceso. La lógica es idéntica (28 tests lo
> garantizan); solo cambió la ubicación y las rutas de import.

```
src/
├── config.py ················ transversal: rutas, HORIZON=14, LAG_DAYS, MODEL_PARAMS
├── viz.py ··················· transversal: estética de notebooks (matplotlib + plotly)
├── preprocesamiento/ ········ ETAPA 1: data · quality · calendar_utils · features
├── entrenamiento/ ··········· ETAPA 2: models (DemandForecaster + baselines)
├── inferencia/ ·············· ETAPA 3: prediccion (generar_pronostico)
├── postproceso/ ············· ETAPA 4: priors (vintage + complemento de eventos)
├── evaluacion/ ·············· ETAPA 5: metrics (IC + pareado) · splits · backtest ·
│                                        holdout · calibration (diagnóstico aislado)
├── seguimiento/ ············· ETAPA 6: artifacts (run dirs) · mlflow_tracking
│                                        (params + métricas + artifacts + signature,
│                                         backend sqlite:///mlflow.db, registry local)
└── pipelines/ ··············· entry points CLI: run_backtest · run_train ·
                               run_predict · run_holdout · run_all (orquestador)
```

Dirección de dependencia estricta: `config` no depende de nadie; `pipelines/` depende de todo; ningún paquete importa "hacia arriba". `evaluacion/calibration` y `preprocesamiento/quality` están fuera del camino crítico de predicción.

**Orquestador `retail-flow`** (`pipelines/run_all.py`): ejecuta el proceso completo en una corrida — carga y ventana de entrenamiento → holdout honesto → (backtest opcional) → entrenamiento final → registro MLflow con signature y Model Registry → pronóstico operativo. Parámetros: `--train-start/--train-end` (ventana), `--horizon`, `--holdout-days`, `--with-backtest`, `--learning-rate/--max-iter/--seed`, `--experiment/--run-name/--no-register`, `--no-forecast`.

---

## 3 · El flujo, etapa por etapa

Las cuatro etapas lógicas que pide la refactorización ya existen, pero hoy **fluyen en memoria** dentro de cada entry point (no hay artefactos intermedios serializados entre ellas). Esta sección documenta cada etapa con sus contratos implícitos de entrada/salida — el insumo directo para diseñar los contratos serializados.

### 3.1 · PREPROCESADO

**Código:** `data.py` + `features.complete_grid`.

| | Contrato |
|---|---|
| Entrada | `data/*.csv` crudos (nunca se sobrescriben) |
| Salida (en memoria) | Panel rectangular de 204,000 filas: producto completo tienda × categoría × día del rango del calendario, con atributos de tienda y calendario ya unidos (`validate="m:1"` en ambos merges) |

Decisiones clave:
- **Grid rectangular forzado:** los días caídos de POS (3 tiendas, 2–3 días c/u) quedan como filas con target nulo. Esto garantiza que *desplazamiento por posición ≡ desplazamiento por fecha calendario* dentro de cada serie — prerequisito de los rezagos.
- **Sin imputación del target:** las filas con `units_sold` nulo (~3%) se excluyen del ajuste (`fit_mask = target.notna()`) y de la evaluación (reportadas como `n_sin_real`), nunca se inventan.
- Tipado en origen: fechas como `datetime`, categorías como `Categorical` con orden canónico de `config`.

### 3.2 · FEATURE ENGINEERING

**Código:** `features.add_demand_features` + `features.build_features` + `priors.attach_uplift_prior`.

| | Contrato |
|---|---|
| Entrada | Panel rectangular de 3.1 |
| Salida (en memoria) | `(panel, feature_cols)`: panel con 28 features + columnas auxiliares (target, `replenishment_signal`, `uplift_prior`, flags); `feature_cols` es la lista explícita de lo que entra al modelo |

Las 28 features, por familia:

| Familia | Features | Información que usan |
|---|---|---|
| Demanda rezagada (7) | `lag14/21/28`, `lag_dow_mean`, `ma7`, `ma28`, `std7` | Solo hasta **t − 14** (las MA se calculan sobre la serie ya desplazada 14 días) |
| Tendencia (1) | `t_index` (días desde el inicio) | Determinista |
| Calendario (10) | dow, semana ISO, mes, trimestre, festivo, quincena, fin de semana, navidad, buen fin, semana santa | Conocidas de antemano por definición |
| Tienda numéricas (5) | m², cajas, año de apertura, farmacia, gasolinera | Estáticas |
| Categóricas (5) | `store_id`, `category`, formato, región, NSE | Estáticas; dtype `category` (HGBR nativo) |

**El invariante central del sistema** (protegido por `ValueError` en código y por `tests/test_features.py`): *ninguna feature derivada de la demanda usa información posterior a t − HORIZON*. Los rezagos son múltiplos de 7 (mismo día de semana) y ≥ 14. Consecuencia arquitectónica valiosa: **el mismo `build_features` sirve idéntico para entrenar, backtestear y predecir** — no hay una versión "training" y otra "serving" que puedan divergir (paridad train/serving por construcción).

Exclusiones deliberadas: `replenishment_signal` (se deriva de la demanda contemporánea → leakage; su rol es baseline), `has_promotion` (el EDA demostró lift nulo en venta), `uplift_prior` (viaja en el panel pero **no** está en `feature_cols` — es insumo del postproceso, no del modelo).

### 3.3 · PRIORS DE EVENTOS (conocimiento externo)

**Código:** `priors.py` + `data/external/event_priors.csv`.

La tabla versiona factores por evento × categoría con columna **`vintage`** (año en que la información era pública) y fuente citada. Dos generaciones conviven: vintage 2022 (estimaciones de INEGI EMEC / ANTAD / BBVA Research, ediciones 2018–2022) y vintage 2023 (lo observado en los propios datos, aplicable desde 2024).

- **Regla de vintage (anti-leakage simétrica a la de los rezagos):** para una fecha del año *Y* solo son usables filas con `vintage < Y`. El Buen Fin 2023 se pronostica con el factor ~2× de ANTAD pre-2023, no con el 3.7× que nosotros medimos *durante* 2023.
- **Sub-eventos por fecha** (`_subeventos_navidad`): la temporada navideña se desglosa en regímenes diarios — `navidad_ordinaria` (~1.4), `nochebuena`/`navidad_dia`/`fin_de_ano` (~2.5–3×) y **supresión** con factor < 1 (`ano_nuevo` 0.5, `cuesta_enero` 0.8). Un factor plano por temporada demostró ser la fuente principal de error en diciembre.
- Categoría comodín `*` con precedencia de la fila específica.
- Salida: columna `uplift_prior` en el panel (1.0 en días sin evento).

**Diagnóstico de riesgo del dato externo** (`calibration.py` + `event_reference.csv`, fuera del pipeline): compara factor externo vs observado en 10 eventos medibles. Conclusión empírica: el dato externo es confiable donde hay mecanismo económico universal (quincena +2% de brecha, Independencia −6%) y peligroso en festivos de puente (dirección opuesta: el público asume alza, los datos muestran −50%). Esto acota cuándo un prior externo debe entrar a la tabla productiva.

### 3.4 · ENTRENAMIENTO

**Código:** `models.DemandForecaster` + `pipelines/run_train.py`.

| | Contrato |
|---|---|
| Entrada | Panel + `feature_cols` de 3.2; filas con target no nulo |
| Salida (serializada) | `outputs/models/<ts>/model.joblib` + `run_config.json` |

- Estimador: `HistGradientBoostingRegressor` con `loss="poisson"` (demanda = conteo no negativo), `categorical_features="from_dtype"`, hiperparámetros en `config.MODEL_PARAMS` (lr 0.06, 400 iter, 63 hojas, L2 1.0, seed fija).
- **Modelo global:** uno para las 480 series. Los patrones compartidos (efecto sábado, quincena) se aprenden una vez con ~198k filas; la individualidad de cada serie entra vía `store_id`/rezagos/atributos.
- El artefacto `DemandForecaster` encapsula estimador + `feature_cols` + `horizon` + `prior_col`: la inferencia **no puede** desalinearse del entrenamiento (hay verificación explícita en `run_predict`).
- **El entrenamiento no toca el prior** (el target no se transforma). Esto hace que la variante con y sin prior compartan exactamente el mismo modelo entrenado — el prior es 100% postproceso.

### 3.5 · INFERENCIA

**Código:** `pipelines/run_predict.py` + `calendar_utils.extend_calendar`.

| | Contrato |
|---|---|
| Entrada | Último `model.joblib` (o `--model-dir`), CSVs crudos, `--cutoff` opcional |
| Salida (serializada) | `outputs/forecasts/<ts>/forecast.csv` (fecha, tienda, categoría, unidades) + `run_config.json` |

Mecánica: se trunca `tx` al cutoff, se **extiende el calendario** 14 días con los campos deterministas (dow, quincena, temporada navideña, festivos fijos; los de fecha móvil quedan en `False` con warning), se construye el mismo panel de 3.1–3.3 (las fechas futuras entran como filas vacías cuyos rezagos ≥ 14 sí existen), y se predice sobre las filas posteriores al cutoff.

### 3.6 · POSTPROCESO (complemento de prior)

**Código:** dentro de `DemandForecaster.predict` — hoy es la única pieza de postproceso y vive acoplada al modelo (candidata natural a etapa propia en la refactorización).

Para cada fila con `uplift_prior ≠ 1`:

```
pred        = model(X)                        # con flags del día
contrafactual = model(X | flags de evento = False)   # "mismo día sin evento"
implícito   = clip(pred / contrafactual, 0.05, 20)
final       = contrafactual × max(implícito, factor)   si factor ≥ 1  (pico)
            = contrafactual × min(implícito, factor)   si factor < 1  (supresión)
```

Semántica: el factor externo **completa la escala que el modelo no detecta por sí mismo** — nunca se duplica sobre lo que el modelo ya generaliza de otros festivos (elimina el doble conteo del diseño multiplicativo ingenuo), y nunca modera a un modelo que detecta un efecto mayor. Cierre final: `clip(pred, 0, ∞)`.

Flags de contrafactual: `is_buen_fin`, `is_navidad_season`, `is_holiday`.

### 3.7 · EVALUACIÓN (walk-forward)

**Código:** `splits.py` + `backtest.py` + `metrics.py` + `pipelines/run_backtest.py`.

- **Folds:** 6 orígenes (`BACKTEST_ORIGINS`) elegidos para cubrir 4 periodos normales + Buen Fin + Navidad. Por fold: entrena con `date ≤ origen`, evalúa `(origen, origen+14]`. `make_folds` falla si un fold excede los datos.
- **4 competidores sobre exactamente las mismas celdas:** campeón con prior, campeón sin prior (ablación), naïve estacional (= `lag_dow_mean`), y sistema actual (`replenishment_signal` reescalada por serie con factor aprendido solo en train — comparador con ventaja informacional documentada: usa la señal del mismo día).
- **Métricas:** WAPE (decide), MAE (dimensiona), bias (diagnostica quiebre vs merma), segmentadas total/pico/normal y por cualquier corte vía `evaluate(by=...)`. Filas sin real observado se excluyen y se reportan.
- Salida serializada: `predictions.csv` (formato largo: fold × fecha × serie × modelo), `metrics.csv`, `run_config.json`.

### 3.8 · TRAZABILIDAD

**Código:** `artifacts.py`. Cada corrida escribe un directorio inmutable `outputs/{backtests,models,forecasts}/<timestamp_utc>/` con `run_config.json`: timestamp, versiones de Python/pandas/sklearn, plataforma, horizonte, `feature_cols` completas, hiperparámetros, orígenes de folds. `latest_run_dir()` resuelve la corrida más reciente (así `predict` encuentra al último modelo sin acoplamiento).

---

## 4 · Invariantes que cualquier refactorización debe preservar

Estas reglas son la esencia del sistema; todas tienen test que las protege:

1. **Rezagos ≥ horizonte** — ninguna feature de demanda usa información posterior a t−14 (`test_no_leakage_perturbando_el_futuro_cercano`: perturbar los 13 días previos no cambia ninguna feature).
2. **Rezagos por fecha calendario, no por posición** (`test_rezagos_son_por_fecha_calendario`).
3. **Regla de vintage** — información del año Y jamás se usa para pronosticar el año Y (`test_regla_de_vintage`).
4. **El prior nunca reduce a un modelo que detecta más, ni empuja contra su propia dirección** (`test_complemento_no_reduce…`, `test_prior_de_supresion…`); neutro exacto sin eventos (`test_prior_es_neutro_sin_eventos`).
5. **`replenishment_signal` jamás es feature** — solo baseline.
6. **Un solo `build_features`** para train/backtest/predict (paridad por construcción).
7. **Folds sin traslape** train/test (`test_folds_no_traslapan…`).
8. **Calibración aislada** — `event_reference.csv` no alimenta predicciones.

Suite completa: 20 tests, ~2s.

---

## 5 · Deuda técnica conocida (diagnosticada, no resuelta)

1. **Contaminación de ventanas móviles por eventos:** `ma28` para mediados de diciembre incluye el Buen Fin → infla la base → sobre-pronóstico ~+30% en días ordinarios del 15–23 dic. Refinamiento identificado: calcular rolling features excluyendo días de evento.
2. **Eventos de fecha móvil en el futuro:** `extend_calendar` no puede extrapolar Buen Fin/Semana Santa (quedan `False` con warning). En producción, el calendario oficial de la cadena debería ser insumo.
3. **Factores externos por categoría vs realidad de la cadena:** las fuentes públicas traen estructura sectorial (electrónica ≫ abarrotes en Buen Fin) que esta cadena no exhibe (uplift uniforme 3.7×). Se auto-corrige al usar vintage 2023 en 2024; el diagnóstico de `calibration.py` acota el riesgo para eventos nuevos.
4. **Etapas acopladas en memoria:** preprocesado → features → modelo fluyen dentro de cada entry point sin artefactos intermedios; el postproceso del prior vive dentro de `DemandForecaster.predict`. (Motivación directa de la refactorización que este documento alimenta.)
5. **Sin pronóstico probabilístico:** salida puntual; reposición real necesita cuantiles (P50/P90) para nivel de servicio. Identificado como siguiente capacidad, no iniciado.

---

## 6 · Anexo: comandos y layout

```bash
uv sync                    # entorno completo desde uv.lock (Python 3.11.13 gestionado)
uv run pytest              # 28 tests
uv run ruff check src tests  # PEP8 verificable
uv run retail-flow         # ORQUESTADOR: proceso completo + MLflow → outputs/flows/<ts>/
uv run retail-backtest     # evaluación walk-forward   → outputs/backtests/<ts>/
uv run retail-train        # modelo final serializado  → outputs/models/<ts>/
uv run retail-predict      # pronóstico 14 días        → outputs/forecasts/<ts>/
uv run retail-holdout      # holdout final honesto     → outputs/holdouts/<ts>/
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # explorar corridas
```

Notebooks (solo lectura/validación, sin lógica de negocio): `01_eda.ipynb` (hallazgos y propuesta), `02_resultados.ipynb` (resumen ejecutivo con artefactos del backtest).
