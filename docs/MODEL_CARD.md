# Model Card — `retail-demand-forecaster`

## Detalles del modelo

| | |
|---|---|
| Nombre en el registry | `retail-demand-forecaster` (MLflow Model Registry local, `sqlite:///mlflow.db`) |
| Tipo | `HistGradientBoostingRegressor` (scikit-learn) global + complemento de priors de eventos en inferencia |
| Objetivo | `units_sold` diario por tienda × categoría (480 series) |
| Pérdida | Poisson (la demanda es un conteo no negativo) |
| Horizonte de diseño | **14 días** — cada predicción se emite con toda la información disponible 14 días antes |
| Entrenamiento | ~198k filas (ene-2023 → feb-2024), semilla fija (42), ~10 s en laptop |
| Artefacto desplegable | `DemandForecaster` (estimador + contrato de features + política de prior), `model.joblib` |
| Reproducción | `uv run retail-flow` (registra hiperparámetros, métricas, artefactos y signature en MLflow) |

## Uso previsto

- **Uso principal:** alimentar decisiones de reposición de inventario por tienda × categoría con 1–14 días de anticipación, en reemplazo/complemento de la señal reactiva vigente.
- **Modo de operación recomendado:** re-emisión con origen rodante cada 1–2 semanas. **No** emitir a >14 días en un solo disparo: fuera del horizonte los rezagos de demanda caducan y el modelo pierde su ventaja frente a un naïve estacional (cuantificado en `notebooks/03_backtest.ipynb`).
- **Fuera de alcance:** pronóstico a nivel SKU (los datos son categoría-agregados), tiendas nuevas sin historia (~4 semanas mínimas para poblar rezagos), decisiones de precio/promoción.

## Datos y features

- 3 fuentes del reto (transacciones diarias, catálogo de 80 tiendas, calendario de eventos MX) + 1 tabla externa curada (`data/external/event_priors.csv`: factores de eventos de INEGI EMEC / ANTAD / BBVA Research, versionados por año de publicación).
- 28 features en 5 familias: demanda rezagada (rezagos 14/21/28 y medias móviles, **todas con información ≤ t−14**), tendencia, calendario del día objetivo, atributos de tienda, categóricas.
- Exclusiones deliberadas: `replenishment_signal` (leakage: se deriva de la demanda contemporánea; se usa como baseline), `has_promotion` (lift nulo demostrado en el EDA).
- Faltantes: target nulo (~3%, MCAR) excluido del ajuste y de la evaluación, nunca imputado.

## Métricas (con incertidumbre)

WAPE con IC 95% (bootstrap por bloques de fecha); "vs sistema" = ΔWAPE **pareado** contra el sistema de reposición vigente.

| Evaluación | WAPE [IC 95%] | Δ vs sistema [IC 95%] | ¿Significativo? |
|---|---|---|---|
| Backtest walk-forward (6 cortes) — total | 0.275 [0.265, 0.286] | −0.050 [−0.081, −0.020] | **Sí** |
| Backtest — días normales | 0.253 [0.245, 0.262] | −0.048 [−0.063, −0.034] | **Sí** |
| Backtest — picos (Buen Fin/Navidad) | 0.316 [0.299, 0.339] | −0.053 [−0.127, +0.038] | No (15 días de evento) |
| Holdout final 14 días (nunca vistos) | 0.239 [0.235, 0.243] | −0.050 [−0.077, −0.022] | **Sí** |

Comportamiento en eventos: Buen Fin (cero ejemplos en entrenamiento) sesgo −60% → **−12%** con priors públicos pre-2023; Nochebuena −46% → **+8%** con el perfil diario. Aporte del prior vs. ablación en picos: −0.142 de WAPE, significativo.

## Limitaciones

1. **Horizonte útil ≈ 14 días.** A 30 días de una sola emisión el WAPE sube a 0.307 y cae por debajo del naïve (0.287). Mitigación operativa: re-emisión rodante.
2. **Un solo año de eventos anuales:** la ventaja en picos es consistente pero no concluyente estadísticamente; el modelo depende de los priors externos para eventos nunca vistos.
3. **Medias móviles contaminadas por eventos:** la ma28 arrastra el Buen Fin hasta mediados de diciembre (+~30% de sobre-pronóstico en días ordinarios del 15–23 dic). Refinamiento identificado: ventanas móviles que excluyan días de evento.
4. **Eventos de fecha móvil futuros** (Buen Fin, Semana Santa) no son extrapolables por `extend_calendar`; en producción el calendario comercial oficial debe ser insumo.
5. Datos **sintéticos** del reto: los números validan el *método*; las magnitudes reales requerirían recalibración con datos de la cadena. Sin PII ni consideraciones de equidad aplicables.

## Criterios de reentrenamiento

| Disparador | Acción |
|---|---|
| **Cadencia base** | Reentrenar y re-emitir cada 7–14 días (mantiene frescos los rezagos; el costo es ~10 s) |
| **Post-evento anual** | Tras cada Buen Fin/Navidad: recalcular el factor observado y versionarlo en `event_priors.csv` con el vintage del año (el mecanismo ya está construido: los factores 2023 se activan solos para 2024) |
| **Por monitoreo** | Cualquier umbral crítico de la sección siguiente sostenido 7 días |
| **Por cambio estructural** | Alta/baja de tiendas o categorías, cambio de formato, nueva fuente de datos → reentrenar y re-validar con `retail-backtest` antes de promover en el registry |

## Monitoreo en producción (métricas y umbrales)

Comparar siempre contra el real observado con el rezago natural de los datos (día vencido):

| Métrica | Cálculo | Alerta | Crítico |
|---|---|---|---|
| WAPE rodante 7d (global) | error de los pronósticos emitidos vs real | > 0.30 | > 0.35 |
| Sesgo rodante 7d | (Σpred − Σreal) / Σreal | \|sesgo\| > 0.08 | \|sesgo\| > 0.15 |
| WAPE vs naïve estacional | correr el naïve en paralelo (es gratis) | modelo pierde 7 días seguidos | modelo pierde 14 días seguidos |
| Cobertura de datos | % de celdas tienda×categoría con dato del día | < 97% | < 90% |
| Deriva de features | % de rezagos nulos en la emisión | > 5% | > 15% |
| Días de evento | sesgo del día del evento (post-mortem) | \|sesgo\| > 0.15 | \|sesgo\| > 0.30 → recalibrar prior |

Racional de los umbrales: el WAPE esperado en operación normal es ~0.24 (holdout) — 0.30 es ~25% de degradación y 0.35 es peor que el naïve histórico (0.27). El sesgo esperado es ±3%; ±8% sostenido implica quiebre o merma sistemáticos. "Perder contra el naïve" es el detector barato de que algo estructural cambió.
