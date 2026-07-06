# Pronóstico de demanda para reposición inteligente — Retail MX

Sistema de pronóstico de demanda a **14 días** para 480 series tienda × categoría, diseñado para reemplazar la señal reactiva del sistema de reposición vigente. Validado con backtest walk-forward, holdout final e intervalos de confianza:

| WAPE | Modelo propuesto | Sistema actual | Naïve estacional |
|---|---|---|---|
| Backtest (6 cortes) — total | **0.275** [0.265, 0.286] | 0.325 | 0.416 |
| Backtest — picos (Buen Fin/Navidad) | **0.316** | 0.369 | 0.519 |
| Holdout final (14 días nunca vistos) | **0.239** [0.235, 0.243] | 0.289 | 0.268 |

La ventaja sobre el sistema actual es **estadísticamente significativa** (ΔWAPE pareado −0.050, IC 95% [−0.081, −0.020]) y el modelo decide con **14 días de anticipación**, mientras el sistema vigente reacciona el mismo día. En el Buen Fin — un evento con *cero* ejemplos en la historia de entrenamiento — el sesgo pasa de −60% a **−12%** gracias a priors de fuentes públicas con disciplina de vintage.

> El enunciado original del reto está en [docs/enunciado_original.md](docs/enunciado_original.md). El proceso de trabajo y el uso de herramientas de AI se documentan en [PROCESS.md](PROCESS.md).

---

## Quickstart (3 comandos)

Requiere [uv](https://docs.astral.sh/uv/) (gestiona el Python 3.11.13 exacto y todas las dependencias desde `uv.lock`):

```bash
uv sync              # entorno completo y reproducible
uv run pytest        # 28 tests (incluye garantías anti-leakage)
uv run retail-flow   # flujo completo: valida → entrena → registra en MLflow → pronostica
```

`retail-flow` deja: métricas de holdout en `outputs/flows/<ts>/`, el modelo desplegable en `outputs/models/<ts>/model.joblib`, la corrida completa en MLflow (hiperparámetros, métricas, artefactos, signature) y el pronóstico operativo de los próximos 14 días.

## Comandos

| Comando | Qué hace |
|---|---|
| `uv run retail-flow` | **Orquestador**: proceso completo parametrizable (ver abajo) |
| `uv run retail-backtest` | Evaluación walk-forward (6 cortes) contra 3 baselines, con IC 95% y prueba pareada |
| `uv run retail-holdout` | Holdout final honesto: aparta los últimos N días y los pronostica sin haberlos visto |
| `uv run retail-train` | Entrena el modelo final con toda la historia y lo serializa |
| `uv run retail-predict` | Pronóstico de los próximos 14 días con el último modelo |
| `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` | Explorar corridas, métricas y el Model Registry |
| `uv run ruff check src tests` | Verificación PEP8 |

Parámetros del orquestador: `--train-start/--train-end` (ventana de entrenamiento), `--horizon`, `--holdout-days` (evaluación honesta previa; 0 la desactiva), `--with-backtest`, `--learning-rate/--max-iter/--seed`, `--experiment/--run-name/--no-register`, `--no-forecast`. Ejemplo:

```bash
uv run retail-flow --train-end 2024-01-31 --holdout-days 14 --run-name ventana-enero
```

## Estructura

```
data/                  CSV originales del reto (intactos) + external/ (priors de eventos con fuentes)
src/
├── config.py          parámetros centrales (horizonte, rezagos, hiperparámetros, folds)
├── preprocesamiento/  carga tipada · calidad · calendario · features anti-leakage
├── entrenamiento/     DemandForecaster (HGBR global, Poisson) + baselines
├── inferencia/        generación del pronóstico operativo
├── postproceso/       priors de eventos (vintage + complemento)
├── evaluacion/        métricas con IC · splits · backtest · holdout · calibración
├── seguimiento/       artefactos por corrida · tracking MLflow con signature
└── pipelines/         entry points CLI (retail-*)
tests/                 28 tests: fórmulas, invariantes anti-leakage, integración
notebooks/             01 EDA · 02 resultados · 03 backtest 30d · 04 backtest 14d
docs/                  resumen técnico · model card · enunciado original
outputs/               artefactos por corrida (regenerables con los comandos)
```

## Las 4 decisiones que definen el sistema

1. **Anti-leakage por construcción.** Toda feature de demanda usa rezagos ≥ 14 días (el horizonte): cada predicción es un pronóstico honesto emitido 14 días antes, y el mismo `build_features` sirve para entrenar, backtestear y predecir (paridad train/serving). Protegido por test: perturbar los 13 días previos no cambia ninguna feature.
2. **La variable trampa, detectada y reconvertida.** `replenishment_signal` se deriva de la demanda contemporánea (corr 0.87 con la media móvil de la demanda *pasada*): usarla como feature sería fuga de información. Se excluye del modelo y se usa como **baseline "sistema actual"** — el comparador que hay que vencer.
3. **Conocimiento externo con disciplina de vintage.** Los eventos anuales (Buen Fin) no existen en un año de historia. Se inyectan factores de fuentes públicas (INEGI EMEC, ANTAD, BBVA Research) versionados por año de publicación: para pronosticar el año Y solo se usa lo publicado antes de Y. El factor actúa como **complemento** (nunca multiplicador ciego): solo completa la escala que el modelo no detecta por sí mismo.
4. **Evaluación con incertidumbre.** IC 95% por bootstrap de bloques de fecha (los errores se correlacionan dentro de un día) y prueba **pareada** de ΔWAPE para afirmar "le gana" con rigor — incluida la honestidad de que en picos la ventaja aún no es concluyente con un solo año de eventos.

## Notebooks (exploración y validación; la lógica vive en `src/`)

| Notebook | Contenido |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Calidad de datos, hallazgos (transacciones fantasma, señal reactiva, promos sin lift) y propuesta |
| [02_resultados](notebooks/02_resultados.ipynb) | Resumen ejecutivo: decisiones de diseño, backtest completo con IC, predicciones por formato/categoría |
| [03_backtest](notebooks/03_backtest.ipynb) | Simulación de producción a 30 días: expone el **horizonte útil** del modelo |
| [04_backtest_14d](notebooks/04_backtest_14d.ipynb) | Simulación a 14 días (horizonte de diseño): desglose por formato y categoría |

## Limitaciones conocidas (diagnosticadas, con plan)

- **Horizonte útil ≈ 14 días**: más allá, los rezagos caducan y el naïve lo alcanza (notebook 03). Operación recomendada: re-emisión con origen rodante cada 1–2 semanas.
- **Un solo año de eventos anuales**: la ventaja en picos es consistente pero no estadísticamente concluyente; los priors vintage-2023 ya versionados se activan para 2024.
- **Medias móviles contaminadas por eventos** (ma28 arrastra el Buen Fin a mediados de diciembre): siguiente refinamiento identificado.
- Detalle completo en la [model card](docs/MODEL_CARD.md) y el [resumen técnico](docs/RESUMEN_TECNICO.md).
