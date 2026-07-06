# PROCESS.md — Proceso de trabajo y herramientas

Documento requerido por el reto: cómo se construyó esta solución, con qué herramientas, y en particular **cómo se usó la inteligencia artificial** — con honestidad sobre qué hizo la AI y qué criterio fue humano.

## Herramientas

| Herramienta | Rol |
|---|---|
| **Claude Code (Anthropic)** | Asistente de AI principal: exploración de datos, generación de código, ejecución de pipelines y redacción de documentación, bajo dirección y revisión humana continua |
| **uv** | Entorno reproducible: Python 3.11.13 gestionado + `uv.lock` con las 174 dependencias exactas |
| **pytest** | 28 tests; los críticos protegen *invariantes* (anti-leakage, regla de vintage), no solo funciones |
| **ruff** | PEP8 verificable (`uv run ruff check src tests`) |
| **MLflow** | Tracking de corridas (hiperparámetros, métricas, artefactos, signature) + Model Registry local en SQLite |
| **Git** | Versionado incremental durante todo el desarrollo |
| matplotlib / plotly+kaleido | Gráficos de notebooks (plotly interactivo con respaldo PNG para que GitHub los muestre) |

## Cronología del proceso

1. **EDA primero, solución después.** Perfilado exhaustivo de los 3 insumos antes de decidir el problema. Tres hallazgos definieron el proyecto: (a) ~18% de "transacciones fantasma" sin pago en días de promoción; (b) `replenishment_signal` es una media móvil *reactiva* de la demanda pasada (corr 0.87) que sub-surte 34% en picos; (c) las promociones no generan venta incremental.
2. **Elección del problema:** pronóstico de demanda a 14 días para reposición — ataca directamente la debilidad medida del sistema vigente y usa los 3 insumos.
3. **Pipeline productivo desde el inicio** (regla autoimpuesta: los notebooks solo leen artefactos; toda la lógica en `src/` con tests).
4. **Iteración sobre los picos:** el Buen Fin no existe en la historia de entrenamiento → priors de fuentes públicas (INEGI/ANTAD/BBVA) con regla de vintage → mecanismo de *complemento* → perfil diario de la temporada navideña. Cada iteración validada con el backtest.
5. **Rigor estadístico:** IC 95% por bootstrap de bloques de fecha + prueba pareada de ΔWAPE.
6. **Backtests de estrés:** holdout de 30 días (expuso el horizonte útil del modelo) y de 14 días (su zona de diseño).
7. **Refactorización final:** paquetes por etapa de proceso, orquestador `retail-flow`, tracking MLflow, PEP8 con ruff, docstrings completos.

## Uso de AI, con transparencia

**Qué hizo la AI (Claude Code):** el grueso de la escritura de código y documentación, el perfilado de datos, la ejecución y verificación de pipelines/tests/notebooks, y propuestas de diseño con alternativas argumentadas.

**Qué fue criterio humano — con ejemplos concretos verificables en el resultado:**

- **Detección de riesgos de fuga de información como política, no como accidente.** Cuando se propuso enriquecer los picos con cifras públicas del Buen Fin 2023, se rechazó usar información del mismo periodo que se pronostica ("sería trampa") y se exigió la **regla de vintage** (solo información publicada antes del año objetivo). Esa regla es hoy un test automático.
- **El mecanismo de complemento del prior fue una corrección humana al diseño de la AI:** la primera versión multiplicaba el factor externo sobre la predicción (doble conteo); se pidió explícitamente "identificar la magnitud que ya detecta el modelo y completar la escala" — que es el mecanismo final (`max/min(implícito, externo)` sobre el contrafactual).
- **Cuestionamiento de afirmaciones del análisis:** la afirmación "los montos cuadran sin excepción" fue objetada (los nulos de `amount_cash` no eran verificables); el análisis resultante (nulos MCAR, reconstruibles por identidad contable) quedó en el notebook 01, sección 2.3.
- **Los experimentos de estrés fueron pedidos humanos:** el holdout de 30 días que reveló que el modelo pierde su ventaja fuera de su horizonte de diseño — un hallazgo incómodo que se documentó en lugar de ocultarse (notebook 03).
- **Decisiones de alcance:** pipeline-antes-que-notebooks, la estructura por etapas de proceso, qué modelos alternativos explorar y cuáles descartar.

**Cómo se verificó el trabajo de la AI:** cada pieza generada pasó por (a) los 28 tests — varios escritos específicamente para verificar *promesas de diseño* (perturbar datos futuros no cambia features; alterar el holdout no cambia predicciones); (b) re-ejecución completa de backtests tras cada cambio, con comparación de métricas; (c) revisión de que las conclusiones escritas coincidieran con las tablas (hubo correcciones: una narrativa optimista pre-escrita para el backtest de 30 días fue reescrita cuando los números la contradijeron).

## Decisiones de datos

- Los CSV del reto se incluyen en el repo **sin modificar**: son sintéticos, provistos por el evaluador, sin PII y de tamaño manejable — incluirlos hace el repo reproducible con un solo `uv sync`. Con datos reales de negocio la decisión sería otra: DVC o un object store con credenciales, y solo los `.dvc`/manifiestos en el repo.
- Todo artefacto derivado (`outputs/`) es regenerable por comando; `mlflow.db` + `mlruns/` se versionan como evidencia de tracking.
