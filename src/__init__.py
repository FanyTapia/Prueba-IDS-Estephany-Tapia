"""Sistema de pronóstico de demanda para reposición inteligente.

El paquete está organizado por etapa del proceso:

- ``preprocesamiento``: carga tipada, calidad de datos, panel y features.
- ``entrenamiento``: el modelo campeón y los baselines.
- ``inferencia``: generación del pronóstico operativo.
- ``postproceso``: priors de eventos externos (regla de vintage, complemento).
- ``evaluacion``: métricas con IC, backtest walk-forward, holdout, calibración.
- ``seguimiento``: artefactos por corrida y tracking en MLflow.
- ``pipelines``: puntos de entrada de terminal (``retail-*``).

Transversales en la raíz: ``config`` (parámetros) y ``viz`` (estética).
"""
