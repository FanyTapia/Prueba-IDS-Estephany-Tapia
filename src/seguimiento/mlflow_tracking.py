"""Registro de corridas de entrenamiento en MLflow.

Cubre los cuatro requisitos de tracking: hiperparámetros, métricas,
artefactos y **signature** del modelo. El tracking es local y sin servicios
externos (SQLite, el backend recomendado por MLflow 3): el revisor lo
explora con ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.

Decisión de diseño: se registran dos representaciones del modelo.

1. El estimador interno (``HistGradientBoostingRegressor``) con el flavor
   ``mlflow.sklearn`` — es quien porta la *signature* (esquema de entrada →
   salida) y el ``input_example``, y queda en el Model Registry local.
2. El ``DemandForecaster`` completo (con el complemento de prior y el
   recorte a no-negativos) como artefacto ``model.joblib`` — este es el
   objeto realmente desplegable; el flavor sklearn no puede representar la
   lógica de postproceso.
"""
import logging
import tempfile
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from .. import config

logger = logging.getLogger(__name__)

TRACKING_URI = f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}"
REGISTRY_NAME = "retail-demand-forecaster"


def _sample_for_signature(panel, feature_cols, n: int = 200):
    """Toma una muestra representativa del panel para inferir la signature.

    Las columnas categóricas se convierten a ``str`` porque el esquema de
    MLflow no modela el dtype ``category`` de pandas; la predicción para la
    signature se calcula sobre los dtypes originales (los del estimador).

    Args:
        panel: Panel con features y target.
        feature_cols: Columnas de entrada del modelo.
        n: Tamaño máximo de la muestra.

    Returns:
        Tupla ``(X_original, X_esquema)``: la muestra con dtypes reales y su
        copia con categóricas como texto para el esquema.
    """
    filas = panel[panel[config.TARGET].notna()].tail(n)
    x = filas[feature_cols]
    x_esquema = x.copy()
    for col in x_esquema.select_dtypes(include="category").columns:
        x_esquema[col] = x_esquema[col].astype(str)
    return x, x_esquema


def log_training_run(model, panel, feature_cols: list, params: dict,
                     metrics: dict, artifacts: dict, experiment: str,
                     run_name: str, register: bool = True) -> str:
    """Registra una corrida de entrenamiento completa en MLflow.

    Args:
        model: ``DemandForecaster`` ya entrenado.
        panel: Panel usado (para inferir la signature con datos reales).
        feature_cols: Columnas de entrada del modelo.
        params: Hiperparámetros y configuración a loggear (aplanados).
        metrics: Métricas numéricas de la corrida (holdout/backtest).
        artifacts: Mapeo nombre → ruta de archivos a adjuntar (CSVs de
            métricas, tabla de priors, modelo serializado…).
        experiment: Nombre del experimento MLflow.
        run_name: Nombre legible de la corrida.
        register: Si registrar el modelo en el Model Registry local.

    Returns:
        El ``run_id`` de MLflow de la corrida creada.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for _, path in artifacts.items():
            mlflow.log_artifact(str(path))

        x, x_esquema = _sample_for_signature(panel, feature_cols)
        signature = infer_signature(x_esquema, model.model.predict(x))
        mlflow.sklearn.log_model(
            model.model,
            name="model",
            signature=signature,
            input_example=x_esquema.head(5),
            registered_model_name=REGISTRY_NAME if register else None,
            # cloudpickle: el formato skops (default de MLflow 3) rechaza
            # tipos internos legítimos del HistGradientBoostingRegressor.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # El objeto desplegable completo (estimador + complemento de prior).
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "model.joblib"
            model.save(ruta)
            mlflow.log_artifact(str(ruta), artifact_path="deployable")

        logger.info("Corrida MLflow registrada: run_id=%s", run.info.run_id)
        return run.info.run_id
