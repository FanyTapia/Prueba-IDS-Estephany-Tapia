"""Generación del pronóstico operativo a partir de un modelo entrenado."""
import logging

import pandas as pd

from ..entrenamiento.models import DemandForecaster
from ..preprocesamiento.calendar_utils import extend_calendar
from ..preprocesamiento.features import build_features

logger = logging.getLogger(__name__)


def generar_pronostico(model: DemandForecaster, tx: pd.DataFrame,
                       stores: pd.DataFrame, calendar: pd.DataFrame,
                       cutoff=None) -> pd.DataFrame:
    """Emite el pronóstico de los próximos ``model.horizon`` días.

    Trunca la historia al corte (en producción el futuro no existe),
    extiende el calendario con los campos deterministas, reconstruye el
    panel con el MISMO builder del entrenamiento y valida que las features
    coincidan con las del modelo antes de predecir.

    Args:
        model: ``DemandForecaster`` entrenado (define horizonte y features).
        tx: Transacciones históricas.
        stores: Catálogo de tiendas.
        calendar: Calendario oficial (se extiende si hace falta).
        cutoff: Último día con datos a usar; por defecto el máximo de ``tx``.

    Returns:
        DataFrame con columnas ``date``, ``store_id``, ``category`` y
        ``unidades_pronosticadas`` para cada día del horizonte.

    Raises:
        RuntimeError: Si las features del panel no coinciden con las del
            modelo entrenado (desalineación de versiones).
    """
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else tx["date"].max()
    tx = tx[tx["date"] <= cutoff]
    forecast_end = cutoff + pd.Timedelta(days=model.horizon)
    calendar = extend_calendar(calendar, until=forecast_end)
    calendar = calendar[calendar["date"] <= forecast_end]

    panel, feature_cols = build_features(tx, stores, calendar, horizon=model.horizon)
    if feature_cols != model.feature_cols:
        raise RuntimeError("Las features del panel no coinciden con las del modelo entrenado.")

    future = panel[panel["date"] > cutoff].copy()
    future["y_pred"] = model.predict(future)
    logger.info("Pronóstico emitido: %s series × %s días desde %s",
                future[["store_id", "category"]].drop_duplicates().shape[0],
                future["date"].nunique(), cutoff.date())
    return future[["date", "store_id", "category", "y_pred"]].rename(
        columns={"y_pred": "unidades_pronosticadas"}
    )
