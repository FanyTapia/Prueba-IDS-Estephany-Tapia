"""Evaluación de holdout final: los últimos días del dataset se apartan,
el modelo se entrena sin verlos nunca y se pronostican de una sola emisión.

Diferencia metodológica con el backtest walk-forward: aquí los datos
apartados se eliminan FÍSICAMENTE de las transacciones antes de construir
features — igual que en producción, donde el futuro no existe — de modo que
ninguna estadística (rezagos, medias móviles) puede tocarlos ni por error.

Los días más allá de cutoff + 14 tienen el rezago de 14 días nulo (esa fecha
cae dentro del holdout y no está disponible); el modelo lo maneja nativamente
con el resto de rezagos y medias. Son pronósticos a más días que el horizonte
de diseño y se reportan segmentados para verlo.
"""
import logging

import pandas as pd

from .. import config
from ..entrenamiento.models import DemandForecaster, seasonal_naive_forecast
from ..postproceso.priors import PRIOR_COL
from ..preprocesamiento.features import SERIES_KEYS, build_features
from .metrics import evaluate, paired_wape_diff_ci

logger = logging.getLogger(__name__)

MODEL_CHAMPION = "gradient_boosting"
MODEL_CHAMPION_PRIOR = "gradient_boosting_prior"
MODEL_NAIVE = "naive_estacional"
MODEL_CURRENT = "sistema_actual"


def run_holdout(tx: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame,
                cutoff, params: dict = None) -> pd.DataFrame:
    """Entrena hasta el corte y pronostica todos los días posteriores.

    Los días posteriores al corte no existen para el modelo (se eliminan
    de las transacciones antes de construir features); solo reaparecen al
    final como valores reales contra los que se califica.

    Args:
        tx: Transacciones completas (incluyen los días a apartar).
        stores: Catálogo de tiendas.
        calendar: Calendario que cubre también los días apartados.
        cutoff: Última fecha visible para el modelo.
        params: Hiperparámetros del campeón; por defecto los de ``config``.

    Returns:
        DataFrame en formato largo con las predicciones de los 4
        competidores y el real apartado (``y_true``), más ``tramo`` según
        distancia al corte.

    Raises:
        ValueError: Si no hay días posteriores al corte que apartar.
    """
    cutoff = pd.Timestamp(cutoff)
    holdout = tx[tx["date"] > cutoff]
    if holdout.empty:
        raise ValueError(f"No hay días posteriores a {cutoff.date()} para apartar.")
    tx_train = tx[tx["date"] <= cutoff]
    logger.info("Apartados %s días (%s -> %s); entrenando solo con datos <= %s",
                holdout["date"].nunique(), holdout["date"].min().date(),
                holdout["date"].max().date(), cutoff.date())

    # Las features se construyen SOLO con las transacciones de entrenamiento;
    # el calendario completo aporta las fechas futuras como filas vacías.
    panel, feature_cols = build_features(tx_train, stores, calendar)
    fit_mask = (panel["date"] <= cutoff) & panel[config.TARGET].notna()

    champion = DemandForecaster(feature_cols, config.HORIZON, params)
    champion.fit(panel.loc[fit_mask], panel.loc[fit_mask, config.TARGET])
    champion_prior = DemandForecaster(feature_cols, config.HORIZON, params,
                                      prior_col=PRIOR_COL)
    champion_prior.model = champion.model  # el prior actúa solo en inferencia

    future = panel[panel["date"] > cutoff].copy()

    # Sistema actual como comparador: escala aprendida SOLO en entrenamiento,
    # aplicada a la señal que el sistema emitió durante los días apartados.
    m = panel.loc[fit_mask].groupby(SERIES_KEYS, observed=True).agg(
        u=(config.TARGET, "mean"), r=("replenishment_signal", "mean"))
    scale = (m["u"] / m["r"]).rename("rs_scale").reset_index()
    rs = holdout[SERIES_KEYS + ["date", "replenishment_signal"]].rename(
        columns={"replenishment_signal": "rs_holdout"})
    fut = future.merge(rs, on=SERIES_KEYS + ["date"], how="left") \
                .merge(scale, on=SERIES_KEYS, how="left")

    preds = {
        MODEL_CHAMPION: champion.predict(future),
        MODEL_CHAMPION_PRIOR: champion_prior.predict(future),
        MODEL_NAIVE: seasonal_naive_forecast(future).to_numpy(),
        MODEL_CURRENT: (fut["rs_holdout"] * fut["rs_scale"]).to_numpy(),
    }

    real = holdout[SERIES_KEYS + ["date", config.TARGET]].rename(
        columns={config.TARGET: "y_true"})
    base = future[["date", "store_id", "category"]].merge(
        real, on=SERIES_KEYS + ["date"], how="left")
    base["dias_desde_cutoff"] = (base["date"] - cutoff).dt.days
    base["tramo"] = base["dias_desde_cutoff"].map(
        lambda d: "días 1-7" if d <= 7 else "días 8+")

    out = []
    for nombre, yhat in preds.items():
        p = base.copy()
        p["modelo"] = nombre
        p["y_pred"] = yhat
        out.append(p)
    return pd.concat(out, ignore_index=True)


def summarize_holdout(predictions: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por modelo y tramo de distancia al corte, con IC 95%.

    Args:
        predictions: Salida de :func:`run_holdout`.

    Returns:
        DataFrame con una fila por (modelo, tramo) incluyendo total.
    """
    total = evaluate(predictions, by=["modelo"], ci=True)
    total.insert(1, "tramo", "total")
    tramos = evaluate(predictions, by=["modelo", "tramo"], ci=True)
    out = pd.concat([total, tramos], ignore_index=True)
    orden = pd.CategoricalIndex(out["tramo"], categories=["total", "días 1-7", "días 8+"])
    return out.assign(tramo=orden).sort_values(["modelo", "tramo"]).reset_index(drop=True)


def summarize_holdout_paired(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ΔWAPE pareado del campeón contra cada rival en el holdout.

    Args:
        predictions: Salida de :func:`run_holdout`.

    Returns:
        DataFrame con ΔWAPE, IC 95% y significancia por rival.
    """
    tablas = []
    for rival in [MODEL_CURRENT, MODEL_NAIVE]:
        tablas.append(paired_wape_diff_ci(predictions, MODEL_CHAMPION_PRIOR, rival))
    return pd.concat(tablas, ignore_index=True)
