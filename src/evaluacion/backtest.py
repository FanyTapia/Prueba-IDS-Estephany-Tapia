"""Backtest walk-forward: campeón vs. baselines, segmentado pico / normal."""
import logging

import pandas as pd

from .. import config
from ..entrenamiento.models import (
    DemandForecaster,
    current_system_forecast,
    seasonal_naive_forecast,
)
from ..postproceso.priors import PRIOR_COL
from .metrics import evaluate, paired_wape_diff_ci

logger = logging.getLogger(__name__)

MODEL_CHAMPION = "gradient_boosting"
MODEL_CHAMPION_PRIOR = "gradient_boosting_prior"
MODEL_NAIVE = "naive_estacional"
MODEL_CURRENT = "sistema_actual"


def run_backtest(panel: pd.DataFrame, feature_cols: list, folds: list,
                 params: dict = None) -> pd.DataFrame:
    """Corre todos los folds del walk-forward y reúne las predicciones.

    Por cada fold entrena el campeón con datos hasta el origen y genera las
    predicciones de los 4 competidores sobre exactamente las mismas celdas
    de la ventana de prueba.

    Args:
        panel: Panel con features (salida de ``build_features``).
        feature_cols: Columnas que entran al modelo.
        folds: Cortes temporales (ver ``splits.make_folds``).
        params: Hiperparámetros del campeón; por defecto los de ``config``.

    Returns:
        DataFrame en formato largo: una fila por (fold, fecha, tienda,
        categoría, modelo) con ``y_true``, ``y_pred`` y ``segmento``.
    """
    results = []
    for fold in folds:
        train_mask, test_mask = fold.masks(panel["date"])
        fit_mask = train_mask & panel[config.TARGET].notna()
        logger.info(
            "Fold origen=%s | train=%s filas | test=%s filas (%s -> %s)",
            fold.origin.date(), fit_mask.sum(), test_mask.sum(),
            fold.test_start.date(), fold.test_end.date(),
        )

        champion = DemandForecaster(feature_cols, fold.horizon, params)
        champion.fit(panel.loc[fit_mask], panel.loc[fit_mask, config.TARGET])

        # El prior actúa solo en inferencia (complemento): ambas variantes
        # comparten exactamente el mismo modelo entrenado.
        champion_prior = DemandForecaster(feature_cols, fold.horizon, params,
                                          prior_col=PRIOR_COL)
        champion_prior.model = champion.model

        test = panel.loc[test_mask]
        preds = {
            MODEL_CHAMPION: champion.predict(test),
            MODEL_CHAMPION_PRIOR: champion_prior.predict(test),
            MODEL_NAIVE: seasonal_naive_forecast(test).to_numpy(),
            MODEL_CURRENT: current_system_forecast(panel, train_mask, test_mask).to_numpy(),
        }
        base = test[["date", "store_id", "category"]].copy()
        base["y_true"] = test[config.TARGET].to_numpy()
        base["segmento"] = (test["is_buen_fin"] | test["is_navidad_season"]).map(
            {True: "pico", False: "normal"}
        )
        base["fold_origin"] = fold.origin.date().isoformat()
        for name, yhat in preds.items():
            out = base.copy()
            out["modelo"] = name
            out["y_pred"] = yhat
            results.append(out)

    return pd.concat(results, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Resume métricas por modelo y segmento, con IC 95% bootstrap.

    Args:
        predictions: Salida de :func:`run_backtest`.

    Returns:
        DataFrame con una fila por (modelo, segmento) incluyendo total;
        columnas WAPE/MAE/sesgo y sus intervalos.
    """
    total = evaluate(predictions, by=["modelo"], ci=True)
    total.insert(1, "segmento", "total")
    seg = evaluate(predictions, by=["modelo", "segmento"], ci=True)
    out = pd.concat([total, seg], ignore_index=True)
    order = pd.CategoricalIndex(out["segmento"], categories=["total", "pico", "normal"])
    return out.assign(segmento=order).sort_values(["modelo", "segmento"]).reset_index(drop=True)


def summarize_paired(predictions: pd.DataFrame,
                     champion: str = MODEL_CHAMPION_PRIOR) -> pd.DataFrame:
    """Calcula el ΔWAPE pareado del campeón contra cada competidor.

    Es la prueba formal de "le gana": IC de la diferencia sobre los mismos
    días remuestreados juntos, no superposición de intervalos marginales.

    Args:
        predictions: Salida de :func:`run_backtest`.
        champion: Nombre del modelo de referencia.

    Returns:
        DataFrame con ΔWAPE, IC 95% y bandera de significancia por rival y
        segmento (incluye total).
    """
    rivales = [MODEL_CURRENT, MODEL_NAIVE, MODEL_CHAMPION]
    tablas = []
    for rival in rivales:
        total = paired_wape_diff_ci(predictions, champion, rival)
        total.insert(1, "segmento", "total")
        seg = paired_wape_diff_ci(predictions, champion, rival, by=["segmento"])
        tablas.append(pd.concat([total, seg], ignore_index=True))
    return pd.concat(tablas, ignore_index=True)
