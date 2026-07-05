"""Backtest walk-forward: campeón vs. baselines, segmentado pico / normal."""
import logging

import pandas as pd

from . import config
from .metrics import evaluate
from .models import DemandForecaster, current_system_forecast, seasonal_naive_forecast

logger = logging.getLogger(__name__)

MODEL_CHAMPION = "gradient_boosting"
MODEL_NAIVE = "naive_estacional"
MODEL_CURRENT = "sistema_actual"


def run_backtest(panel: pd.DataFrame, feature_cols: list, folds: list,
                 params: dict = None) -> pd.DataFrame:
    """Corre todos los folds y devuelve las predicciones en formato largo:
    una fila por (fold, fecha, tienda, categoría, modelo)."""
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

        test = panel.loc[test_mask]
        preds = {
            MODEL_CHAMPION: champion.predict(test),
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
    """Métricas por modelo, total y por segmento (pico / normal)."""
    total = evaluate(predictions, by=["modelo"])
    total.insert(1, "segmento", "total")
    seg = evaluate(predictions, by=["modelo", "segmento"])
    out = pd.concat([total, seg], ignore_index=True)
    order = pd.CategoricalIndex(out["segmento"], categories=["total", "pico", "normal"])
    return out.assign(segmento=order).sort_values(["modelo", "segmento"]).reset_index(drop=True)
