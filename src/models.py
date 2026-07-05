"""Modelos: el campeón (gradient boosting) y los dos baselines a vencer."""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import SERIES_KEYS


class DemandForecaster:
    """Envoltorio serializable del modelo de pronóstico.

    Guarda junto al estimador la lista de features y el horizonte con el que
    fue entrenado, de modo que la etapa de predicción no pueda desalinearse
    del entrenamiento.
    """

    def __init__(self, feature_cols: list, horizon: int, params: dict = None):
        self.feature_cols = list(feature_cols)
        self.horizon = horizon
        self.params = dict(params or config.MODEL_PARAMS)
        self.model = HistGradientBoostingRegressor(
            categorical_features="from_dtype", **self.params
        )

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "DemandForecaster":
        self.model.fit(df[self.feature_cols], y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(df[self.feature_cols])
        return np.clip(pred, 0.0, None)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "DemandForecaster":
        return joblib.load(path)


def seasonal_naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Baseline 1 — naïve estacional.

    Promedio de la demanda del mismo día de la semana en las 3 semanas
    disponibles al emitir el pronóstico (rezagos 14/21/28). Si faltan, cae a
    la media móvil de 28 días.
    """
    fallback = df.get(f"{config.TARGET}_ma28")
    return df["lag_dow_mean"].fillna(fallback)


def current_system_forecast(panel: pd.DataFrame, train_mask: pd.Series,
                            test_mask: pd.Series) -> pd.Series:
    """Baseline 2 — el sistema de reposición actual, como comparador.

    Reescala `replenishment_signal` a unidades por serie con un factor
    aprendido SOLO en el periodo de entrenamiento. Ojo: usa la señal emitida
    el mismo día objetivo, algo que un pronóstico real no tendría disponible;
    no es un modelo desplegable sino la vara del sistema vigente.
    """
    train = panel.loc[train_mask]
    m = train.groupby(SERIES_KEYS, observed=True).agg(
        u=(config.TARGET, "mean"), r=("replenishment_signal", "mean")
    )
    scale = (m["u"] / m["r"]).rename("rs_scale").reset_index()

    test = panel.loc[test_mask, SERIES_KEYS + ["replenishment_signal"]].merge(
        scale, on=SERIES_KEYS, how="left"
    )
    pred = (test["replenishment_signal"] * test["rs_scale"]).to_numpy()
    return pd.Series(pred, index=panel.index[test_mask])
