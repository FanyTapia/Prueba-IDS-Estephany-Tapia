"""Modelos: el campeón (gradient boosting) y los dos baselines a vencer."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import config
from ..postproceso.priors import COUNTERFACTUAL_FLAGS
from ..preprocesamiento.features import SERIES_KEYS


class DemandForecaster:
    """Envoltorio serializable del modelo de pronóstico.

    Guarda junto al estimador la lista de features, el horizonte y la
    política de prior con que fue entrenado, de modo que la etapa de
    predicción no pueda desalinearse del entrenamiento.

    El prior de eventos actúa como COMPLEMENTO en inferencia, nunca como
    multiplicador ciego. En días con prior externo (≠ 1) se mide cuánto
    efecto ya detecta el modelo por sí mismo — predicción normal vs.
    contrafactual con los flags del evento apagados — y el exterior solo
    completa la escala faltante, en la dirección que indique el prior::

        factor ≥ 1 (pico):      final = contrafactual × max(implícito, factor)
        factor < 1 (supresión): final = contrafactual × min(implícito, factor)

    (supresión: p. ej. 1 de enero y "cuesta de enero", donde el evento
    REDUCE la demanda). Así el factor externo nunca se duplica sobre lo ya
    aprendido y nunca modera a un modelo que detecta un efecto mayor.

    Args:
        feature_cols: Columnas que entran al estimador (orden incluido).
        horizon: Horizonte de diseño en días.
        params: Hiperparámetros del ``HistGradientBoostingRegressor``; por
            defecto ``config.MODEL_PARAMS`` (pérdida Poisson).
        prior_col: Nombre de la columna con el factor de prior; ``None``
            desactiva el complemento (variante de ablación).
    """

    def __init__(self, feature_cols: list, horizon: int, params: dict = None,
                 prior_col: str = None):
        self.feature_cols = list(feature_cols)
        self.horizon = horizon
        self.params = dict(params or config.MODEL_PARAMS)
        self.prior_col = prior_col
        self.model = HistGradientBoostingRegressor(
            categorical_features="from_dtype", **self.params
        )

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "DemandForecaster":
        """Ajusta el estimador.

        Args:
            df: Panel con al menos ``feature_cols`` (filas de entrenamiento).
            y: Target alineado a ``df`` (sin nulos).

        Returns:
            La propia instancia, para encadenar.
        """
        self.model.fit(df[self.feature_cols], y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice y aplica el complemento de prior en días de evento.

        Args:
            df: Panel con ``feature_cols`` y, si hay prior activo, la
                columna ``prior_col`` y los flags de calendario.

        Returns:
            Array de predicciones no negativas, alineado a ``df``.
        """
        pred = self.model.predict(df[self.feature_cols])
        if self.prior_col is not None and self.prior_col in df.columns:
            prior = df[self.prior_col].to_numpy(dtype=float)
            mask = prior != 1.0
            if mask.any():
                contrafactual = df.loc[mask].copy()
                for col in COUNTERFACTUAL_FLAGS:
                    if col in contrafactual.columns:
                        contrafactual[col] = False
                pred_plain = np.clip(
                    self.model.predict(contrafactual[self.feature_cols]), 1e-6, None
                )
                # efecto que el modelo ya detecta por sí solo
                implied = np.clip(pred[mask] / pred_plain, 0.05, 20.0)
                f = prior[mask]
                factor_final = np.where(f >= 1.0, np.maximum(implied, f),
                                        np.minimum(implied, f))
                pred = pred.copy()
                pred[mask] = pred_plain * factor_final
        return np.clip(pred, 0.0, None)

    def save(self, path) -> None:
        """Serializa la instancia completa (estimador + contrato) con joblib.

        Args:
            path: Ruta destino del ``.joblib``.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "DemandForecaster":
        """Carga una instancia serializada con :meth:`save`.

        Args:
            path: Ruta del ``.joblib``.

        Returns:
            La instancia deserializada.
        """
        return joblib.load(path)


def seasonal_naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Baseline 1 — naïve estacional.

    Promedio de la demanda del mismo día de la semana en las 3 semanas
    disponibles al emitir el pronóstico (rezagos 14/21/28). Si faltan, cae
    a la media móvil de 28 días.

    Args:
        df: Panel con las features de demanda ya construidas.

    Returns:
        Serie de predicciones alineada al índice de ``df``.
    """
    fallback = df.get(f"{config.TARGET}_ma28")
    return df["lag_dow_mean"].fillna(fallback)


def current_system_forecast(panel: pd.DataFrame, train_mask: pd.Series,
                            test_mask: pd.Series) -> pd.Series:
    """Baseline 2 — el sistema de reposición actual, como comparador.

    Reescala ``replenishment_signal`` a unidades por serie con un factor
    aprendido SOLO en el periodo de entrenamiento. Ojo: usa la señal emitida
    el mismo día objetivo, algo que un pronóstico real no tendría disponible;
    no es un modelo desplegable sino la vara del sistema vigente.

    Args:
        panel: Panel completo con target y ``replenishment_signal``.
        train_mask: Máscara booleana del periodo de entrenamiento.
        test_mask: Máscara booleana del periodo a predecir.

    Returns:
        Serie de predicciones indexada como ``panel[test_mask]``.
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
