"""Construcción de features para el pronóstico de demanda a HORIZON días.

Regla anti-leakage del diseño: toda feature derivada de la demanda usa
exclusivamente información hasta t − HORIZON (rezagos de 14, 21 y 28 días,
todos múltiplos de 7 para respetar el ciclo semanal). Con eso, cada fila del
panel es un ejemplo honesto de "pronóstico emitido HORIZON días antes" y el
mismo builder sirve, sin cambios, para entrenar, backtestear y predecir.

`replenishment_signal` se excluye deliberadamente de las features: se deriva
de la demanda observada contemporánea (notebook 01, sección 6) y usarla sería
fuga de información. Su rol es de baseline ("sistema actual") en el backtest.
"""
import pandas as pd

from . import config

SERIES_KEYS = ["store_id", "category"]

# Features del calendario del día objetivo: conocidas de antemano por definición.
CALENDAR_COLS = [
    "day_of_week", "week_of_year", "month", "quarter",
    "is_holiday", "is_payday", "is_weekend",
    "is_navidad_season", "is_buen_fin", "is_semana_santa",
]

STORE_NUMERIC = ["size_sqm", "num_checkouts", "opening_year", "has_pharmacy", "has_fuel_station"]
STORE_CATEGORICAL = ["store_format", "region", "socioeconomic_level"]
CATEGORICAL_FEATURES = ["store_id", "category"] + STORE_CATEGORICAL


def complete_grid(tx: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Panel rectangular tienda × categoría × día sobre el rango del calendario.

    Los huecos (días caídos de POS, fechas futuras en predicción) quedan como
    filas con target nulo; así los rezagos por posición dentro de cada serie
    equivalen a rezagos por fecha calendario.
    """
    series = tx[SERIES_KEYS].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(calendar["date"].min(), calendar["date"].max(), freq="D")})
    panel = series.merge(dates, how="cross")
    panel = panel.merge(tx, on=SERIES_KEYS + ["date"], how="left")
    panel = panel.merge(stores, on="store_id", how="left", validate="m:1")
    panel = panel.merge(calendar, on="date", how="left", validate="m:1")
    return panel.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)


def add_demand_features(panel: pd.DataFrame, target: str, horizon: int,
                        lags: list, windows: list) -> tuple:
    """Rezagos y medias móviles de la demanda, todos con información <= t − horizon."""
    bad = [lag for lag in lags if lag < horizon]
    if bad:
        raise ValueError(f"Rezagos {bad} < horizonte {horizon}: fuga de información.")

    g = panel.groupby(SERIES_KEYS, observed=True)[target]
    cols = []
    for lag in lags:
        col = f"{target}_lag{lag}"
        panel[col] = g.shift(lag)
        cols.append(col)

    # Promedio de los rezagos del mismo día de la semana (14/21/28): es a la vez
    # una feature robusta y el pronóstico del baseline naïve estacional.
    panel["lag_dow_mean"] = panel[[f"{target}_lag{lag}" for lag in lags]].mean(axis=1)
    cols.append("lag_dow_mean")

    # Medias móviles sobre la serie desplazada `horizon` días: nivel reciente
    # de la serie conocido al momento de emitir el pronóstico.
    panel["_shifted"] = g.shift(horizon)
    gs = panel.groupby(SERIES_KEYS, observed=True)["_shifted"]
    for w in windows:
        col = f"{target}_ma{w}"
        panel[col] = gs.transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 4)).mean())
        cols.append(col)
    panel[f"{target}_std7"] = gs.transform(lambda s: s.rolling(7, min_periods=3).std())
    cols.append(f"{target}_std7")
    panel = panel.drop(columns="_shifted")
    return panel, cols


def build_features(tx: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame,
                   horizon: int = None) -> tuple:
    """Panel completo con features listo para entrenar/predecir.

    Devuelve (panel, feature_cols). El panel conserva columnas auxiliares
    (target, replenishment_signal, flags de calendario) para evaluación.
    """
    horizon = horizon if horizon is not None else config.HORIZON
    panel = complete_grid(tx, stores, calendar)
    panel, demand_cols = add_demand_features(
        panel, target=config.TARGET, horizon=horizon,
        lags=config.LAG_DAYS, windows=config.ROLLING_WINDOWS,
    )

    # Tendencia suave de largo plazo observada en el EDA.
    panel["t_index"] = (panel["date"] - panel["date"].min()).dt.days

    for col in CATEGORICAL_FEATURES:
        panel[col] = panel[col].astype("category")

    feature_cols = demand_cols + ["t_index"] + CALENDAR_COLS + STORE_NUMERIC + CATEGORICAL_FEATURES
    return panel, feature_cols
