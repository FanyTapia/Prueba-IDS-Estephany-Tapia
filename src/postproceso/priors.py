"""Priors de eventos desde fuentes públicas, con disciplina de vintage.

La tabla ``data/external/event_priors.csv`` versiona factores de uplift por
evento × categoría junto con su ``vintage``: el año en que esa información ya
era pública. La regla anti-leakage es dura y simétrica a la de los rezagos:

    para una fecha objetivo en el año Y solo son usables los priors
    con vintage < Y.

Así, el Buen Fin 2023 solo puede apoyarse en lo publicado hasta 2022
(EMEC/ANTAD/BBVA), y lo observado en 2023 se convierte en el prior legítimo
del 2024 — exactamente como viviría el sistema en producción.

El factor NO es una feature ni un multiplicador ciego: el modelo lo aplica
en inferencia como *complemento* (ver ``entrenamiento.models``), completando
solo la escala que no detecta por sí mismo.
"""
import logging

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

PRIOR_COL = "uplift_prior"

# Flags que definen el "día contrafactual" (el mismo día sin el evento
# especial): se apagan para medir cuánto lift detecta el modelo por sí solo.
COUNTERFACTUAL_FLAGS = ["is_buen_fin", "is_navidad_season", "is_holiday"]


def _subeventos_navidad(dates: pd.Series) -> pd.Series:
    """Clasifica cada día de la temporada navideña en su régimen diario.

    La temporada NO es un evento plano. Perfil público (BBVA TPV diario,
    EMEC dic/ene): días ordinarios elevados, tres días explosivos (24, 25 y
    31 de diciembre) y una depresión al inicio de enero (cierre del 1 de
    enero y "cuesta de enero"). Cada régimen tiene su factor en la tabla.

    Args:
        dates: Fechas (dentro de la temporada) a clasificar.

    Returns:
        Serie de nombres de sub-evento alineada al índice de entrada.
    """
    m, d = dates.dt.month, dates.dt.day
    out = pd.Series("navidad_ordinaria", index=dates.index, dtype="object")
    out[(m == 12) & (d == 24)] = "nochebuena"
    out[(m == 12) & (d == 25)] = "navidad_dia"
    out[(m == 12) & (d == 31)] = "fin_de_ano"
    out[(m == 1) & (d == 1)] = "ano_nuevo"
    out[(m == 1) & (d >= 2) & (d <= 6)] = "cuesta_enero"
    return out


def load_event_priors(path=None) -> pd.DataFrame:
    """Carga la tabla de priors de eventos.

    Args:
        path: Ruta alternativa. Por defecto ``config.EVENT_PRIORS_CSV``.

    Returns:
        DataFrame con columnas evento, categoria, factor, vintage y fuente.
    """
    return pd.read_csv(path or config.EVENT_PRIORS_CSV)


def resolve_factor(priors: pd.DataFrame, evento: str, categoria: str, year: int) -> float:
    """Resuelve el factor aplicable a una fecha bajo la regla de vintage.

    Devuelve el factor de vintage más reciente entre los estrictamente
    anteriores al año objetivo. ``categoria`` admite comodín: una fila con
    ``categoria="*"`` aplica a todas, pero la fila específica de la
    categoría tiene precedencia.

    Args:
        priors: Tabla de priors (ver :func:`load_event_priors`).
        evento: Nombre del (sub)evento, p. ej. ``"buen_fin"``.
        categoria: Categoría de producto de la fila objetivo.
        year: Año calendario de la fecha a pronosticar.

    Returns:
        Factor multiplicativo; ``1.0`` si no hay prior usable.
    """
    usable = priors[
        (priors["evento"] == evento)
        & (priors["categoria"].isin([categoria, "*"]))
        & (priors["vintage"] < year)
    ]
    if usable.empty:
        return 1.0
    exacta = usable[usable["categoria"] == categoria]
    elegida = exacta if not exacta.empty else usable
    return float(elegida.sort_values("vintage")["factor"].iloc[-1])


def attach_uplift_prior(panel: pd.DataFrame, priors: pd.DataFrame = None) -> pd.DataFrame:
    """Agrega la columna ``uplift_prior`` al panel (1.0 en días sin evento).

    Cada día de evento recibe el factor de su (sub)evento: la temporada
    navideña se desglosa en régimen diario (:func:`_subeventos_navidad`) y
    el Buen Fin tiene precedencia si coincidiera con la temporada.

    Args:
        panel: Panel con columnas ``date``, ``category`` y los flags de
            calendario (``is_navidad_season``, ``is_buen_fin``).
        priors: Tabla de priors; si se omite se carga la del proyecto (con
            advertencia y factor neutro si el CSV no existe).

    Returns:
        El mismo panel con la columna ``uplift_prior`` poblada.
    """
    if priors is None:
        try:
            priors = load_event_priors()
        except FileNotFoundError:
            logger.warning("No existe %s: uplift_prior=1.0 en todo el panel.",
                           config.EVENT_PRIORS_CSV)
            panel[PRIOR_COL] = 1.0
            return panel

    panel[PRIOR_COL] = 1.0
    evento = pd.Series(pd.NA, index=panel.index, dtype="object")
    nav = panel["is_navidad_season"].astype(bool)
    evento[nav] = _subeventos_navidad(panel.loc[nav, "date"])
    bf = panel["is_buen_fin"].astype(bool)
    evento[bf] = "buen_fin"

    con_evento = evento.notna()
    sel = panel.loc[con_evento, ["category", "date"]].copy()
    sel["evento"] = evento[con_evento]
    sel["year"] = sel["date"].dt.year
    for (ev, cat, yr), idx in sel.groupby(["evento", "category", "year"],
                                          observed=True).groups.items():
        panel.loc[idx, PRIOR_COL] = resolve_factor(priors, ev, str(cat), int(yr))
    return panel
