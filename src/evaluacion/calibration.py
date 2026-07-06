"""Calibración del dato externo contra el observado.

Herramienta de DIAGNÓSTICO, deliberadamente separada del pipeline del modelo:
nada en `features.py`/`models.py` importa este módulo, y `event_reference.csv`
NO alimenta ninguna predicción. Su único fin es responder una pregunta de
riesgo: *¿qué tan confiable es apoyarse en factores externos para un evento
que no podemos observar (p. ej. el primer Buen Fin)?*

La respuesta se obtiene validando las fuentes externas contra los eventos que
SÍ están en los datos: si el factor externo acierta en Independencia, Muertos
y quincena, ganamos confianza en el factor externo del Buen Fin; si falla en
otros (festivos de 'puente', Día de las Madres), aprendemos dónde el dato
externo es peligroso y no debe usarse a ciegas.
"""
import pandas as pd

from .. import config

# Definición de cada evento como una máscara sobre el calendario. Se mantiene
# aquí (no en priors.py) porque estos eventos son solo para calibración.
EVENT_DEFINITIONS = {
    "semana_santa": lambda cal: cal["is_semana_santa"].astype(bool),
    "dia_constitucion": lambda cal: cal["holiday_name"].str.contains("Constitución", na=False),
    "benito_juarez": lambda cal: cal["holiday_name"].str.contains("Juárez", na=False),
    "dia_trabajo": lambda cal: cal["holiday_name"].str.contains("Trabajo", na=False),
    "independencia": lambda cal: cal["holiday_name"].str.contains("Independencia", na=False),
    "dia_muertos": lambda cal: cal["holiday_name"].str.contains("Muertos", na=False),
    "virgen_guadalupe": lambda cal: cal["holiday_name"].str.contains("Guadalupe", na=False),
    "quincena": lambda cal: cal["is_payday"] & ~cal["is_holiday"] & ~cal["is_navidad_season"],
    "dia_madres": lambda cal: (cal["month"] == 5) & (cal["date"].dt.day == 10),
    "dia_nino": lambda cal: (cal["month"] == 4) & (cal["date"].dt.day == 30),
}

# Eventos "limpios" que definen la línea base (día típico sin efecto especial).
_BASELINE_EXCLUDE = ["is_holiday", "is_payday", "is_buen_fin",
                     "is_navidad_season", "is_semana_santa"]


def load_event_reference(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.EVENT_REFERENCE_CSV)


def observed_event_factors(tx: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Factor observado de cada evento = venta media en los días del evento
    dividida por la línea base del mismo día de la semana en días limpios.

    Se calcula por serie (tienda × categoría × día de semana) y luego se
    promedia, para no dejar que las tiendas grandes dominen el factor.
    """
    m = tx.merge(calendar, on="date", how="left")
    clean = ~m[_BASELINE_EXCLUDE].any(axis=1)
    base = (m[clean].groupby(["store_id", "category", "day_of_week"], observed=True)
            [config.TARGET].mean().rename("_base"))
    m = m.merge(base, on=["store_id", "category", "day_of_week"], how="left")
    m["_idx"] = m[config.TARGET] / m["_base"]

    rows = []
    for evento, definicion in EVENT_DEFINITIONS.items():
        mask = definicion(m).fillna(False) & m["_idx"].notna()
        sub = m[mask]
        if sub.empty:
            continue
        rows.append({
            "evento": evento,
            "factor_observado": round(sub["_idx"].mean(), 2),
            "n_dias": sub["date"].nunique(),
            "n_filas": len(sub),
        })
    return pd.DataFrame(rows)


def calibration_report(tx: pd.DataFrame, calendar: pd.DataFrame,
                       reference: pd.DataFrame = None) -> pd.DataFrame:
    """Compara factor externo vs. observado y clasifica el riesgo de confiar
    en el dato externo para cada evento.

    `brecha_rel` = observado / externo − 1  (0 = el externo acierta).
    `riesgo`: confiable (|brecha| ≤ 15%), moderado (≤ 40%) o alto (> 40%),
    y marca aparte cuando ambos apuntan en direcciones opuestas (uno sube,
    el otro baja respecto de 1.0), que es el caso más peligroso.
    """
    reference = reference if reference is not None else load_event_reference()
    obs = observed_event_factors(tx, calendar)
    rep = reference.merge(obs, on="evento", how="left")

    rep["brecha_rel"] = (rep["factor_observado"] / rep["factor_externo"] - 1).round(2)
    signo_opuesto = ((rep["factor_externo"] - 1) * (rep["factor_observado"] - 1)) < 0

    def clasifica(row, opuesto):
        if pd.isna(row["factor_observado"]):
            return "sin_dato"
        if opuesto:
            return "DIRECCIÓN OPUESTA"
        g = abs(row["brecha_rel"])
        return "confiable" if g <= 0.15 else "moderado" if g <= 0.40 else "alto"

    rep["riesgo"] = [clasifica(r, o) for (_, r), o in zip(rep.iterrows(), signo_opuesto)]
    cols = ["evento", "nombre", "factor_externo", "factor_observado",
            "brecha_rel", "riesgo", "n_dias", "fuente"]
    return rep[cols].sort_values("brecha_rel", key=lambda s: s.abs(), ascending=False) \
                    .reset_index(drop=True)
