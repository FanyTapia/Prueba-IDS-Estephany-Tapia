"""Extensión determinista del calendario para pronosticar más allá de los datos."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Festivos de fecha fija (mes, día). Los de fecha móvil (Buen Fin, Semana
# Santa) no se extrapolan: se marcan False y se advierte si el horizonte
# cae en temporada donde podrían ocurrir.
FIXED_HOLIDAYS = {
    (1, 1): "Año Nuevo",
    (5, 1): "Día del Trabajo",
    (9, 16): "Independencia",
    (12, 25): "Navidad",
}

SEASON_BY_MONTH = {
    12: "Invierno", 1: "Invierno", 2: "Invierno",
    3: "Primavera", 4: "Primavera", 5: "Primavera",
    6: "Verano", 7: "Verano", 8: "Verano",
    9: "Otoño", 10: "Otoño", 11: "Otoño",
}


def extend_calendar(calendar: pd.DataFrame, until) -> pd.DataFrame:
    """Agrega filas futuras al calendario con los campos derivables de la fecha.

    Los campos estructurales (día de semana, quincena, temporada navideña,
    festivos fijos) son deterministas. `is_buen_fin` y `is_semana_santa` son
    de fecha móvil y quedan en False fuera del calendario oficial provisto.
    """
    until = pd.Timestamp(until)
    last = calendar["date"].max()
    if until <= last:
        return calendar

    dates = pd.date_range(last + pd.Timedelta(days=1), until, freq="D")
    ext = pd.DataFrame({"date": dates})
    ext["day_of_week"] = ext["date"].dt.dayofweek
    ext["day_name"] = ext["date"].dt.day_name()
    ext["week_of_year"] = ext["date"].dt.isocalendar().week.astype(int)
    ext["month"] = ext["date"].dt.month
    ext["year"] = ext["date"].dt.year
    ext["quarter"] = ext["date"].dt.quarter
    ext["season"] = ext["month"].map(SEASON_BY_MONTH)
    md = list(zip(ext["month"], ext["date"].dt.day))
    ext["is_holiday"] = [k in FIXED_HOLIDAYS for k in md]
    ext["holiday_name"] = [FIXED_HOLIDAYS.get(k) for k in md]
    ext["is_payday"] = (ext["date"].dt.day == 15) | ext["date"].dt.is_month_end
    ext["is_weekend"] = ext["day_of_week"] >= 5
    ext["is_navidad_season"] = ((ext["month"] == 12) & (ext["date"].dt.day >= 15)) | (
        (ext["month"] == 1) & (ext["date"].dt.day <= 6)
    )
    ext["is_buen_fin"] = False
    ext["is_semana_santa"] = False

    if ext["month"].isin([3, 4, 11]).any():
        logger.warning(
            "El horizonte extendido cae en meses con eventos de fecha móvil "
            "(Buen Fin / Semana Santa) que no se pueden extrapolar y quedaron en False."
        )
    return pd.concat([calendar, ext], ignore_index=True)
