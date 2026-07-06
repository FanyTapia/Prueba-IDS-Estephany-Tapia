"""La calibración es diagnóstico puro: mide factores observados y los compara
con los externos, sin tocar el pipeline de predicción."""
import pandas as pd

from src.evaluacion.calibration import calibration_report, observed_event_factors


def _toy_con_evento(toy_data):
    """Añade un festivo (Día del Trabajo, 1 may) con demanda al doble."""
    tx, stores, cal = toy_data
    tx, cal = tx.copy(), cal.copy()
    dia = pd.Timestamp("2023-03-13")  # dentro del rango de toy_data (90 días)
    cal["holiday_name"] = None
    cal["is_holiday"] = False
    cal.loc[cal.date == dia, ["holiday_name", "is_holiday"]] = ["Día del Trabajo", True]
    tx.loc[tx.date == dia, "units_sold"] *= 2
    return tx, cal, dia


def test_factor_observado_recupera_el_uplift(toy_data):
    tx, cal, dia = _toy_con_evento(toy_data)
    obs = observed_event_factors(tx, cal).set_index("evento")
    assert "dia_trabajo" in obs.index
    # demanda al doble en el evento -> factor observado ~2
    assert 1.7 < obs.loc["dia_trabajo", "factor_observado"] < 2.3


def test_reporte_detecta_direccion_opuesta(toy_data):
    tx, cal, dia = _toy_con_evento(toy_data)
    # externo dice que el evento REDUCE la venta (0.5) pero se observa ~2x
    ref = pd.DataFrame([{
        "evento": "dia_trabajo", "nombre": "Día del Trabajo",
        "factor_externo": 0.5, "fuente": "test",
    }])
    rep = calibration_report(tx, cal, ref)
    assert rep.loc[0, "riesgo"] == "DIRECCIÓN OPUESTA"


def test_reporte_marca_confiable_cuando_coincide(toy_data):
    tx, cal, dia = _toy_con_evento(toy_data)
    ref = pd.DataFrame([{
        "evento": "dia_trabajo", "nombre": "Día del Trabajo",
        "factor_externo": 2.0, "fuente": "test",
    }])
    rep = calibration_report(tx, cal, ref)
    assert rep.loc[0, "riesgo"] == "confiable"
