"""La prueba más importante del proyecto: ninguna feature de demanda puede
usar información posterior a t − HORIZON (anti-leakage por construcción)."""
import pandas as pd
import pytest

from src import config
from src.features import add_demand_features, build_features


def test_no_leakage_perturbando_el_futuro_cercano(toy_data):
    """Si altero la demanda de los últimos HORIZON−1 días antes de t, las
    features en t NO deben cambiar: al emitir el pronóstico esos días aún
    no habían ocurrido."""
    tx, stores, cal = toy_data
    panel_a, cols = build_features(tx, stores, cal)

    t = tx["date"].max()
    ventana_no_disponible = tx["date"] > t - pd.Timedelta(days=config.HORIZON)
    tx_b = tx.copy()
    tx_b.loc[ventana_no_disponible, "units_sold"] += 10_000

    panel_b, _ = build_features(tx_b, stores, cal)

    fila_a = panel_a.loc[panel_a["date"] == t, cols].reset_index(drop=True)
    fila_b = panel_b.loc[panel_b["date"] == t, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(fila_a, fila_b)


def test_rezago_menor_al_horizonte_es_rechazado(toy_data):
    tx, stores, cal = toy_data
    from src.features import complete_grid
    panel = complete_grid(tx, stores, cal)
    with pytest.raises(ValueError, match="fuga"):
        add_demand_features(panel, target="units_sold", horizon=14,
                            lags=[7, 14], windows=[7])


def test_rezagos_son_por_fecha_calendario(toy_data):
    """Con un hueco en los datos (día caído de POS), el rezago de 14 días
    debe seguir apuntando a la fecha correcta, no 14 posiciones atrás."""
    tx, stores, cal = toy_data
    tx_con_hueco = tx[tx["date"] != pd.Timestamp("2023-02-10")]

    panel, _ = build_features(tx_con_hueco, stores, cal)
    s1 = panel[(panel["store_id"] == "S1") & (panel["category"] == "Abarrotes")]

    objetivo = s1.loc[s1["date"] == pd.Timestamp("2023-02-24"), "units_sold_lag14"]
    assert objetivo.isna().all()  # 2023-02-10 no existe: el rezago debe ser nulo

    vecino = s1.loc[s1["date"] == pd.Timestamp("2023-02-25"), "units_sold_lag14"]
    esperado = tx[(tx["store_id"] == "S1") & (tx["category"] == "Abarrotes")
                  & (tx["date"] == pd.Timestamp("2023-02-11"))]["units_sold"].iloc[0]
    assert vecino.iloc[0] == esperado
