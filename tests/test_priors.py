"""La regla de vintage es la garantía anti-leakage de los priors externos:
información publicada en el año Y solo puede usarse para fechas > Y."""
import numpy as np
import pandas as pd

from src.entrenamiento.models import DemandForecaster
from src.postproceso.priors import PRIOR_COL, attach_uplift_prior, resolve_factor
from src.preprocesamiento.features import build_features

PRIORS = pd.DataFrame({
    "evento": ["buen_fin", "buen_fin", "navidad_ordinaria", "nochebuena", "ano_nuevo"],
    "categoria": ["Abarrotes", "Abarrotes", "*", "*", "*"],
    "factor": [2.0, 3.7, 1.5, 3.0, 0.5],
    "vintage": [2022, 2023, 2022, 2022, 2022],
    "fuente": ["externa", "observado 2023", "externa", "externa", "externa"],
})


def test_regla_de_vintage():
    # Para 2023 solo es usable el prior 2022 (el 3.7 se observó EN 2023: prohibido)
    assert resolve_factor(PRIORS, "buen_fin", "Abarrotes", 2023) == 2.0
    # Para 2024 ya es legítimo usar lo observado en 2023
    assert resolve_factor(PRIORS, "buen_fin", "Abarrotes", 2024) == 3.7
    # Sin prior publicado antes del año objetivo -> neutro
    assert resolve_factor(PRIORS, "buen_fin", "Abarrotes", 2022) == 1.0
    assert resolve_factor(PRIORS, "buen_fin", "Ropa", 2024) == 1.0


def test_comodin_de_categoria():
    # La fila "*" aplica a cualquier categoría…
    assert resolve_factor(PRIORS, "nochebuena", "Electronica", 2023) == 3.0
    # …pero una fila específica de la categoría tiene precedencia
    pri = pd.concat([PRIORS, pd.DataFrame([{
        "evento": "nochebuena", "categoria": "Electronica", "factor": 2.0,
        "vintage": 2022, "fuente": "test"}])], ignore_index=True)
    assert resolve_factor(pri, "nochebuena", "Electronica", 2023) == 2.0
    assert resolve_factor(pri, "nochebuena", "Abarrotes", 2023) == 3.0


def test_attach_asigna_subevento_por_fecha():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-11-17", "2023-11-16", "2023-12-20",
                                "2023-12-24", "2024-01-01"]),
        "category": ["Abarrotes"] * 5,
        "is_buen_fin": [True, False, False, False, False],
        "is_navidad_season": [False, False, True, True, True],
    })
    out = attach_uplift_prior(panel, PRIORS)
    #                                buen_fin, nada, ordinaria, nochebuena, año nuevo (supresión)
    assert out[PRIOR_COL].tolist() == [2.0, 1.0, 1.5, 3.0, 0.5]


def test_prior_es_neutro_sin_eventos(toy_data):
    """Con calendario sin eventos (factor 1.0 en todo el panel), el modelo
    con prior debe producir exactamente lo mismo que el modelo sin prior."""
    tx, stores, cal = toy_data
    panel, cols = build_features(tx, stores, cal)
    assert (panel[PRIOR_COL] == 1.0).all()

    fit = panel[panel.units_sold.notna() & (panel.date <= "2023-03-17")]
    test = panel[panel.date > "2023-03-17"]
    params = {"loss": "poisson", "max_iter": 30, "random_state": 0}

    a = DemandForecaster(cols, 14, params).fit(fit, fit.units_sold).predict(test)
    b = DemandForecaster(cols, 14, params, prior_col=PRIOR_COL) \
        .fit(fit, fit.units_sold).predict(test)
    np.testing.assert_allclose(a, b)


def test_complemento_completa_la_escala_en_evento_nunca_visto(toy_data):
    """Evento ausente del entrenamiento: el modelo no detecta lift propio
    (implícito ≈ 1), así que el exterior debe completar la escala completa
    (~factor) respecto al día equivalente sin evento."""
    tx, stores, cal = toy_data
    cal = cal.copy()
    dia_evento = pd.Timestamp("2023-03-30")
    cal.loc[cal.date == dia_evento, "is_buen_fin"] = True

    priors = pd.DataFrame({
        "evento": ["buen_fin"] * 2, "categoria": ["Abarrotes", "Bebidas"],
        "factor": [3.0, 3.0], "vintage": [2022, 2022], "fuente": ["test"] * 2,
    })
    panel, cols = build_features(tx, stores, cal)
    panel = attach_uplift_prior(panel.drop(columns=PRIOR_COL), priors)

    fit = panel[panel.units_sold.notna() & (panel.date <= "2023-03-16")]
    model = DemandForecaster(cols, 14, {"loss": "poisson", "max_iter": 30, "random_state": 0},
                             prior_col=PRIOR_COL)
    model.fit(fit, fit.units_sold)

    test = panel[panel.date.isin([dia_evento, dia_evento - pd.Timedelta(days=7)])]
    pred = test.assign(y_pred=model.predict(test))
    s1 = pred[pred.store_id == "S1"].set_index("date")
    ratio = (s1.loc[dia_evento, "y_pred"].mean()
             / s1.loc[dia_evento - pd.Timedelta(days=7), "y_pred"].mean())
    # mismo día de la semana, sin lift propio -> el cociente ≈ el factor externo
    assert 2.5 < ratio < 3.5


def test_complemento_no_reduce_al_modelo_que_ya_detecta_mas(toy_data):
    """Evento presente en el entrenamiento con lift ~4x y prior externo de
    solo 1.5: el complemento debe respetar la magnitud que el modelo ya
    detecta (final = max(implícito, externo)), nunca reducirla."""
    tx, stores, cal = toy_data
    cal = cal.copy()
    tx = tx.copy()

    # 6 días de "evento" en el entrenamiento con demanda 4x
    dias_train = pd.to_datetime(["2023-02-02", "2023-02-09", "2023-02-16",
                                 "2023-02-23", "2023-03-02", "2023-03-09"])
    dia_test = pd.Timestamp("2023-03-30")
    marcados = list(dias_train) + [dia_test]
    cal.loc[cal.date.isin(marcados), "is_buen_fin"] = True
    tx.loc[tx.date.isin(dias_train), "units_sold"] *= 4

    priors = pd.DataFrame({
        "evento": ["buen_fin"] * 2, "categoria": ["Abarrotes", "Bebidas"],
        "factor": [1.5, 1.5], "vintage": [2022, 2022], "fuente": ["test"] * 2,
    })
    panel, cols = build_features(tx, stores, cal)
    panel = attach_uplift_prior(panel.drop(columns=PRIOR_COL), priors)

    fit = panel[panel.units_sold.notna() & (panel.date <= "2023-03-16")]
    model = DemandForecaster(cols, 14, {"loss": "poisson", "max_iter": 60, "random_state": 0},
                             prior_col=PRIOR_COL)
    model.fit(fit, fit.units_sold)

    test = panel[panel.date.isin([dia_test, dia_test - pd.Timedelta(days=7)])]
    pred = test.assign(y_pred=model.predict(test))
    s1 = pred[pred.store_id == "S1"].set_index("date")
    ratio = (s1.loc[dia_test, "y_pred"].mean()
             / s1.loc[dia_test - pd.Timedelta(days=7), "y_pred"].mean())
    # si el complemento "aplastara" al modelo hacia el 1.5 externo, ratio ~1.5;
    # lo correcto es conservar el ~4x aprendido
    assert ratio > 2.5


def test_prior_de_supresion_reduce_la_prediccion(toy_data):
    """Factor < 1 (p. ej. Año Nuevo): el evento SUPRIME demanda. Con un
    evento nunca visto (implícito ≈ 1), el final debe ser ~factor × el día
    equivalente, es decir, empujar hacia abajo."""
    tx, stores, cal = toy_data
    cal = cal.copy()
    dia_evento = pd.Timestamp("2023-03-30")
    cal.loc[cal.date == dia_evento, "is_navidad_season"] = True  # -> navidad_ordinaria

    priors = pd.DataFrame({
        "evento": ["navidad_ordinaria"], "categoria": ["*"],
        "factor": [0.5], "vintage": [2022], "fuente": ["test"],
    })
    panel, cols = build_features(tx, stores, cal)
    panel = attach_uplift_prior(panel.drop(columns=PRIOR_COL), priors)

    fit = panel[panel.units_sold.notna() & (panel.date <= "2023-03-16")]
    model = DemandForecaster(cols, 14, {"loss": "poisson", "max_iter": 30, "random_state": 0},
                             prior_col=PRIOR_COL)
    model.fit(fit, fit.units_sold)

    test = panel[panel.date.isin([dia_evento, dia_evento - pd.Timedelta(days=7)])]
    pred = test.assign(y_pred=model.predict(test))
    s1 = pred[pred.store_id == "S1"].set_index("date")
    ratio = (s1.loc[dia_evento, "y_pred"].mean()
             / s1.loc[dia_evento - pd.Timedelta(days=7), "y_pred"].mean())
    assert 0.35 < ratio < 0.65
