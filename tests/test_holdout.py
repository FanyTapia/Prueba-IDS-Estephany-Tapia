"""El holdout es honesto solo si los datos apartados no existen para el
modelo: ni en el entrenamiento ni en las features de los días a predecir."""
import pandas as pd
import pytest

from src.config import MODEL_PARAMS
from src.evaluacion.holdout import run_holdout, summarize_holdout

PARAMS = {**MODEL_PARAMS, "max_iter": 40}


def test_perturbar_el_holdout_no_cambia_ninguna_prediccion(toy_data):
    """Multiplicar por 10 la demanda de los días apartados debe dejar las
    predicciones idénticas: si algo cambiara, el modelo estaría viendo
    información del periodo que se supone no existe."""
    tx, stores, cal = toy_data
    cutoff = tx["date"].max() - pd.Timedelta(days=10)

    a = run_holdout(tx, stores, cal, cutoff, params=PARAMS)

    tx_b = tx.copy()
    tx_b.loc[tx_b["date"] > cutoff, "units_sold"] *= 10
    b = run_holdout(tx_b, stores, cal, cutoff, params=PARAMS)

    pd.testing.assert_frame_equal(
        a.drop(columns="y_true"), b.drop(columns="y_true")
    )
    # ...pero el real evaluado sí refleja los datos apartados
    assert (b["y_true"].dropna() > a["y_true"].dropna()).any()


def test_holdout_cubre_exactamente_los_dias_apartados(toy_data):
    tx, stores, cal = toy_data
    cutoff = tx["date"].max() - pd.Timedelta(days=10)
    preds = run_holdout(tx, stores, cal, cutoff, params=PARAMS)

    assert preds["date"].min() == cutoff + pd.Timedelta(days=1)
    assert preds["date"].max() == tx["date"].max()
    assert set(preds["modelo"]) == {"gradient_boosting", "gradient_boosting_prior",
                                    "naive_estacional", "sistema_actual"}
    assert preds[preds["modelo"] == "gradient_boosting"]["y_pred"].notna().all()

    resumen = summarize_holdout(preds)
    assert set(resumen["tramo"].astype(str)) == {"total", "días 1-7", "días 8+"}


def test_holdout_sin_dias_posteriores_es_rechazado(toy_data):
    tx, stores, cal = toy_data
    with pytest.raises(ValueError, match="apartar"):
        run_holdout(tx, stores, cal, tx["date"].max(), params=PARAMS)
