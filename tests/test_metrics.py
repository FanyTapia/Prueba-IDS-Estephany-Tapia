import numpy as np
import pandas as pd

from src.metrics import bias, evaluate, mae, wape


def test_wape_valores_conocidos():
    assert wape([100, 100], [90, 110]) == 0.1
    assert wape([50, 50], [50, 50]) == 0.0


def test_mae_valores_conocidos():
    assert mae([10, 20], [12, 16]) == 3.0


def test_bias_signo():
    assert bias([100], [120]) == 0.2      # sobre-pronóstico -> positivo
    assert bias([100], [80]) == -0.2      # sub-pronóstico -> negativo


def test_evaluate_excluye_reales_faltantes():
    df = pd.DataFrame({
        "y_true": [100.0, np.nan, 100.0],
        "y_pred": [90.0, 50.0, 110.0],
    })
    out = evaluate(df)
    assert out.loc[0, "n"] == 2
    assert out.loc[0, "n_sin_real"] == 1
    assert out.loc[0, "wape"] == 0.1
