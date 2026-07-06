import numpy as np
import pandas as pd

from src.evaluacion.metrics import bias, bootstrap_cis, evaluate, mae, paired_wape_diff_ci, wape


def _panel_scored(n_dias=60, ruido=10.0, sesgo=0.0, seed=0):
    """Panel sintético: 20 series × n_dias con error controlado."""
    rng = np.random.RandomState(seed)
    fechas = pd.date_range("2023-01-01", periods=n_dias, freq="D")
    filas = []
    for serie in range(20):
        y = 100 + rng.normal(0, 5, n_dias)
        filas.append(pd.DataFrame({
            "date": fechas, "serie": serie, "y_true": y,
            "y_pred": y + sesgo + rng.normal(0, ruido, n_dias),
        }))
    return pd.concat(filas, ignore_index=True)


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


def test_ic_contiene_al_estimador_y_es_reproducible():
    df = _panel_scored()
    ci_a = bootstrap_cis(df, n_boot=500, seed=7)
    ci_b = bootstrap_cis(df, n_boot=500, seed=7)
    assert ci_a == ci_b  # misma semilla -> mismo IC
    w = wape(df["y_true"], df["y_pred"])
    assert ci_a["wape_lo"] <= w <= ci_a["wape_hi"]
    assert ci_a["wape_lo"] < ci_a["wape_hi"]


def test_ic_colapsa_con_prediccion_perfecta():
    df = _panel_scored(ruido=0.0)
    df["y_pred"] = df["y_true"]
    ci = bootstrap_cis(df, n_boot=200)
    assert ci["wape_lo"] == ci["wape_hi"] == 0.0


def test_ic_se_estrecha_con_mas_dias():
    ancho = {}
    for n_dias in [25, 400]:
        ci = bootstrap_cis(_panel_scored(n_dias=n_dias), n_boot=500)
        ancho[n_dias] = ci["wape_hi"] - ci["wape_lo"]
    assert ancho[400] < ancho[25] / 2  # ~1/sqrt(n): 4x días -> mitad de ancho


def test_diferencia_pareada_detecta_al_mejor():
    """Modelo A con error chico vs B con error grande sobre las MISMAS celdas:
    el IC de ΔWAPE debe quedar completo debajo de 0 (A significativamente mejor)."""
    base = _panel_scored(ruido=5.0, seed=1).assign(modelo="A")
    peor = base.copy().assign(modelo="B")
    rng = np.random.RandomState(2)
    peor["y_pred"] = peor["y_true"] + rng.normal(0, 25, len(peor))

    rep = paired_wape_diff_ci(pd.concat([base, peor]), "A", "B", n_boot=500)
    assert rep.loc[0, "delta_wape"] < 0
    assert rep.loc[0, "delta_hi"] < 0
    assert bool(rep.loc[0, "significativo"])


def test_diferencia_pareada_no_inventa_ganador():
    """El mismo modelo contra sí mismo (con otro nombre): ΔWAPE = 0 exacto
    y no significativo."""
    a = _panel_scored(seed=3).assign(modelo="A")
    b = a.copy().assign(modelo="B")
    rep = paired_wape_diff_ci(pd.concat([a, b]), "A", "B", n_boot=300)
    assert rep.loc[0, "delta_wape"] == 0.0
    assert not bool(rep.loc[0, "significativo"])
