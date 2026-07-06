"""Métricas de pronóstico, evaluación segmentada e intervalos de confianza.

Los IC se calculan con bootstrap por bloques de FECHA (cluster bootstrap):
se remuestrean días completos, no filas sueltas, porque los errores de las
480 series están correlacionados dentro de un mismo día (un evento o un
festivo mueve a todas las tiendas a la vez). Remuestrear filas individuales
ignoraría esa correlación y produciría intervalos engañosamente estrechos.

Para comparar dos modelos, `paired_wape_diff_ci` remuestrea los mismos días
para ambos (diseño pareado): el IC de la diferencia de WAPE es la prueba
correcta de "A le gana a B", no la superposición de dos IC marginales.
"""
import numpy as np
import pandas as pd


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: sum|error| / sum(real)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.abs(y_true - y_pred).sum() / y_true.sum())


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.abs(y_true - y_pred).mean())


def bias(y_true, y_pred) -> float:
    """Sesgo relativo: (+) sobre-pronostica, (−) sub-pronostica."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float((y_pred.sum() - y_true.sum()) / y_true.sum())


def _per_date_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos suficientes por día para reconstruir WAPE/MAE/sesgo
    sobre cualquier remuestra de fechas: sum|e|, sum(real), sum(pred), n."""
    d = df.assign(_abs_err=(df["y_true"] - df["y_pred"]).abs())
    return d.groupby("date").agg(
        abs_err=("_abs_err", "sum"), y=("y_true", "sum"),
        pred=("y_pred", "sum"), n=("y_true", "size"),
    )


def bootstrap_cis(df: pd.DataFrame, n_boot: int = 1000, alpha: float = 0.05,
                  seed: int = 42) -> dict:
    """IC percentil (1−alpha) de WAPE y sesgo, bootstrap por bloques de fecha.

    Trabaja sobre los estadísticos suficientes por día, así cada réplica es
    O(n_días) y el bootstrap completo corre en milisegundos.
    """
    per_date = _per_date_sums(df)
    A, Y, P = (per_date[c].to_numpy(float) for c in ["abs_err", "y", "pred"])
    n_dates = len(per_date)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_dates, size=(n_boot, n_dates))
    Ab, Yb, Pb = A[idx].sum(axis=1), Y[idx].sum(axis=1), P[idx].sum(axis=1)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    wape_b, bias_b = Ab / Yb, Pb / Yb - 1
    return {
        "wape_lo": float(np.percentile(wape_b, lo)),
        "wape_hi": float(np.percentile(wape_b, hi)),
        "bias_lo": float(np.percentile(bias_b, lo)),
        "bias_hi": float(np.percentile(bias_b, hi)),
        "n_dias": int(n_dates),
    }


def evaluate(df: pd.DataFrame, by: list = None, ci: bool = False,
             n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Tabla de métricas sobre un DataFrame con columnas y_true / y_pred.

    Con `ci=True` agrega el IC 95% (bootstrap por fecha) de WAPE y sesgo.
    Las filas sin real observado (huecos de POS) no son evaluables y se
    excluyen del cálculo, reportando cuántas fueron.
    """
    scored = df.dropna(subset=["y_true", "y_pred"])

    def _score(g: pd.DataFrame) -> pd.Series:
        out = {
            "n": len(g),
            "wape": wape(g["y_true"], g["y_pred"]),
            "mae": mae(g["y_true"], g["y_pred"]),
            "bias": bias(g["y_true"], g["y_pred"]),
        }
        if ci:
            out.update(bootstrap_cis(g, n_boot=n_boot, seed=seed))
        return pd.Series(out)

    if by:
        out = scored.groupby(by, observed=True).apply(_score, include_groups=False).reset_index()
    else:
        out = _score(scored).to_frame().T
    out["n"] = out["n"].astype(int)
    out["n_sin_real"] = len(df) - len(scored)
    return out


def paired_wape_diff_ci(predictions: pd.DataFrame, model_a: str, model_b: str,
                        by: list = None, n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 42) -> pd.DataFrame:
    """IC de ΔWAPE = WAPE(A) − WAPE(B) con bootstrap PAREADO por fecha.

    Ambos modelos se evalúan sobre exactamente las mismas celdas y cada
    réplica remuestrea los mismos días para los dos, de modo que la variación
    común (días fáciles/difíciles) se cancela y el IC refleja solo la
    diferencia real entre modelos. Si `delta_hi < 0`, A es significativamente
    mejor que B al nivel 1−alpha.
    """
    scored = predictions[predictions["modelo"].isin([model_a, model_b])] \
        .dropna(subset=["y_true", "y_pred"])

    def _diff(group: pd.DataFrame) -> pd.Series:
        pa = _per_date_sums(group[group["modelo"] == model_a])
        pb = _per_date_sums(group[group["modelo"] == model_b])
        fechas = pa.index.intersection(pb.index)
        Aa, Ya = pa.loc[fechas, "abs_err"].to_numpy(), pa.loc[fechas, "y"].to_numpy()
        Ab = pb.loc[fechas, "abs_err"].to_numpy()

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(fechas), size=(n_boot, len(fechas)))
        delta_b = Aa[idx].sum(axis=1) / Ya[idx].sum(axis=1) \
            - Ab[idx].sum(axis=1) / Ya[idx].sum(axis=1)

        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        delta = Aa.sum() / Ya.sum() - Ab.sum() / Ya.sum()
        return pd.Series({
            "delta_wape": delta,
            "delta_lo": float(np.percentile(delta_b, lo)),
            "delta_hi": float(np.percentile(delta_b, hi)),
            "significativo": bool(np.percentile(delta_b, hi) < 0
                                  or np.percentile(delta_b, lo) > 0),
            "n_dias": int(len(fechas)),
        })

    if by:
        out = scored.groupby(by, observed=True).apply(_diff, include_groups=False).reset_index()
    else:
        out = _diff(scored).to_frame().T
    out.insert(0, "comparacion", f"{model_a} − {model_b}")
    return out
