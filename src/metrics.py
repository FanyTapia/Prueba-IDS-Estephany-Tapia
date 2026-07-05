"""Métricas de pronóstico y evaluación segmentada."""
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


def evaluate(df: pd.DataFrame, by: list = None) -> pd.DataFrame:
    """Tabla de métricas sobre un DataFrame con columnas y_true / y_pred.

    Las filas sin real observado (huecos de POS) no son evaluables y se
    excluyen del cálculo, reportando cuántas fueron.
    """
    scored = df.dropna(subset=["y_true", "y_pred"])

    def _score(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "wape": wape(g["y_true"], g["y_pred"]),
            "mae": mae(g["y_true"], g["y_pred"]),
            "bias": bias(g["y_true"], g["y_pred"]),
        })

    if by:
        out = scored.groupby(by, observed=True).apply(_score, include_groups=False).reset_index()
    else:
        out = _score(scored).to_frame().T
    out["n"] = out["n"].astype(int)
    out["n_sin_real"] = len(df) - len(scored)
    return out
