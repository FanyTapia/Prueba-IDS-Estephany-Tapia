"""Chequeos de calidad e integridad sobre el panel de transacciones.

Funciones de *verificación*, no de transformación: devuelven diagnósticos
(nulos, huecos, identidades contables) sin modificar los datos. Aquí vive la
lógica que detectó las "transacciones fantasma" de los días con promoción.
"""
import pandas as pd


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume los valores nulos por columna.

    Args:
        df: Cualquier DataFrame a auditar.

    Returns:
        DataFrame indexado por columna con ``n_nulos`` y ``pct_nulos``,
        solo columnas con al menos un nulo, ordenado descendente.
    """
    out = pd.DataFrame({
        "n_nulos": df.isnull().sum(),
        "pct_nulos": (df.isnull().mean() * 100).round(2),
    })
    return out[out.n_nulos > 0].sort_values("pct_nulos", ascending=False)


def grid_gaps(tx: pd.DataFrame) -> pd.DataFrame:
    """Encuentra días completamente ausentes por tienda (huecos del panel).

    El panel esperado es tienda × categoría × día; una tienda con menos
    fechas únicas que el total del periodo tiene días sin registro alguno
    (caídas de POS/conectividad).

    Args:
        tx: Transacciones crudas.

    Returns:
        DataFrame con una fila por tienda afectada: ``store_id``,
        ``n_dias_faltantes`` y la lista de ``fechas`` ausentes.
    """
    all_dates = set(tx["date"].unique())
    rows = []
    for store_id, g in tx.groupby("store_id"):
        missing = sorted(all_dates - set(g["date"].unique()))
        if missing:
            rows.append({
                "store_id": store_id,
                "n_dias_faltantes": len(missing),
                "fechas": [pd.Timestamp(d).date().isoformat() for d in missing],
            })
    return pd.DataFrame(rows)


def payment_consistency(tx: pd.DataFrame) -> pd.DataFrame:
    """Verifica las identidades contables del dataset.

    Comprueba, sobre las filas verificables (sin nulos en las columnas
    involucradas): que el monto total sea cash + card, que el conteo de
    transacciones cuadre (falla sistemáticamente con promoción: las
    "transacciones fantasma") y contra qué denominador se calcula el
    ticket promedio.

    Args:
        tx: Transacciones crudas.

    Returns:
        DataFrame con una fila por identidad y el porcentaje de filas que
        la cumplen, desglosado total / con promoción / sin promoción.
    """
    sub = tx.dropna(subset=["cash_transactions", "avg_ticket"]).copy()
    paid = sub.cash_transactions + sub.card_transactions

    checks = {
        "monto: total == cash + card":
            (sub.amount_total - sub.amount_cash - sub.amount_card).abs() < 0.01,
        "conteo: total == cash + card": (sub.total_transactions - paid) == 0,
        "avg_ticket == monto / total_transactions":
            (sub.amount_total / sub.total_transactions - sub.avg_ticket).abs() < 0.01,
        "avg_ticket == monto / pagadas (cash+card)":
            (sub.amount_total / paid - sub.avg_ticket).abs() < 0.01,
    }
    return pd.DataFrame({
        "pct_filas_cumplen": {k: round(v.mean() * 100, 2) for k, v in checks.items()},
        "pct_en_dias_promo":
            {k: round(v[sub.has_promotion == 1].mean() * 100, 2) for k, v in checks.items()},
        "pct_en_dias_sin_promo":
            {k: round(v[sub.has_promotion == 0].mean() * 100, 2) for k, v in checks.items()},
    })
