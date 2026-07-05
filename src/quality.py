"""Chequeos de calidad e integridad sobre el panel de transacciones."""
import pandas as pd

from . import config


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje y conteo de nulos por columna, ordenado descendente."""
    out = pd.DataFrame({
        "n_nulos": df.isnull().sum(),
        "pct_nulos": (df.isnull().mean() * 100).round(2),
    })
    return out[out.n_nulos > 0].sort_values("pct_nulos", ascending=False)


def grid_gaps(tx: pd.DataFrame) -> pd.DataFrame:
    """Días completamente ausentes por tienda (huecos en el panel).

    El panel esperado es tienda x categoría x día; una tienda con menos
    fechas únicas que el total del periodo tiene días sin registro alguno.
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

    - amount_total == amount_cash + amount_card (esperado: exacto)
    - total_transactions == cash + card (falla sistemáticamente en promos)
    - avg_ticket == amount_total / transacciones *pagadas* (cash + card)
    """
    sub = tx.dropna(subset=["cash_transactions", "avg_ticket"]).copy()
    paid = sub.cash_transactions + sub.card_transactions

    checks = {
        "monto: total == cash + card": (sub.amount_total - sub.amount_cash - sub.amount_card).abs() < 0.01,
        "conteo: total == cash + card": (sub.total_transactions - paid) == 0,
        "avg_ticket == monto / total_transactions": (sub.amount_total / sub.total_transactions - sub.avg_ticket).abs() < 0.01,
        "avg_ticket == monto / pagadas (cash+card)": (sub.amount_total / paid - sub.avg_ticket).abs() < 0.01,
    }
    return pd.DataFrame({
        "pct_filas_cumplen": {k: round(v.mean() * 100, 2) for k, v in checks.items()},
        "pct_en_dias_promo": {k: round(v[sub.has_promotion == 1].mean() * 100, 2) for k, v in checks.items()},
        "pct_en_dias_sin_promo": {k: round(v[sub.has_promotion == 0].mean() * 100, 2) for k, v in checks.items()},
    })
