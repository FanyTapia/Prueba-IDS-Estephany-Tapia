"""Fixtures: un mini-dataset sintético con la misma estructura que los CSV."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_data():
    """2 tiendas × 2 categorías × 90 días, con demanda semanal conocida."""
    rng = np.random.RandomState(7)
    dates = pd.date_range("2023-01-01", periods=90, freq="D")

    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["quarter"] = cal["date"].dt.quarter
    for col in ["is_holiday", "is_payday", "is_navidad_season", "is_buen_fin", "is_semana_santa"]:
        cal[col] = False
    cal["is_weekend"] = cal["day_of_week"] >= 5

    stores = pd.DataFrame({
        "store_id": ["S1", "S2"],
        "store_format": ["Express", "Bodega"],
        "region": ["Norte", "Sur"],
        "size_sqm": [1500, 5000],
        "num_checkouts": [6, 14],
        "opening_year": [2018, 2015],
        "socioeconomic_level": ["C", "B"],
        "has_pharmacy": [False, True],
        "has_fuel_station": [False, False],
    })

    rows = []
    for sid, base in [("S1", 100), ("S2", 300)]:
        for cat, mult in [("Abarrotes", 1.0), ("Bebidas", 0.5)]:
            weekly = 1 + 0.5 * (cal["day_of_week"] >= 5).astype(float)
            units = base * mult * weekly + rng.normal(0, 2, len(dates))
            rows.append(pd.DataFrame({
                "date": dates, "store_id": sid, "category": cat,
                "units_sold": units.round(1),
                "replenishment_signal":
                    pd.Series(units).shift(2).rolling(7, min_periods=1).mean().values,
            }))
    tx = pd.concat(rows, ignore_index=True)
    return tx, stores, cal
