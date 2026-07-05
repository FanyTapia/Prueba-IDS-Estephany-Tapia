"""Carga y ensamblado de los datos crudos.

Cada loader devuelve un DataFrame tipado; `build_master` une los tres
insumos (transacciones + tiendas + calendario) en la tabla de análisis.
"""
import pandas as pd

from . import config


def load_transactions(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.TRANSACTIONS_CSV, parse_dates=["date"])
    df["category"] = pd.Categorical(df["category"], categories=config.CATEGORIES)
    return df


def load_stores(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.STORES_CSV)
    df["store_format"] = pd.Categorical(df["store_format"], categories=config.STORE_FORMATS)
    df["socioeconomic_level"] = pd.Categorical(
        df["socioeconomic_level"], categories=config.NSE_LEVELS, ordered=True
    )
    return df


def load_calendar(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.CALENDAR_CSV, parse_dates=["date"])


def build_master(tx=None, stores=None, calendar=None) -> pd.DataFrame:
    """Panel tienda x categoría x día enriquecido con atributos de tienda y calendario."""
    tx = tx if tx is not None else load_transactions()
    stores = stores if stores is not None else load_stores()
    calendar = calendar if calendar is not None else load_calendar()

    master = tx.merge(stores, on="store_id", how="left", validate="m:1")
    master = master.merge(calendar, on="date", how="left", validate="m:1")
    return master
