"""Carga tipada de los datos crudos y ensamblado de la tabla maestra.

Única puerta de entrada a los CSV: cada loader devuelve un DataFrame con los
tipos correctos (fechas como ``datetime``, categorías como ``Categorical``
con el orden canónico de ``config``). Si mañana la fuente fuera una base de
datos, este sería el único módulo que cambiaría.
"""
import pandas as pd

from .. import config


def load_transactions(path=None) -> pd.DataFrame:
    """Carga las transacciones diarias por tienda × categoría.

    Args:
        path: Ruta alternativa al CSV. Por defecto ``config.TRANSACTIONS_CSV``.

    Returns:
        DataFrame con una fila por (tienda, categoría, día); ``date`` como
        datetime y ``category`` como Categorical con el orden canónico.
    """
    df = pd.read_csv(path or config.TRANSACTIONS_CSV, parse_dates=["date"])
    df["category"] = pd.Categorical(df["category"], categories=config.CATEGORIES)
    return df


def load_stores(path=None) -> pd.DataFrame:
    """Carga el catálogo de tiendas con sus atributos estáticos.

    Args:
        path: Ruta alternativa al CSV. Por defecto ``config.STORES_CSV``.

    Returns:
        DataFrame con una fila por tienda; ``store_format`` y
        ``socioeconomic_level`` como Categorical (el NSE, ordenado C → A/B).
    """
    df = pd.read_csv(path or config.STORES_CSV)
    df["store_format"] = pd.Categorical(df["store_format"], categories=config.STORE_FORMATS)
    df["socioeconomic_level"] = pd.Categorical(
        df["socioeconomic_level"], categories=config.NSE_LEVELS, ordered=True
    )
    return df


def load_calendar(path=None) -> pd.DataFrame:
    """Carga el calendario de eventos y variables temporales.

    Args:
        path: Ruta alternativa al CSV. Por defecto ``config.CALENDAR_CSV``.

    Returns:
        DataFrame con una fila por día del periodo, ``date`` como datetime.
    """
    return pd.read_csv(path or config.CALENDAR_CSV, parse_dates=["date"])


def build_master(tx=None, stores=None, calendar=None) -> pd.DataFrame:
    """Une los tres insumos en el panel tienda × categoría × día.

    Args:
        tx: Transacciones (se cargan si se omiten).
        stores: Catálogo de tiendas (se carga si se omite).
        calendar: Calendario (se carga si se omite).

    Returns:
        DataFrame con las transacciones enriquecidas con atributos de tienda
        y calendario. Los merges validan cardinalidad m:1 y fallan ante
        duplicados en las dimensiones.
    """
    tx = tx if tx is not None else load_transactions()
    stores = stores if stores is not None else load_stores()
    calendar = calendar if calendar is not None else load_calendar()

    master = tx.merge(stores, on="store_id", how="left", validate="m:1")
    master = master.merge(calendar, on="date", how="left", validate="m:1")
    return master
