"""Configuración central del proyecto: rutas y constantes de dominio."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
STORES_CSV = DATA_DIR / "stores.csv"
CALENDAR_CSV = DATA_DIR / "calendar.csv"

# Orden canónico (del diccionario de datos). Se usa también para asignar
# colores de forma fija: una categoría siempre recibe el mismo color.
CATEGORIES = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
STORE_FORMATS = ["Supercenter", "Bodega", "Express"]
REGIONS = ["Norte", "Centro", "Sur", "Occidente", "Oriente"]
NSE_LEVELS = ["C", "C+", "B", "A/B"]

DATE_MIN = "2023-01-01"
DATE_MAX = "2024-02-29"

# Llaves del panel: una fila = tienda x categoría x día
PANEL_KEYS = ["store_id", "category", "date"]
