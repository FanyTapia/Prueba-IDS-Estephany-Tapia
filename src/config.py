"""Configuración central del proyecto: rutas, constantes de dominio y modelo."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
STORES_CSV = DATA_DIR / "stores.csv"
CALENDAR_CSV = DATA_DIR / "calendar.csv"
EVENT_PRIORS_CSV = DATA_DIR / "external" / "event_priors.csv"
EVENT_REFERENCE_CSV = DATA_DIR / "external" / "event_reference.csv"

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

# ---------------------------------------------------------------- modelo

TARGET = "units_sold"

# Horizonte operativo del pronóstico (días). Los rezagos de demanda deben ser
# >= HORIZON para que ninguna feature use información posterior al momento en
# que se emite el pronóstico (regla anti-leakage, ver src/features.py).
HORIZON = 14

# Rezagos en días: todos múltiplos de 7 (mismo día de la semana) y >= HORIZON.
LAG_DAYS = [14, 21, 28]

# Ventanas de medias móviles sobre la demanda desplazada HORIZON días.
ROLLING_WINDOWS = [7, 28]

# Hiperparámetros del campeón. Poisson: la demanda es un conteo no negativo.
MODEL_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 42,
}

# Orígenes del backtest walk-forward: se entrena con datos <= origen y se
# pronostica (origen, origen + HORIZON]. Elegidos para cubrir periodos
# normales (jul, sep, ene, feb) y los dos picos del año: Buen Fin
# (origen 2023-11-06 -> pronostica 07-20 nov) y Navidad
# (origen 2023-12-11 -> pronostica 12-25 dic).
BACKTEST_ORIGINS = [
    "2023-07-03",
    "2023-09-04",
    "2023-11-06",
    "2023-12-11",
    "2024-01-15",
    "2024-02-12",
]
