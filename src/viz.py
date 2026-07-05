"""Estilo y paleta de visualización del proyecto.

Paleta categórica con orden fijo (validada para daltonismo): cada entidad
conserva siempre el mismo color en todos los gráficos del proyecto.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from . import config

# Superficie y tinta (modo claro)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Slots categóricos en orden fijo (el orden maximiza separación CVD)
_SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

CATEGORY_COLORS = dict(zip(config.CATEGORIES, _SLOTS))
FORMAT_COLORS = dict(zip(config.STORE_FORMATS, _SLOTS))
REGION_COLORS = dict(zip(config.REGIONS, _SLOTS))

ACCENT = _SLOTS[0]          # serie única / destacado
ACCENT_ALT = _SLOTS[5]      # contraste puntual (rojo)

# Rampa secuencial de un solo tono (azul 100 -> 700) para magnitudes
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)


def apply_style():
    """rcParams del proyecto: marcas delgadas, rejilla recesiva, sin chartjunk."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "figure.dpi": 110,
        "figure.figsize": (10, 4.2),
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "text.color": INK,
    })
