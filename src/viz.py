"""Estilo y paleta de visualización del proyecto.

Paleta categórica con orden fijo (validada para daltonismo): cada entidad
conserva siempre el mismo color en todos los gráficos del proyecto.
Se usan gráficos estáticos (matplotlib) para que el entregable sea visible
directamente en GitHub, que no ejecuta JavaScript en la vista de notebooks.
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
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)
# Rampa ordinal corta (C -> A/B) que aún contrasta con la superficie
ORDINAL_STEPS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


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


def apply_plotly_style():
    """Registra el tema 'retail' en Plotly y activa el renderer combinado.

    El renderer `plotly_mimetype+png` emite en cada figura tanto la versión
    interactiva (para inspección local: hover, zoom) como un PNG estático
    (para que GitHub, que no ejecuta JavaScript, muestre el gráfico). Import
    diferido de Plotly para no cargarlo en los notebooks que solo usan
    matplotlib.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    eje = dict(gridcolor=GRID, gridwidth=1, linecolor=BASELINE, zeroline=False,
               ticks="outside", tickcolor=BASELINE,
               tickfont=dict(color=INK_MUTED, size=11),
               title=dict(font=dict(color=INK_SECONDARY, size=12)))
    tpl = go.layout.Template()
    tpl.layout = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=INK, size=12),
        title=dict(font=dict(size=15, color=INK), x=0.01, xanchor="left"),
        colorway=_SLOTS, xaxis=eje, yaxis=eje,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=60, r=30, t=55, b=45),
    )
    pio.templates["retail"] = tpl
    pio.templates.default = "retail"
    pio.renderers.default = "plotly_mimetype+png"
