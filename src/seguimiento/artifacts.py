"""Artefactos de corrida: directorios versionados, metadatos y logging."""
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import config


def setup_logging() -> None:
    """Configura el logging estándar de los pipelines (nivel INFO)."""
    # La consola de Windows usa cp1252 por defecto y no puede codificar
    # caracteres como ✓ o Δ que imprimen los pipelines; UTF-8 con
    # ``errors="replace"`` evita que un print cosmético tumbe la corrida.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def create_run_dir(kind: str) -> Path:
    """Crea un directorio inmutable para una corrida.

    Args:
        kind: Tipo de corrida (``backtests``, ``models``, ``forecasts``,
            ``holdouts``, ``flows``…); agrupa las corridas por subcarpeta.

    Returns:
        Ruta ``outputs/<kind>/<timestamp UTC>/`` recién creada. Falla si ya
        existe (los directorios de corrida no se reutilizan).
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUTS_DIR / kind / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_run_dir(kind: str) -> Path:
    """Localiza la corrida más reciente de un tipo.

    Args:
        kind: Tipo de corrida (ver :func:`create_run_dir`).

    Returns:
        Ruta del directorio más reciente.

    Raises:
        FileNotFoundError: Si no existe ninguna corrida de ese tipo.
    """
    base = config.OUTPUTS_DIR / kind
    runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
    if not runs:
        raise FileNotFoundError(
            f"No hay corridas en {base}. Ejecuta primero el pipeline correspondiente."
        )
    return runs[-1]


def run_metadata(**extra) -> dict:
    """Reúne los metadatos mínimos para reproducir una corrida.

    Args:
        **extra: Pares clave-valor específicos de la corrida (horizonte,
            features, hiperparámetros, ventanas…).

    Returns:
        Diccionario con timestamp UTC, versiones de Python/pandas/sklearn,
        plataforma, y los extras recibidos.
    """
    import pandas
    import sklearn

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        **extra,
    }


def save_json(obj: dict, path: Path) -> None:
    """Escribe un diccionario como JSON legible (UTF-8, indentado).

    Args:
        obj: Diccionario serializable (los no serializables van vía ``str``).
        path: Ruta destino.
    """
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
