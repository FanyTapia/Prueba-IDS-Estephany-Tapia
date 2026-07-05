"""Artefactos de corrida: directorios versionados, metadatos y logging."""
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def create_run_dir(kind: str) -> Path:
    """outputs/<kind>/<timestamp UTC>/ — un directorio inmutable por corrida."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUTS_DIR / kind / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_run_dir(kind: str) -> Path:
    base = config.OUTPUTS_DIR / kind
    runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
    if not runs:
        raise FileNotFoundError(
            f"No hay corridas en {base}. Ejecuta primero el pipeline correspondiente."
        )
    return runs[-1]


def run_metadata(**extra) -> dict:
    """Metadatos mínimos para reproducir una corrida."""
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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
