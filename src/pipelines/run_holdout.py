"""Pipeline de holdout final: aparta los últimos días del dataset, entrena
sin verlos, los pronostica de una sola emisión y evalúa contra lo apartado.

Uso:  uv run retail-holdout [--holdout-days 15]
"""
import argparse
import logging

import pandas as pd

from .. import config
from ..evaluacion.holdout import run_holdout, summarize_holdout, summarize_holdout_paired
from ..preprocesamiento.data import load_calendar, load_stores, load_transactions
from ..seguimiento.artifacts import create_run_dir, run_metadata, save_json, setup_logging

logger = logging.getLogger("pipelines.holdout")


def main(argv=None) -> None:
    """Punto de entrada de la evaluación por holdout final.

    Args:
        argv: Argumentos de línea de comandos (None = ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-days", type=int, default=15,
                        help="Días finales a apartar (default: 15)")
    args = parser.parse_args(argv)
    setup_logging()

    tx, stores, cal = load_transactions(), load_stores(), load_calendar()
    cutoff = tx["date"].max() - pd.Timedelta(days=args.holdout_days)
    logger.info("Corte: %s | holdout: %s días finales", cutoff.date(), args.holdout_days)

    predictions = run_holdout(tx, stores, cal, cutoff)
    metrics = summarize_holdout(predictions)
    paired = summarize_holdout_paired(predictions)

    run_dir = create_run_dir("holdouts")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    paired.to_csv(run_dir / "paired_diffs.csv", index=False)
    save_json(run_metadata(
        cutoff=cutoff, holdout_days=args.holdout_days,
        model_params=config.MODEL_PARAMS, target=config.TARGET,
    ), run_dir / "run_config.json")

    logger.info("Artefactos en %s", run_dir)
    print("\n=== Holdout final — WAPE por modelo y tramo (IC 95% bootstrap por fecha) ===")
    print(metrics.round(3).to_string(index=False))
    print("\n=== ΔWAPE pareado: campeón − rival (negativo = campeón mejor) ===")
    print(paired.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
