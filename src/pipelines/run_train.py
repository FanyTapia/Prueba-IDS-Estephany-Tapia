"""Pipeline de entrenamiento: ajusta el modelo final con toda la historia
disponible y lo serializa como artefacto desplegable.

Uso:  uv run retail-train [--horizon 14]
"""
import argparse
import logging

from .. import config
from ..artifacts import create_run_dir, run_metadata, save_json, setup_logging
from ..data import load_calendar, load_stores, load_transactions
from ..features import build_features
from ..models import DemandForecaster

logger = logging.getLogger("pipelines.train")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    args = parser.parse_args(argv)
    setup_logging()

    logger.info("Cargando datos y construyendo features (horizonte=%s días)…", args.horizon)
    tx, stores, cal = load_transactions(), load_stores(), load_calendar()
    panel, feature_cols = build_features(tx, stores, cal, horizon=args.horizon)

    fit_mask = panel[config.TARGET].notna()
    logger.info("Entrenando con %s filas (hasta %s)…", fit_mask.sum(), panel["date"].max().date())
    model = DemandForecaster(feature_cols, args.horizon)
    model.fit(panel.loc[fit_mask], panel.loc[fit_mask, config.TARGET])

    run_dir = create_run_dir("models")
    model.save(run_dir / "model.joblib")
    save_json(run_metadata(
        horizon=args.horizon, feature_cols=feature_cols,
        model_params=model.params, target=config.TARGET,
        train_rows=int(fit_mask.sum()),
        train_date_min=panel["date"].min(), train_date_max=panel["date"].max(),
    ), run_dir / "run_config.json")
    logger.info("Modelo serializado en %s", run_dir / "model.joblib")


if __name__ == "__main__":
    main()
