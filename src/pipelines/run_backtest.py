"""Pipeline de backtest: valida el campeón contra los baselines walk-forward.

Uso:  uv run retail-backtest [--horizon 14] [--origins 2023-11-06,2023-12-11]
"""
import argparse
import logging

from .. import config
from ..artifacts import create_run_dir, run_metadata, save_json, setup_logging
from ..backtest import run_backtest, summarize
from ..data import load_calendar, load_stores, load_transactions
from ..features import build_features
from ..splits import make_folds

logger = logging.getLogger("pipelines.backtest")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--origins", type=str, default=",".join(config.BACKTEST_ORIGINS),
                        help="Orígenes walk-forward separados por coma (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    setup_logging()

    logger.info("Cargando datos y construyendo features (horizonte=%s días)…", args.horizon)
    tx, stores, cal = load_transactions(), load_stores(), load_calendar()
    panel, feature_cols = build_features(tx, stores, cal, horizon=args.horizon)

    origins = [o.strip() for o in args.origins.split(",") if o.strip()]
    folds = make_folds(origins, args.horizon, panel["date"].max())

    predictions = run_backtest(panel, feature_cols, folds)
    metrics = summarize(predictions)

    run_dir = create_run_dir("backtests")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    save_json(run_metadata(
        horizon=args.horizon, origins=origins, feature_cols=feature_cols,
        model_params=config.MODEL_PARAMS, target=config.TARGET,
    ), run_dir / "run_config.json")

    logger.info("Artefactos en %s", run_dir)
    print("\n=== Backtest — WAPE por modelo y segmento ===")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
