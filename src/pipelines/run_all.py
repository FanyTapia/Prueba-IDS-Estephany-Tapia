"""Orquestador del flujo completo: de los CSV crudos al modelo registrado.

Ejecuta en una sola corrida las etapas del proceso, en orden:

1. Carga y validación ligera de los datos (preprocesamiento).
2. Evaluación honesta por holdout: aparta los últimos N días de la ventana,
   entrena sin ellos y mide contra lo apartado (evaluación).
3. Backtest walk-forward completo (opcional, más lento).
4. Entrenamiento del modelo final con TODA la ventana (entrenamiento).
5. Registro en MLflow: hiperparámetros, métricas, artefactos y signature
   (seguimiento) + serialización local en ``outputs/models/``.
6. Pronóstico operativo de los próximos días (inferencia, opcional).

Uso típico:
    uv run retail-flow                          # flujo completo con defaults
    uv run retail-flow --train-end 2024-01-31   # ventana de entrenamiento acotada
    uv run retail-flow --holdout-days 0 --no-forecast   # solo entrenar y registrar
    uv run retail-flow --with-backtest          # incluye el walk-forward completo
"""
import argparse
import logging

import pandas as pd

from .. import config
from ..entrenamiento.models import DemandForecaster
from ..evaluacion.backtest import run_backtest, summarize, summarize_paired
from ..evaluacion.holdout import run_holdout, summarize_holdout, summarize_holdout_paired
from ..evaluacion.splits import make_folds
from ..inferencia.prediccion import generar_pronostico
from ..postproceso.priors import PRIOR_COL
from ..preprocesamiento.data import load_calendar, load_stores, load_transactions
from ..preprocesamiento.features import build_features
from ..seguimiento.artifacts import create_run_dir, run_metadata, save_json, setup_logging
from ..seguimiento.mlflow_tracking import log_training_run

logger = logging.getLogger("pipelines.flow")


def parse_args(argv=None) -> argparse.Namespace:
    """Define y parsea los parámetros del flujo.

    Args:
        argv: Argumentos de línea de comandos (None = ``sys.argv``).

    Returns:
        Namespace con la configuración de la corrida.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ventana = parser.add_argument_group("ventana de entrenamiento")
    ventana.add_argument("--train-start", type=str, default=None,
                         help="Primer día de datos a usar (default: inicio del dataset)")
    ventana.add_argument("--train-end", type=str, default=None,
                         help="Último día de datos a usar (default: fin del dataset)")

    modelo = parser.add_argument_group("modelo")
    modelo.add_argument("--horizon", type=int, default=config.HORIZON,
                        help="Horizonte del pronóstico en días (default: %(default)s)")
    modelo.add_argument("--learning-rate", type=float, default=None,
                        help="Override del learning rate del boosting")
    modelo.add_argument("--max-iter", type=int, default=None,
                        help="Override del número de iteraciones del boosting")
    modelo.add_argument("--seed", type=int, default=None,
                        help="Semilla del estimador (default: la de config)")

    evalua = parser.add_argument_group("evaluación")
    evalua.add_argument("--holdout-days", type=int, default=14,
                        help="Días finales de la ventana apartados para la evaluación "
                             "honesta previa al entrenamiento final; 0 la desactiva "
                             "(default: %(default)s)")
    evalua.add_argument("--with-backtest", action="store_true",
                        help="Corre también el backtest walk-forward completo (lento)")

    salida = parser.add_argument_group("salida y tracking")
    salida.add_argument("--no-forecast", action="store_true",
                        help="No emitir el pronóstico operativo al final")
    salida.add_argument("--experiment", type=str, default="retail-demand",
                        help="Experimento MLflow (default: %(default)s)")
    salida.add_argument("--run-name", type=str, default=None,
                        help="Nombre de la corrida en MLflow (default: automático)")
    salida.add_argument("--no-register", action="store_true",
                        help="No registrar el modelo en el Model Registry local")
    return parser.parse_args(argv)


def _resolver_params(args: argparse.Namespace) -> dict:
    """Combina los hiperparámetros de config con los overrides de la CLI.

    Args:
        args: Parámetros parseados.

    Returns:
        Diccionario de hiperparámetros listo para ``DemandForecaster``.
    """
    params = dict(config.MODEL_PARAMS)
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.max_iter is not None:
        params["max_iter"] = args.max_iter
    if args.seed is not None:
        params["random_state"] = args.seed
    return params


def main(argv=None) -> None:
    """Ejecuta el flujo completo según los parámetros recibidos.

    Args:
        argv: Argumentos de línea de comandos (None = ``sys.argv``).
    """
    args = parse_args(argv)
    setup_logging()
    params = _resolver_params(args)
    run_dir = create_run_dir("flows")

    # ── 1 · Preprocesamiento: carga y ventana de entrenamiento ────────────
    tx, stores, cal = load_transactions(), load_stores(), load_calendar()
    train_start = pd.Timestamp(args.train_start) if args.train_start else tx["date"].min()
    train_end = pd.Timestamp(args.train_end) if args.train_end else tx["date"].max()
    tx = tx[(tx["date"] >= train_start) & (tx["date"] <= train_end)]
    if tx.empty:
        raise ValueError("La ventana de entrenamiento no contiene datos.")
    logger.info("Ventana de datos: %s → %s (%s filas)",
                train_start.date(), train_end.date(), len(tx))

    metrics_mlflow: dict = {}
    artifacts_mlflow: dict = {}

    # ── 2 · Evaluación honesta por holdout (previa al entrenamiento final) ─
    if args.holdout_days > 0:
        cutoff = train_end - pd.Timedelta(days=args.holdout_days)
        logger.info("Holdout de %s días (corte %s)…", args.holdout_days, cutoff.date())
        h_preds = run_holdout(tx, stores, cal, cutoff, params=params)
        h_metrics = summarize_holdout(h_preds)
        h_paired = summarize_holdout_paired(h_preds)
        h_metrics.to_csv(run_dir / "holdout_metrics.csv", index=False)
        h_paired.to_csv(run_dir / "holdout_paired_diffs.csv", index=False)
        artifacts_mlflow["holdout_metrics"] = run_dir / "holdout_metrics.csv"
        artifacts_mlflow["holdout_paired"] = run_dir / "holdout_paired_diffs.csv"

        tot = h_metrics[h_metrics["tramo"] == "total"].set_index("modelo")
        metrics_mlflow.update({
            "holdout_wape": tot.loc["gradient_boosting_prior", "wape"],
            "holdout_mae": tot.loc["gradient_boosting_prior", "mae"],
            "holdout_bias": tot.loc["gradient_boosting_prior", "bias"],
            "holdout_wape_sistema_actual": tot.loc["sistema_actual", "wape"],
            "holdout_wape_naive": tot.loc["naive_estacional", "wape"],
            "holdout_delta_wape_vs_sistema": h_paired.iloc[0]["delta_wape"],
        })
        print("\n=== Holdout (evaluación honesta) ===")
        print(h_metrics.round(3).to_string(index=False))

    # ── 3 · Backtest walk-forward (opcional) ──────────────────────────────
    if args.with_backtest:
        panel_bt, cols_bt = build_features(tx, stores, cal, horizon=args.horizon)
        origins = [o for o in config.BACKTEST_ORIGINS
                   if train_start < pd.Timestamp(o)
                   and pd.Timestamp(o) + pd.Timedelta(days=args.horizon) <= train_end]
        if origins:
            logger.info("Backtest walk-forward con %s cortes…", len(origins))
            folds = make_folds(origins, args.horizon, panel_bt["date"].max())
            b_preds = run_backtest(panel_bt, cols_bt, folds, params=params)
            b_metrics = summarize(b_preds)
            b_paired = summarize_paired(b_preds)
            b_metrics.to_csv(run_dir / "backtest_metrics.csv", index=False)
            b_paired.to_csv(run_dir / "backtest_paired_diffs.csv", index=False)
            artifacts_mlflow["backtest_metrics"] = run_dir / "backtest_metrics.csv"
            tot = b_metrics[b_metrics["segmento"] == "total"].set_index("modelo")
            metrics_mlflow["backtest_wape"] = tot.loc["gradient_boosting_prior", "wape"]
            print("\n=== Backtest walk-forward ===")
            print(b_metrics.round(3).to_string(index=False))
        else:
            logger.warning("Ningún origen de backtest cabe en la ventana; se omite.")

    # ── 4 · Entrenamiento final con toda la ventana ───────────────────────
    panel, feature_cols = build_features(tx, stores, cal, horizon=args.horizon)
    fit_mask = (panel["date"] <= train_end) & panel[config.TARGET].notna()
    logger.info("Entrenamiento final: %s filas hasta %s…", fit_mask.sum(), train_end.date())
    model = DemandForecaster(feature_cols, args.horizon, params, prior_col=PRIOR_COL)
    model.fit(panel.loc[fit_mask], panel.loc[fit_mask, config.TARGET])

    # Serialización local: en el flow y en outputs/models/ (donde la busca
    # retail-predict), con los mismos metadatos de siempre.
    model_dir = create_run_dir("models")
    model.save(model_dir / "model.joblib")
    meta = run_metadata(
        horizon=args.horizon, feature_cols=feature_cols, model_params=params,
        target=config.TARGET, train_rows=int(fit_mask.sum()),
        train_start=train_start, train_end=train_end,
        holdout_days=args.holdout_days, flow_dir=str(run_dir),
    )
    save_json(meta, model_dir / "run_config.json")
    save_json(meta, run_dir / "run_config.json")
    artifacts_mlflow["run_config"] = run_dir / "run_config.json"
    artifacts_mlflow["event_priors"] = config.EVENT_PRIORS_CSV

    # ── 5 · Registro en MLflow (params, métricas, artefactos, signature) ──
    metrics_mlflow["train_rows"] = float(fit_mask.sum())
    params_mlflow = {
        **params, "horizon": args.horizon,
        "train_start": train_start.date(), "train_end": train_end.date(),
        "holdout_days": args.holdout_days, "lags": config.LAG_DAYS,
        "rolling_windows": config.ROLLING_WINDOWS,
        "n_features": len(feature_cols), "target": config.TARGET,
    }
    run_name = args.run_name or f"flow_{run_dir.name}"
    run_id = log_training_run(
        model, panel, feature_cols, params=params_mlflow, metrics=metrics_mlflow,
        artifacts=artifacts_mlflow, experiment=args.experiment,
        run_name=run_name, register=not args.no_register,
    )

    # ── 6 · Pronóstico operativo (inferencia) ─────────────────────────────
    if not args.no_forecast:
        forecast = generar_pronostico(model, tx, stores, cal, cutoff=train_end)
        forecast.to_csv(run_dir / "forecast.csv", index=False)
        print("\n=== Pronóstico operativo (unidades/día de la cadena) ===")
        print(forecast.groupby("date")["unidades_pronosticadas"].sum().round(0).to_string())

    print(f"\nFlujo completo ✓  artefactos: {run_dir}")
    print(f"Modelo desplegable: {model_dir / 'model.joblib'}")
    print(f"MLflow run_id: {run_id}  (explorar con `uv run mlflow ui`)")


if __name__ == "__main__":
    main()
