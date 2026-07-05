"""Pipeline de inferencia: carga el último modelo entrenado y pronostica los
próximos HORIZON días para las 480 series tienda × categoría.

Uso:  uv run retail-predict [--model-dir outputs/models/<run_id>] [--cutoff 2024-02-29]
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import config
from ..artifacts import create_run_dir, latest_run_dir, run_metadata, save_json, setup_logging
from ..calendar_utils import extend_calendar
from ..data import load_calendar, load_stores, load_transactions
from ..features import build_features
from ..models import DemandForecaster

logger = logging.getLogger("pipelines.predict")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directorio del modelo (default: última corrida de retail-train)")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Último día con datos a usar (default: máximo disponible)")
    args = parser.parse_args(argv)
    setup_logging()

    model_dir = Path(args.model_dir) if args.model_dir else latest_run_dir("models")
    model = DemandForecaster.load(model_dir / "model.joblib")
    logger.info("Modelo cargado de %s (horizonte=%s días)", model_dir, model.horizon)

    tx, stores, cal = load_transactions(), load_stores(), load_calendar()
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else tx["date"].max()
    tx = tx[tx["date"] <= cutoff]
    forecast_end = cutoff + pd.Timedelta(days=model.horizon)
    cal = extend_calendar(cal, until=forecast_end)
    cal = cal[cal["date"] <= forecast_end]

    panel, feature_cols = build_features(tx, stores, cal, horizon=model.horizon)
    if feature_cols != model.feature_cols:
        raise RuntimeError("Las features del panel no coinciden con las del modelo entrenado.")

    future = panel[panel["date"] > cutoff].copy()
    future["y_pred"] = model.predict(future)
    forecast = future[["date", "store_id", "category", "y_pred"]].rename(
        columns={"y_pred": "unidades_pronosticadas"}
    )

    run_dir = create_run_dir("forecasts")
    forecast.to_csv(run_dir / "forecast.csv", index=False)
    save_json(run_metadata(
        model_dir=str(model_dir), cutoff=cutoff,
        forecast_start=cutoff + pd.Timedelta(days=1), forecast_end=forecast_end,
        n_series=forecast[["store_id", "category"]].drop_duplicates().shape[0],
        n_rows=len(forecast),
    ), run_dir / "run_config.json")

    logger.info("Pronóstico de %s filas escrito en %s", len(forecast), run_dir / "forecast.csv")
    resumen = forecast.groupby("date")["unidades_pronosticadas"].sum().round(0)
    print("\n=== Unidades totales pronosticadas por día (cadena) ===")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
