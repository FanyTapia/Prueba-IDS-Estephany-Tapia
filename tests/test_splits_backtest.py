import pandas as pd
import pytest

from src import config
from src.backtest import run_backtest, summarize
from src.features import build_features
from src.splits import make_folds


def test_folds_no_traslapan_train_y_test():
    dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
    folds = make_folds(["2023-06-01", "2023-09-01"], horizon=14, date_max=dates.max())
    for fold in folds:
        train, test = fold.masks(dates)
        assert not (train & test).any()
        assert dates[train].max() == fold.origin
        assert dates[test].min() == fold.origin + pd.Timedelta(days=1)
        assert dates[test].max() == fold.origin + pd.Timedelta(days=14)


def test_fold_fuera_de_rango_es_rechazado():
    with pytest.raises(ValueError, match="requiere datos hasta"):
        make_folds(["2023-12-25"], horizon=14, date_max="2023-12-31")


def test_backtest_end_to_end_en_dataset_sintetico(toy_data):
    """El pipeline completo corre en el mini-dataset y el campeón produce
    predicciones finitas y razonables para las 4 series."""
    tx, stores, cal = toy_data
    panel, cols = build_features(tx, stores, cal)
    folds = make_folds(["2023-03-17"], horizon=14, date_max=panel["date"].max())

    preds = run_backtest(panel, cols, folds, params={**config.MODEL_PARAMS, "max_iter": 50})
    assert set(preds["modelo"]) == {"gradient_boosting", "naive_estacional", "sistema_actual"}
    assert preds["y_pred"].notna().all()
    assert (preds["y_pred"] >= 0).all()

    metrics = summarize(preds)
    champion_total = metrics[(metrics["modelo"] == "gradient_boosting")
                             & (metrics["segmento"] == "total")]
    assert champion_total["wape"].iloc[0] < 0.5
