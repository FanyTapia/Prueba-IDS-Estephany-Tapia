"""Particiones temporales walk-forward para el backtest."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Un corte del backtest: se entrena con date <= origin y se evalúa
    el pronóstico sobre (origin, origin + horizon]."""
    origin: pd.Timestamp
    horizon: int

    @property
    def test_start(self) -> pd.Timestamp:
        return self.origin + pd.Timedelta(days=1)

    @property
    def test_end(self) -> pd.Timestamp:
        return self.origin + pd.Timedelta(days=self.horizon)

    def masks(self, dates: pd.Series):
        train = dates <= self.origin
        test = (dates >= self.test_start) & (dates <= self.test_end)
        return train, test


def make_folds(origins: list, horizon: int, date_max) -> list:
    folds = []
    for o in origins:
        fold = Fold(origin=pd.Timestamp(o), horizon=horizon)
        if fold.test_end > pd.Timestamp(date_max):
            raise ValueError(
                f"El fold con origen {o} requiere datos hasta {fold.test_end.date()} "
                f"pero el panel termina en {pd.Timestamp(date_max).date()}."
            )
        folds.append(fold)
    return folds
