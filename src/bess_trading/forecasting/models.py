"""Forecast models sharing a small fit/predict interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PersistenceForecaster:
    """Same local hour yesterday baseline."""

    def predict(self, history: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
        values = history.reindex(horizon_index - pd.Timedelta(days=1)).to_numpy()
        return pd.Series(values, index=horizon_index, name="p50")


class HourOfWeekForecaster:
    """Expanding historical mean for each local hour of week."""

    def predict(self, history: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
        local_history = history.index.tz_convert("Europe/Berlin")
        keys = local_history.weekday * 24 + local_history.hour
        means = history.groupby(keys).mean()
        local_horizon = horizon_index.tz_convert("Europe/Berlin")
        horizon_keys = local_horizon.weekday * 24 + local_horizon.hour
        return pd.Series(horizon_keys.map(means).to_numpy(), index=horizon_index, name="p50")


@dataclass
class LinearForecaster:
    """Regularized linear benchmark with standardized leakage-safe inputs."""

    alpha: float = 10.0
    model: object = field(init=False)

    def __post_init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LinearForecaster:
        self.model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(x), index=x.index, name="linear")


@dataclass
class QuantileGBM:
    """Scikit-learn quantile GBM fallback requiring no compiled optional package."""

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    random_state: int = 42
    max_iter: int = 250
    models: dict[float, HistGradientBoostingRegressor] = field(default_factory=dict, init=False)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> QuantileGBM:
        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                max_leaf_nodes=24,
                learning_rate=0.06,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
            model.fit(x, y)
            self.models[q] = model
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model is not fitted")
        output = pd.DataFrame(
            {f"p{int(q * 100):02d}": model.predict(x) for q, model in self.models.items()},
            index=x.index,
        )
        # Independent quantile models can cross; sorting is transparent and stable.
        output.iloc[:, :] = np.sort(output.to_numpy(), axis=1)
        return output
