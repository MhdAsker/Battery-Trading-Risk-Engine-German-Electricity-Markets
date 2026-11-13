"""Calibration, proper scores and coherent trajectory scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalQuantileCalibrator:
    """Symmetric split-conformal correction for a central quantile interval."""

    alpha: float = 0.2
    correction_: float | None = None

    def fit(
        self, actual: pd.Series, lower: pd.Series, upper: pd.Series
    ) -> ConformalQuantileCalibrator:
        scores = np.maximum(
            lower.to_numpy() - actual.to_numpy(), actual.to_numpy() - upper.to_numpy()
        )
        level = min(1.0, np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
        self.correction_ = float(np.quantile(scores, level, method="higher"))
        return self

    def transform(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if self.correction_ is None:
            raise RuntimeError("Calibrator is not fitted")
        result = forecasts.copy()
        result["p10"] -= self.correction_
        result["p90"] += self.correction_
        return result


def weighted_interval_score(
    actual: pd.Series, forecasts: pd.DataFrame, alpha: float = 0.2
) -> float:
    """Central-interval WIS using P50 and a P10–P90 interval."""
    y, lower, median, upper = (
        actual.to_numpy(),
        forecasts.p10.to_numpy(),
        forecasts.p50.to_numpy(),
        forecasts.p90.to_numpy(),
    )
    interval = (
        upper
        - lower
        + 2 / alpha * (lower - y) * (y < lower)
        + 2 / alpha * (y - upper) * (y > upper)
    )
    return float(np.mean((0.5 * np.abs(y - median) + alpha / 2 * interval) / 1.5))


def pit_values(actual: pd.Series, forecasts: pd.DataFrame) -> pd.Series:
    """Approximate PIT values by piecewise-linear interpolation across P10/P50/P90."""
    quantiles = np.array([0.1, 0.5, 0.9])
    values = [
        np.interp(y, row, quantiles, left=0.0, right=1.0)
        for y, row in zip(actual, forecasts[["p10", "p50", "p90"]].to_numpy(), strict=True)
    ]
    return pd.Series(values, index=actual.index, name="pit")


def ensemble_crps(actual: pd.Series, scenarios: pd.DataFrame) -> float:
    """Continuous ranked probability score for an empirical forecast ensemble."""
    if not actual.index.equals(scenarios.index):
        raise ValueError("Actual and scenario indices differ")
    ensemble = scenarios.to_numpy(dtype=float)
    observations = actual.to_numpy(dtype=float)[:, None]
    first = np.mean(np.abs(ensemble - observations), axis=1)
    second = 0.5 * np.mean(
        np.abs(ensemble[:, :, None] - ensemble[:, None, :]), axis=(1, 2)
    )
    return float(np.mean(first - second))


def scenario_diagnostics(
    actual: pd.Series, scenarios: pd.DataFrame, tail_quantile: float = 0.9
) -> dict[str, float]:
    """Compare lag-one dependence and upper/lower-tail frequency to observations."""
    if not actual.index.equals(scenarios.index):
        raise ValueError("Actual and scenario indices differ")
    actual_lag1 = float(actual.autocorr(1))
    scenario_lag1 = float(scenarios.apply(lambda x: x.autocorr(1)).mean())
    lower, upper = actual.quantile(1 - tail_quantile), actual.quantile(tail_quantile)
    actual_tail = float(((actual <= lower) | (actual >= upper)).mean())
    scenario_tail = float(((scenarios <= lower) | (scenarios >= upper)).to_numpy().mean())
    return {
        "actual_lag1_correlation": actual_lag1,
        "mean_scenario_lag1_correlation": scenario_lag1,
        "lag1_absolute_error": abs(actual_lag1 - scenario_lag1),
        "actual_tail_frequency": actual_tail,
        "scenario_tail_frequency": scenario_tail,
        "tail_frequency_absolute_error": abs(actual_tail - scenario_tail),
    }


def residual_block_scenarios(
    point_forecast: pd.Series,
    residual_history: pd.Series,
    scenarios: int = 100,
    block_hours: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample circular residual blocks, preserving within-block temporal dependence."""
    residuals = residual_history.dropna().to_numpy(dtype=float)
    if len(residuals) < block_hours:
        raise ValueError("Residual history is shorter than one scenario block")
    rng = np.random.default_rng(seed)
    output = np.empty((len(point_forecast), scenarios))
    for scenario in range(scenarios):
        draw: list[float] = []
        while len(draw) < len(point_forecast):
            start = int(rng.integers(0, len(residuals)))
            draw.extend(residuals[(start + np.arange(block_hours)) % len(residuals)])
        output[:, scenario] = point_forecast.to_numpy() + np.asarray(draw[: len(point_forecast)])
    return pd.DataFrame(
        output, index=point_forecast.index, columns=[f"scenario_{i:03d}" for i in range(scenarios)]
    )
