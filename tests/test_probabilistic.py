import numpy as np
import pandas as pd

from bess_trading.forecasting.probabilistic import (
    ConformalQuantileCalibrator,
    ensemble_crps,
    pit_values,
    residual_block_scenarios,
    scenario_diagnostics,
    weighted_interval_score,
)


def test_conformal_calibration_expands_undercovered_interval() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    actual = pd.Series(np.arange(20.0), index=index)
    forecasts = pd.DataFrame({"p10": actual - 0.1, "p50": actual, "p90": actual + 0.1}, index=index)
    actual.iloc[-4:] += 5
    calibrated = (
        ConformalQuantileCalibrator().fit(actual, forecasts.p10, forecasts.p90).transform(forecasts)
    )
    assert (calibrated.p90 - calibrated.p10).mean() > (forecasts.p90 - forecasts.p10).mean()
    assert weighted_interval_score(actual, calibrated) >= 0
    assert pit_values(actual, calibrated).between(0, 1).all()


def test_block_scenarios_preserve_shape_and_seed() -> None:
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    point = pd.Series(50.0, index=index)
    residuals = pd.Series(np.tile([1.0, -1.0], 100))
    first = residual_block_scenarios(point, residuals, scenarios=10, seed=3)
    second = residual_block_scenarios(point, residuals, scenarios=10, seed=3)
    assert first.equals(second)
    assert first.shape == (24, 10)


def test_ensemble_crps_is_zero_for_perfect_ensemble() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    actual = pd.Series([1.0, 2.0, 3.0], index=index)
    scenarios = pd.concat([actual, actual], axis=1)
    assert ensemble_crps(actual, scenarios) == 0.0
    assert scenario_diagnostics(actual, scenarios)["lag1_absolute_error"] == 0.0
