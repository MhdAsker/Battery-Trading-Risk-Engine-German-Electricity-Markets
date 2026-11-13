import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.experiments.phase2 import (
    calibrate_without_test_leakage,
    dispatch_probabilistic_strategies,
)


def test_phase2_calibration_and_dispatch_are_split_before_test() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 5, freq="h", tz="UTC")
    actual = pd.Series(50 + 30 * np.sin(2 * np.pi * index.hour / 24), index=index)
    forecasts = pd.DataFrame(
        {"p10": actual - 5, "p50": actual + 1, "p90": actual + 5}, index=index
    )
    test_start = pd.Timestamp("2024-01-03T00:00:00Z")
    calibrated, calibrator = calibrate_without_test_leakage(
        forecasts, actual, test_start, alpha=0.2
    )
    assert calibrated.index.min() == test_start
    assert calibrator.correction_ is not None
    residuals = actual.loc[actual.index < test_start] - forecasts.loc[
        forecasts.index < test_start, "p50"
    ]
    dispatch, scenarios = dispatch_probabilistic_strategies(
        calibrated,
        actual,
        residuals,
        BatteryConfig(max_equivalent_cycles_per_day=1),
        scenario_count=4,
        block_hours=24,
        risk_aversion=0.2,
        alpha=0.95,
        solver="HIGHS",
        seed=4,
    )
    assert set(dispatch.strategy) == {"p50", "stochastic_cvar"}
    assert scenarios.shape[1] == 4
    assert not ((dispatch.charge_mw > 1e-7) & (dispatch.discharge_mw > 1e-7)).any()
