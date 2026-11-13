import numpy as np
import pandas as pd

from bess_trading.battery.degradation import (
    piecewise_degradation_cost,
    rainflow_equivalent_cycles,
)
from bess_trading.config import BatteryConfig
from bess_trading.forecasting.models import LinearForecaster
from bess_trading.forecasting.nested import nested_rolling_select
from bess_trading.metrics.research import (
    attribute_dispatch_errors,
    classify_regimes,
    conditional_interval_coverage,
)
from bess_trading.optimization.frontier import (
    battery_sensitivity_configs,
    revenue_risk_frontier,
    run_sensitivity_study,
)


def test_nested_selection_uses_trailing_validation_and_refits() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
    x = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(2 * x.x + 1, index=index)
    model, params, scores = nested_rolling_select(
        x,
        y,
        [{"alpha": 0.01}, {"alpha": 1000.0}],
        lambda values: LinearForecaster(**values),
        validation_days=7,
    )
    assert params["alpha"] == 0.01
    assert len(scores) == 2
    assert len(model.predict(x.tail(2))) == 2


def test_regime_coverage_and_error_attribution() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "actual": [-10, 20, 30, 200, 40],
            "volatility": [1, 2, 3, 20, 4],
            "renewable_share_forecast": [0.8, 0.2, 0.3, 0.1, 0.4],
        },
        index=index,
    )
    regimes = classify_regimes(
        frame, {"spike": 150, "volatility": 10, "renewable_share": 0.7}
    )
    coverage = conditional_interval_coverage(
        frame.actual, frame.actual - 5, frame.actual + 5, regimes
    )
    assert coverage["negative_price"]["coverage"] == 1.0
    attributed = attribute_dispatch_errors(
        frame.actual,
        pd.Series([0, 20, 30, 50, 40], index=index),
        pd.Series([1, 0, 0, 0, 0], index=index),
        pd.Series([0, 0, 0, 1, 0], index=index),
    )
    assert attributed.peak_underestimation.any()


def test_degradation_and_frontier_outputs() -> None:
    assert piecewise_degradation_cost(pd.Series([1.0, 2.0]), 2, 4, 2) == 8
    assert rainflow_equivalent_cycles(pd.Series([0.0, 2.0, 0.0]), 2.0) == 1.0
    index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    scenarios = pd.DataFrame(
        {"low": [-20, -10, 80, 90], "high": [-10, 0, 100, 120]}, index=index
    )
    config = BatteryConfig(degradation_cost_eur_per_mwh=0, max_equivalent_cycles_per_day=1)
    frontier, schedules = revenue_risk_frontier(scenarios, config, [0.0, 0.5])
    assert set(frontier.columns) >= {"expected_revenue_eur", "cvar_loss_eur"}
    assert set(schedules) == {0.0, 0.5}
    variants = battery_sensitivity_configs(config)
    expected = {"duration_1h", "efficiency_0.95", "degradation_15", "cycle_limit_0.5"}
    assert expected <= set(variants)
    sensitivity = run_sensitivity_study(scenarios, config)
    assert set(sensitivity.sensitivity) == set(variants)
