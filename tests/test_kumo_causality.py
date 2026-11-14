import numpy as np
import pandas as pd
import pytest

from bess_trading.forecasting.kumo.calibration import RegimeResidualCalibrator
from bess_trading.forecasting.kumo.regimes import (
    RegimeThresholds,
    build_regime_snapshot,
    fit_regime_thresholds,
)


def test_regime_thresholds_ignore_post_cutoff_snapshots() -> None:
    index = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    snapshots = pd.DataFrame(
        {
            "recent_negative_share": np.linspace(0, 0.2, 100),
            "recent_volatility_eur_mwh": np.linspace(1, 20, 100),
        },
        index=index,
    )
    cutoff = index[80]
    first = fit_regime_thresholds(snapshots, cutoff)
    snapshots.loc[index >= cutoff] = 99999
    second = fit_regime_thresholds(snapshots, cutoff)
    assert first == second


def test_future_fundamental_availability_is_rejected() -> None:
    index = pd.date_range("2023-01-01", "2024-01-10", freq="h", tz="UTC")
    prices = pd.Series(50.0, index=index)
    horizon = pd.date_range("2024-01-04 23:00", periods=24, freq="h", tz="UTC")
    fundamentals = pd.DataFrame(
        {"residual_load_forecast_mw": 50_000.0}, index=horizon
    )
    availability = pd.Series(pd.Timestamp("2024-01-05 12:00:00Z"), index=horizon)
    with pytest.raises(ValueError, match="unavailable"):
        build_regime_snapshot(
            prices,
            "2024-01-05",
            RegimeThresholds(0.1, 10, residual_load_mw=60_000),
            fundamental_forecasts=fundamentals,
            fundamental_available_at=availability,
        )


def test_residual_calibration_rejects_final_test_observations() -> None:
    index = pd.date_range("2023-01-01", periods=500, freq="h", tz="UTC")
    actual = pd.Series(np.sin(np.arange(500)), index=index)
    predicted = actual + 0.1
    regimes = pd.Series("ordinary", index=index)
    calibrator = RegimeResidualCalibrator(minimum_regime_samples=50)
    with pytest.raises(ValueError, match="strictly pre-test"):
        calibrator.fit(
            actual,
            predicted,
            regimes,
            distribution_end=pd.Timestamp("2023-01-10", tz="UTC"),
            test_start=pd.Timestamp("2023-01-15", tz="UTC"),
        )
