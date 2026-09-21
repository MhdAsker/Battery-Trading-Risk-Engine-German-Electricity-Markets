import pandas as pd
import pytest

from bess_trading.battery.model import simulate_dispatch
from bess_trading.config import BatteryConfig


def test_simulator_applies_efficiency() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    result = simulate_dispatch(
        pd.Series([0.5, 0.0], idx), pd.Series([0.0, 0.2], idx), BatteryConfig()
    )
    assert result.soc_mwh.iloc[0] == pytest.approx(1.475)
    assert result.soc_mwh.iloc[1] == pytest.approx(1.475 - 0.2 / 0.95)


def test_simulator_rejects_power_violation() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Power limit"):
        simulate_dispatch(pd.Series([1.1], idx), pd.Series([0.0], idx), BatteryConfig())
