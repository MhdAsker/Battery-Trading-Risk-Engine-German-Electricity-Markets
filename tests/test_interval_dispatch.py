import pandas as pd
import pytest

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.config import BatteryConfig
from bess_trading.optimization.dispatch import optimize_dispatch


@pytest.mark.parametrize(("frequency", "periods"), [("15min", 96), ("30min", 48), ("1h", 24)])
def test_energy_accounting_for_supported_intervals(frequency: str, periods: int) -> None:
    index = pd.date_range("2024-01-01", periods=periods, freq=frequency, tz="UTC")
    prices = pd.Series([0.0] * (periods // 2) + [100.0] * (periods - periods // 2), index=index)
    config = BatteryConfig()
    result = optimize_dispatch(prices, config, terminal_soc_fraction=0.5)
    expected_soc = result.soc_start_mwh + result.interval_hours * (
        config.charge_efficiency * result.charge_mw
        - result.discharge_mw / config.discharge_efficiency
    )
    assert result.soc_mwh.to_numpy() == pytest.approx(expected_soc.to_numpy(), abs=1e-6)
    settled = settle_dispatch(result, prices, config)
    assert settled.throughput_mwh.sum() == pytest.approx(
        ((result.charge_mw + result.discharge_mw) * result.interval_hours).sum()
    )


def test_negative_prices_cannot_create_simultaneous_cycling() -> None:
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    prices = pd.Series(-1000.0, index=index)
    result = optimize_dispatch(prices, BatteryConfig(), terminal_value_eur_per_mwh=0.0)
    assert not ((result.charge_mw > 1e-8) & (result.discharge_mw > 1e-8)).any()
    assert ((result.charge_mw + result.discharge_mw) <= 1.0 + 1e-8).all()


def test_soc_can_continue_between_delivery_days() -> None:
    first_index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    second_index = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")
    first = optimize_dispatch(pd.Series(-20.0, index=first_index), BatteryConfig())
    second = optimize_dispatch(
        pd.Series(100.0, index=second_index),
        BatteryConfig(),
        initial_soc_mwh=float(first.soc_mwh.iloc[-1]),
    )
    assert second.soc_start_mwh.iloc[0] == pytest.approx(first.soc_mwh.iloc[-1])
