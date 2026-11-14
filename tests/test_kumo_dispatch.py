import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.experiments.kumo_decision_value import dispatch_kumo_forecast_families


def test_kumo_strategies_share_asset_rules_but_keep_independent_soc() -> None:
    index = pd.date_range("2023-01-01 23:00", periods=48, freq="h", tz="UTC")
    actual = pd.Series(50 + 30 * np.sin(2 * np.pi * np.arange(48) / 24), index=index)
    forecasts = {
        "kumo_direct": actual.shift(1).bfill(),
        "kumo_retrieval": actual.shift(2).bfill(),
    }
    config = BatteryConfig(max_equivalent_cycles_per_day=1)
    dispatch = dispatch_kumo_forecast_families(
        forecasts,
        actual,
        config,
        include_perfect_foresight=True,
    )
    assert set(dispatch.strategy) == {
        "zero_trade",
        "kumo_direct",
        "kumo_retrieval",
        "perfect_foresight",
    }
    assert not ((dispatch.charge_mw > 1e-7) & (dispatch.discharge_mw > 1e-7)).any()
    assert dispatch.groupby("strategy").soc_mwh.min().ge(0.2 - 1e-6).all()
    assert dispatch.groupby("strategy").soc_mwh.max().le(1.8 + 1e-6).all()
