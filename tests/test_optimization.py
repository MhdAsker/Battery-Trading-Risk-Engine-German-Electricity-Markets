import pandas as pd
import pytest

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.config import BatteryConfig
from bess_trading.optimization.dispatch import optimize_dispatch, optimize_quantile_robust


def test_optimizer_buys_low_sells_high_and_returns_terminal_soc() -> None:
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    prices = pd.Series([0.0] * 12 + [100.0] * 12, index=idx)
    config = BatteryConfig()
    result = optimize_dispatch(prices, config, terminal_soc_fraction=0.5)
    assert result.charge_mw.iloc[:12].sum() > 0
    assert result.discharge_mw.iloc[12:].sum() > 0
    assert result.soc_mwh.iloc[-1] == pytest.approx(config.initial_soc_mwh, abs=1e-5)
    settled = settle_dispatch(result, prices, config)
    assert settled.net_revenue_eur.sum() > 0


def test_robust_requires_quantiles() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="require"):
        optimize_quantile_robust(pd.DataFrame({"p50": [1, 2]}, index=idx), BatteryConfig())
