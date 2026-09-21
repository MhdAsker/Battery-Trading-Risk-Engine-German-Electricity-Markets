"""Revenue settlement and daily rolling backtests."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from bess_trading.config import BatteryConfig


def settle_dispatch(
    dispatch: pd.DataFrame, realized_prices: pd.Series, config: BatteryConfig
) -> pd.DataFrame:
    """Settle a fixed schedule against realized prices, never forecast prices."""
    out = dispatch.copy()
    actual = realized_prices.reindex(out.index)
    if actual.isna().any():
        raise ValueError("Missing realized prices for dispatch timestamps")
    out["realized_price_eur_mwh"] = actual
    out["gross_revenue_eur"] = actual * (out.discharge_mw - out.charge_mw)
    out["throughput_mwh"] = out.charge_mw + out.discharge_mw
    out["degradation_cost_eur"] = config.degradation_cost_eur_per_mwh * out.throughput_mwh
    out["net_revenue_eur"] = out.gross_revenue_eur - out.degradation_cost_eur
    return out


def backtest_daily(
    forecasts: pd.Series,
    realized_prices: pd.Series,
    config: BatteryConfig,
    optimizer: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """Optimize independent UTC delivery days and concatenate settlements."""
    rows = []
    for _, prices in forecasts.groupby(forecasts.index.floor("D")):
        dispatch = optimizer(prices, config)
        rows.append(settle_dispatch(dispatch, realized_prices, config))
    return pd.concat(rows).sort_index() if rows else pd.DataFrame()
