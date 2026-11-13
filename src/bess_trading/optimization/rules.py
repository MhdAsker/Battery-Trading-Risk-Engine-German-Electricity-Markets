"""Transparent non-ML strategy baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.data.time import interval_hours


def rule_based_dispatch(
    prices: pd.Series,
    config: BatteryConfig,
    initial_soc_mwh: float | None = None,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.DataFrame:
    """Greedily charge low-price and discharge high-price intervals."""
    dt = interval_hours(prices.index)
    low, high = prices.quantile([low_quantile, high_quantile])
    minimum = config.min_soc_fraction * config.energy_mwh
    maximum = config.max_soc_fraction * config.energy_mwh
    soc = config.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    throughput_budget = 2 * config.energy_mwh * config.max_equivalent_cycles_per_day
    charge_values: list[float] = []
    discharge_values: list[float] = []
    soc_start_values: list[float] = []
    soc_values: list[float] = []
    for price in prices:
        soc_start_values.append(soc)
        charge = discharge = 0.0
        remaining_power = min(config.power_mw, throughput_budget / dt)
        if price <= low and soc < maximum:
            charge = min(remaining_power, (maximum - soc) / (dt * config.charge_efficiency))
            soc += dt * config.charge_efficiency * charge
        elif price >= high and soc > minimum:
            discharge = min(
                remaining_power,
                (soc - minimum) * config.discharge_efficiency / dt,
            )
            soc -= dt * discharge / config.discharge_efficiency
        throughput_budget -= dt * (charge + discharge)
        charge_values.append(charge)
        discharge_values.append(discharge)
        soc_values.append(soc)
    return pd.DataFrame(
        {
            "charge_mw": charge_values,
            "discharge_mw": discharge_values,
            "soc_start_mwh": soc_start_values,
            "soc_mwh": soc_values,
            "forecast_price_eur_mwh": prices,
            "interval_hours": dt,
        },
        index=prices.index,
    ).replace({-0.0: 0.0, np.nan: np.nan})
