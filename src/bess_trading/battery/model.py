"""Dispatch simulation independent of the optimizer."""

from __future__ import annotations

import pandas as pd

from bess_trading.config import BatteryConfig


def simulate_dispatch(
    charge_mw: pd.Series,
    discharge_mw: pd.Series,
    config: BatteryConfig,
    initial_soc_mwh: float | None = None,
) -> pd.DataFrame:
    """Recompute SOC and reject an infeasible external dispatch schedule."""
    if not charge_mw.index.equals(discharge_mw.index):
        raise ValueError("Charge and discharge indices differ")
    if (charge_mw < -1e-8).any() or (discharge_mw < -1e-8).any():
        raise ValueError("Power cannot be negative")
    if (charge_mw > config.power_mw + 1e-8).any() or (discharge_mw > config.power_mw + 1e-8).any():
        raise ValueError("Power limit exceeded")
    soc = initial_soc_mwh if initial_soc_mwh is not None else config.initial_soc_mwh
    rows: list[float] = []
    for charge, discharge in zip(charge_mw, discharge_mw, strict=True):
        soc += charge * config.charge_efficiency - discharge / config.discharge_efficiency
        if (
            soc < config.min_soc_fraction * config.energy_mwh - 1e-6
            or soc > config.max_soc_fraction * config.energy_mwh + 1e-6
        ):
            raise ValueError("SOC bound violated")
        rows.append(soc)
    return pd.DataFrame(
        {"charge_mw": charge_mw, "discharge_mw": discharge_mw, "soc_mwh": rows},
        index=charge_mw.index,
    )
