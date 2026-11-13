"""Physically strict deterministic and risk-adjusted BESS dispatch programs."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.data.time import interval_hours


def _solve_dispatch(
    buy_prices: pd.Series,
    sell_prices: pd.Series,
    config: BatteryConfig,
    *,
    initial_soc_mwh: float | None = None,
    terminal_soc_mwh: float | None = None,
    terminal_value_eur_per_mwh: float = 0.0,
    solver: str = "HIGHS",
) -> pd.DataFrame:
    """Solve a mixed-integer dispatch with a shared inverter and explicit mode."""
    if buy_prices.empty or buy_prices.isna().any() or sell_prices.isna().any():
        raise ValueError("Prices must be non-empty and complete")
    if not buy_prices.index.equals(sell_prices.index):
        raise ValueError("Buy and sell price indices differ")
    dt = interval_hours(buy_prices.index)
    n = len(buy_prices)
    initial = config.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    minimum = config.min_soc_fraction * config.energy_mwh
    maximum = config.max_soc_fraction * config.energy_mwh
    if not minimum - 1e-9 <= initial <= maximum + 1e-9:
        raise ValueError("Initial SOC is outside configured limits")

    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    charging_mode = cp.Variable(n, boolean=True)
    soc = cp.Variable(n + 1)
    constraints = [
        soc[0] == initial,
        soc[1:]
        == soc[:-1]
        + dt * (config.charge_efficiency * charge - discharge / config.discharge_efficiency),
        soc >= minimum,
        soc <= maximum,
        charge <= config.power_mw * charging_mode,
        discharge <= config.power_mw * (1 - charging_mode),
        charge + discharge <= config.power_mw,
        dt * cp.sum(charge + discharge)
        <= 2 * config.energy_mwh * config.max_equivalent_cycles_per_day,
    ]
    if terminal_soc_mwh is not None:
        constraints.append(soc[-1] == terminal_soc_mwh)
    gross_margin = dt * (
        sell_prices.to_numpy(dtype=float) @ discharge - buy_prices.to_numpy(dtype=float) @ charge
    )
    degradation = config.degradation_cost_eur_per_mwh * dt * cp.sum(charge + discharge)
    terminal_value = terminal_value_eur_per_mwh * soc[-1]
    problem = cp.Problem(cp.Maximize(gross_margin - degradation + terminal_value), constraints)
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimization failed: {problem.status}")

    def clean(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        result[np.abs(result) < 1e-7] = 0.0
        return result

    result = pd.DataFrame(
        {
            "charge_mw": clean(charge.value),
            "discharge_mw": clean(discharge.value),
            "soc_start_mwh": clean(soc.value[:-1]),
            "soc_mwh": clean(soc.value[1:]),
            "forecast_price_eur_mwh": sell_prices.to_numpy(dtype=float),
            "interval_hours": dt,
        },
        index=buy_prices.index,
    )
    if ((result.charge_mw > 1e-6) & (result.discharge_mw > 1e-6)).any():
        raise RuntimeError("Solver returned simultaneous charge and discharge")
    return result


def optimize_dispatch(
    prices: pd.Series,
    config: BatteryConfig,
    terminal_soc_fraction: float | None = None,
    solver: str = "HIGHS",
    *,
    initial_soc_mwh: float | None = None,
    terminal_value_eur_per_mwh: float = 0.0,
) -> pd.DataFrame:
    """Optimize deterministic energy arbitrage for a regular market time unit."""
    terminal = None if terminal_soc_fraction is None else terminal_soc_fraction * config.energy_mwh
    return _solve_dispatch(
        prices,
        prices,
        config,
        initial_soc_mwh=initial_soc_mwh,
        terminal_soc_mwh=terminal,
        terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
        solver=solver,
    )


def optimize_quantile_robust(
    forecasts: pd.DataFrame,
    config: BatteryConfig,
    risk_aversion: float = 0.35,
    terminal_soc_fraction: float | None = None,
    solver: str = "HIGHS",
    *,
    initial_soc_mwh: float | None = None,
    terminal_value_eur_per_mwh: float = 0.0,
) -> pd.DataFrame:
    """Optimize conservative asymmetric prices derived from P10/P50/P90."""
    needed = {"p10", "p50", "p90"}
    if not needed.issubset(forecasts):
        raise ValueError(f"Forecasts require {sorted(needed)}")
    if not 0 <= risk_aversion <= 1:
        raise ValueError("risk_aversion must be in [0, 1]")
    buy = (1 - risk_aversion) * forecasts.p50 + risk_aversion * forecasts.p90
    sell = (1 - risk_aversion) * forecasts.p50 + risk_aversion * forecasts.p10
    terminal = None if terminal_soc_fraction is None else terminal_soc_fraction * config.energy_mwh
    result = _solve_dispatch(
        buy,
        sell,
        config,
        initial_soc_mwh=initial_soc_mwh,
        terminal_soc_mwh=terminal,
        terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
        solver=solver,
    )
    result["forecast_price_eur_mwh"] = forecasts.p50
    return result


def idle_dispatch(
    index: pd.DatetimeIndex, config: BatteryConfig, initial_soc_mwh: float | None = None
) -> pd.DataFrame:
    """Return the zero-trade baseline with a constant feasible SOC."""
    initial = config.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    dt = interval_hours(index)
    return pd.DataFrame(
        {
            "charge_mw": 0.0,
            "discharge_mw": 0.0,
            "soc_start_mwh": initial,
            "soc_mwh": initial,
            "forecast_price_eur_mwh": np.nan,
            "interval_hours": dt,
        },
        index=index,
    )
