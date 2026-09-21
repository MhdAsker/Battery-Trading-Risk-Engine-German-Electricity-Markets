"""Transparent convex BESS dispatch programs."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig


def optimize_dispatch(
    prices: pd.Series,
    config: BatteryConfig,
    terminal_soc_fraction: float | None = 0.5,
    solver: str = "CLARABEL",
) -> pd.DataFrame:
    """Maximize forecast energy margin net of throughput degradation.

    Charge and discharge are continuous. Simultaneous operation is economically
    dominated whenever degradation is positive; avoiding binaries keeps the model
    fast and auditable. A post-solve tolerance cleans numerical noise.
    """
    if prices.empty or prices.isna().any():
        raise ValueError("prices must be non-empty and complete")
    n = len(prices)
    charge, discharge = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    constraints = [
        soc[0] == config.initial_soc_mwh,
        soc[1:]
        == soc[:-1] + config.charge_efficiency * charge - discharge / config.discharge_efficiency,
        soc >= config.min_soc_fraction * config.energy_mwh,
        soc <= config.max_soc_fraction * config.energy_mwh,
        charge <= config.power_mw,
        discharge <= config.power_mw,
        cp.sum(charge + discharge)
        <= 2 * config.energy_mwh * config.max_equivalent_cycles_per_day * n / 24,
    ]
    if terminal_soc_fraction is not None:
        constraints.append(soc[-1] == terminal_soc_fraction * config.energy_mwh)
    p = prices.to_numpy(dtype=float)
    margin = p @ (discharge - charge)
    degradation = config.degradation_cost_eur_per_mwh * cp.sum(charge + discharge)
    problem = cp.Problem(cp.Maximize(margin - degradation), constraints)
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimization failed: {problem.status}")
    clean = lambda a: np.where(np.abs(a) < 1e-7, 0.0, a)  # noqa: E731
    return pd.DataFrame(
        {
            "charge_mw": clean(charge.value),
            "discharge_mw": clean(discharge.value),
            "soc_mwh": clean(soc.value[1:]),
            "forecast_price_eur_mwh": p,
        },
        index=prices.index,
    )


def optimize_quantile_robust(
    forecasts: pd.DataFrame,
    config: BatteryConfig,
    risk_aversion: float = 0.35,
    terminal_soc_fraction: float | None = 0.5,
    solver: str = "CLARABEL",
) -> pd.DataFrame:
    """Optimize against asymmetric conservative prices derived from P10/P50/P90.

    Charging is valued at an upper-tail price and discharging at a lower-tail price:
    buy_price=(1-lambda)P50+lambda*P90 and
    sell_price=(1-lambda)P50+lambda*P10. This is a robust, explainable downside
    adjustment; lambda=0 recovers median optimization.
    """
    needed = {"p10", "p50", "p90"}
    if not needed.issubset(forecasts):
        raise ValueError(f"Forecasts require {sorted(needed)}")
    if not 0 <= risk_aversion <= 1:
        raise ValueError("risk_aversion must be in [0, 1]")
    buy = (1 - risk_aversion) * forecasts.p50 + risk_aversion * forecasts.p90
    sell = (1 - risk_aversion) * forecasts.p50 + risk_aversion * forecasts.p10
    n = len(forecasts)
    charge, discharge, soc = (
        cp.Variable(n, nonneg=True),
        cp.Variable(n, nonneg=True),
        cp.Variable(n + 1),
    )
    constraints = [
        soc[0] == config.initial_soc_mwh,
        soc[1:]
        == soc[:-1] + config.charge_efficiency * charge - discharge / config.discharge_efficiency,
        soc >= config.min_soc_fraction * config.energy_mwh,
        soc <= config.max_soc_fraction * config.energy_mwh,
        charge <= config.power_mw,
        discharge <= config.power_mw,
        cp.sum(charge + discharge)
        <= 2 * config.energy_mwh * config.max_equivalent_cycles_per_day * n / 24,
    ]
    if terminal_soc_fraction is not None:
        constraints.append(soc[-1] == terminal_soc_fraction * config.energy_mwh)
    objective = (
        sell.to_numpy() @ discharge
        - buy.to_numpy() @ charge
        - config.degradation_cost_eur_per_mwh * cp.sum(charge + discharge)
    )
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimization failed: {problem.status}")
    return pd.DataFrame(
        {
            "charge_mw": charge.value,
            "discharge_mw": discharge.value,
            "soc_mwh": soc.value[1:],
            "forecast_price_eur_mwh": forecasts.p50,
        },
        index=forecasts.index,
    )
