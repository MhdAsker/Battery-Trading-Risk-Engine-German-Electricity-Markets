"""Fifteen-minute energy and balancing-capacity co-optimization."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.data.time import interval_hours
from bess_trading.markets.products import four_hour_block

CAPACITY_COLUMNS = {
    "fcr_mw": "fcr_capacity_price_eur_mw_h",
    "afrr_up_mw": "afrr_up_capacity_price_eur_mw_h",
    "afrr_down_mw": "afrr_down_capacity_price_eur_mw_h",
    "mfrr_up_mw": "mfrr_up_capacity_price_eur_mw_h",
    "mfrr_down_mw": "mfrr_down_capacity_price_eur_mw_h",
}


@dataclass(frozen=True)
class SolverDiagnostics:
    status: str
    objective_eur: float
    solver: str
    iterations: int | None
    solve_time_seconds: float | None


def _check_inputs(market: pd.DataFrame, availability: pd.Series) -> None:
    required = {"day_ahead_price_eur_mwh", "intraday_price_eur_mwh", *CAPACITY_COLUMNS.values()}
    missing = required - set(market)
    if missing:
        raise ValueError(f"Missing co-optimization inputs: {sorted(missing)}")
    if market[list(required)].isna().any().any():
        raise ValueError("Co-optimization inputs must be complete")
    if not market.index.equals(availability.index):
        raise ValueError("Availability index differs from market horizon")
    if not ((availability >= 0) & (availability <= 1)).all():
        raise ValueError("Availability must be in [0, 1]")


def optimize_multimarket(
    market: pd.DataFrame,
    config: BatteryConfig,
    *,
    initial_soc_mwh: float | None = None,
    availability: pd.Series | None = None,
    committed_day_ahead_mw: pd.Series | None = None,
    reserve_duration_hours: float = 0.25,
    transaction_fee_eur_mwh: float = 0.0,
    solver: str = "HIGHS",
) -> tuple[pd.DataFrame, SolverDiagnostics]:
    """Co-optimize energy, intraday recourse and reserve capacity.

    Positive energy means discharge/injection. Capacity products are constrained to
    four-hour blocks and either zero or at least the official 1 MW minimum bid.
    """
    if market.empty:
        raise ValueError("Market horizon is empty")
    dt = interval_hours(market.index)
    n = len(market)
    availability = (
        pd.Series(1.0, index=market.index) if availability is None else availability.astype(float)
    )
    _check_inputs(market, availability)
    power = config.power_mw * availability.to_numpy()
    initial = config.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    da_fixed = committed_day_ahead_mw is not None

    charge, discharge = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    intraday_buy, intraday_sell = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    mode, soc = cp.Variable(n, boolean=True), cp.Variable(n + 1)
    reserve = {name: cp.Variable(n, nonneg=True) for name in CAPACITY_COLUMNS}
    if da_fixed:
        assert committed_day_ahead_mw is not None
        if not committed_day_ahead_mw.index.equals(market.index):
            raise ValueError("Committed day-ahead index differs from market horizon")
        da_net: cp.Expression = cp.Constant(committed_day_ahead_mw.to_numpy(dtype=float))
    else:
        da_sell, da_buy = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
        da_net = da_sell - da_buy

    net = da_net + intraday_sell - intraday_buy
    constraints = [
        discharge - charge == net,
        soc[0] == initial,
        soc[1:]
        == soc[:-1]
        + dt * (config.charge_efficiency * charge - discharge / config.discharge_efficiency),
        soc >= config.min_soc_fraction * config.energy_mwh,
        soc <= config.max_soc_fraction * config.energy_mwh,
        charge <= cp.multiply(power, mode),
        discharge <= cp.multiply(power, 1 - mode),
        charge + discharge <= power,
        intraday_buy <= power,
        intraday_sell <= power,
    ]
    if not da_fixed:
        constraints += [da_buy <= power, da_sell <= power]
    up = reserve["fcr_mw"] + reserve["afrr_up_mw"] + reserve["mfrr_up_mw"]
    down = reserve["fcr_mw"] + reserve["afrr_down_mw"] + reserve["mfrr_down_mw"]
    constraints += [
        discharge + up <= power,
        charge + down <= power,
        soc[:-1] - config.min_soc_fraction * config.energy_mwh
        >= reserve_duration_hours * up / config.discharge_efficiency,
        config.max_soc_fraction * config.energy_mwh - soc[:-1]
        >= reserve_duration_hours * down * config.charge_efficiency,
    ]

    blocks = four_hour_block(market.index)
    for _name, variable in reserve.items():
        enabled = cp.Variable(n, boolean=True)
        constraints += [variable >= enabled, variable <= cp.multiply(power, enabled)]
        for block in blocks.unique():
            positions = np.flatnonzero((blocks == block).to_numpy())
            constraints += [variable[positions] == variable[positions[0]]]

    day_ahead_revenue: cp.Expression | float = 0.0
    if not da_fixed:
        day_ahead_revenue = dt * market.day_ahead_price_eur_mwh.to_numpy() @ da_net
    intraday_net = intraday_sell - intraday_buy
    intraday_revenue = dt * market.intraday_price_eur_mwh.to_numpy() @ intraday_net
    capacity_revenue = sum(
        dt * market[column].to_numpy() @ reserve[name]
        for name, column in CAPACITY_COLUMNS.items()
    )
    throughput = dt * cp.sum(charge + discharge)
    fees = transaction_fee_eur_mwh * dt * cp.sum(intraday_buy + intraday_sell)
    degradation = config.degradation_cost_eur_per_mwh * throughput
    problem = cp.Problem(
        cp.Maximize(day_ahead_revenue + intraday_revenue + capacity_revenue - fees - degradation),
        constraints,
    )
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Multi-market optimization failed: {problem.status}")

    result = pd.DataFrame(
        {
            "day_ahead_mw": np.asarray(da_net.value).reshape(-1),
            "intraday_mw": intraday_net.value,
            "charge_mw": charge.value,
            "discharge_mw": discharge.value,
            "soc_start_mwh": soc.value[:-1],
            "soc_mwh": soc.value[1:],
            "availability": availability,
            "interval_hours": dt,
            **{name: variable.value for name, variable in reserve.items()},
        },
        index=market.index,
    )
    stats = problem.solver_stats
    diagnostics = SolverDiagnostics(
        status=str(problem.status),
        objective_eur=float(problem.value),
        solver=solver,
        iterations=stats.num_iters,
        solve_time_seconds=stats.solve_time,
    )
    return result, diagnostics


def rolling_horizon_mpc(
    market: pd.DataFrame,
    config: BatteryConfig,
    *,
    horizon_intervals: int = 16,
    availability: pd.Series | None = None,
    transaction_fee_eur_mwh: float = 0.0,
    solver: str = "HIGHS",
) -> tuple[pd.DataFrame, list[SolverDiagnostics]]:
    """Re-optimize every quarter hour and execute only the first control action."""
    if horizon_intervals < 1:
        raise ValueError("horizon_intervals must be positive")
    availability = pd.Series(1.0, index=market.index) if availability is None else availability
    state = config.initial_soc_mwh
    decisions, diagnostics = [], []
    for position in range(len(market)):
        stop = min(len(market), position + horizon_intervals)
        horizon = market.iloc[position:stop]
        schedule, diagnostic = optimize_multimarket(
            horizon,
            config,
            initial_soc_mwh=state,
            availability=availability.reindex(horizon.index),
            transaction_fee_eur_mwh=transaction_fee_eur_mwh,
            solver=solver,
        )
        first = schedule.iloc[[0]].copy()
        state = float(first.soc_mwh.iloc[0])
        first["mpc_issue_time_utc"] = market.index[position]
        first["mpc_horizon_intervals"] = len(horizon)
        decisions.append(first)
        diagnostics.append(diagnostic)
    return pd.concat(decisions), diagnostics


def reserve_stress_test(
    schedule: pd.DataFrame,
    config: BatteryConfig,
    *,
    sustained_intervals: int = 1,
) -> dict[str, float | bool]:
    """Check full upward and downward activation against power and energy margins."""
    up = schedule.fcr_mw + schedule.afrr_up_mw + schedule.mfrr_up_mw
    down = schedule.fcr_mw + schedule.afrr_down_mw + schedule.mfrr_down_mw
    dt = schedule.interval_hours * sustained_intervals
    available_power = config.power_mw * schedule.availability
    upward_soc = schedule.soc_start_mwh - dt * up / config.discharge_efficiency
    downward_soc = schedule.soc_start_mwh + dt * down * config.charge_efficiency
    power_ok = bool(
        ((schedule.discharge_mw + up <= available_power + 1e-6)
        & (schedule.charge_mw + down <= available_power + 1e-6)).all()
    )
    energy_ok = bool(
        ((upward_soc >= config.min_soc_fraction * config.energy_mwh - 1e-6)
        & (downward_soc <= config.max_soc_fraction * config.energy_mwh + 1e-6)).all()
    )
    return {
        "passed": power_ok and energy_ok,
        "power_feasible": power_ok,
        "energy_feasible": energy_ok,
        "minimum_upward_stress_soc_mwh": float(upward_soc.min()),
        "maximum_downward_stress_soc_mwh": float(downward_soc.max()),
    }
