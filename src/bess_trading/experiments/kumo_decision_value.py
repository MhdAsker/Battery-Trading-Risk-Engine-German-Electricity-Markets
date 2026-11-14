"""Kumo forecast-family dispatch using the unchanged project battery optimizers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.battery.model import simulate_dispatch
from bess_trading.config import BatteryConfig
from bess_trading.data.time import delivery_day_index
from bess_trading.metrics.core import trading_metrics
from bess_trading.metrics.risk import daily_pnl, daily_risk_metrics, paired_block_bootstrap_ci
from bess_trading.optimization.dispatch import (
    idle_dispatch,
    optimize_dispatch,
    optimize_quantile_robust,
    optimize_stochastic_cvar,
)


def dispatch_kumo_forecast_families(
    point_forecasts: Mapping[str, pd.Series],
    actual: pd.Series,
    config: BatteryConfig,
    *,
    probabilistic_forecasts: Mapping[str, pd.DataFrame] | None = None,
    scenario_forecasts: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    risk_aversion: float = 0.2,
    cvar_alpha: float = 0.95,
    terminal_value_eur_per_mwh: float = 50.0,
    solver: str = "HIGHS",
    include_perfect_foresight: bool = True,
) -> pd.DataFrame:
    """Dispatch identical assets with independent, continuously evolving SOC states."""
    probabilistic_forecasts = probabilistic_forecasts or {}
    scenario_forecasts = scenario_forecasts or {}
    names = ["zero_trade", *point_forecasts, *probabilistic_forecasts, *scenario_forecasts]
    if include_perfect_foresight:
        names.append("perfect_foresight")
    if len(names) != len(set(names)):
        raise ValueError("Strategy names must be unique across forecast families")
    states = {name: config.initial_soc_mwh for name in names}
    all_indices = [forecast.index for forecast in point_forecasts.values()]
    all_indices += [forecast.index for forecast in probabilistic_forecasts.values()]
    if not all_indices:
        raise ValueError("At least one forecast family is required")
    candidate_days = sorted(
        set.intersection(
            *[
                set(index.tz_convert("Europe/Berlin").date)
                for index in all_indices
            ]
        )
    )
    outputs: list[pd.DataFrame] = []
    for day in candidate_days:
        horizon = delivery_day_index(day, "1h")
        realized = actual.reindex(horizon)
        if realized.isna().any():
            continue
        if any(not horizon.isin(index).all() for index in all_indices):
            continue
        schedules: dict[str, pd.DataFrame] = {
            "zero_trade": idle_dispatch(horizon, config, states["zero_trade"])
        }
        for name, forecast in point_forecasts.items():
            schedules[name] = optimize_dispatch(
                forecast.reindex(horizon),
                config,
                initial_soc_mwh=states[name],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
                solver=solver,
            )
        for name, forecast in probabilistic_forecasts.items():
            schedules[name] = optimize_quantile_robust(
                forecast.reindex(horizon),
                config,
                risk_aversion=risk_aversion,
                initial_soc_mwh=states[name],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
                solver=solver,
            )
        day_key = str(day)
        for name, by_day in scenario_forecasts.items():
            if day_key not in by_day:
                raise ValueError(f"Missing scenario matrix for {name} on {day_key}")
            schedules[name] = optimize_stochastic_cvar(
                by_day[day_key],
                config,
                risk_aversion=risk_aversion,
                alpha=cvar_alpha,
                initial_soc_mwh=states[name],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
                solver=solver,
            )
        if include_perfect_foresight:
            schedules["perfect_foresight"] = optimize_dispatch(
                realized,
                config,
                initial_soc_mwh=states["perfect_foresight"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
                solver=solver,
            )
        for name, schedule in schedules.items():
            independent = simulate_dispatch(
                schedule.charge_mw,
                schedule.discharge_mw,
                config,
                initial_soc_mwh=states[name],
            )
            if not independent.soc_mwh.round(6).equals(schedule.soc_mwh.round(6)):
                raise AssertionError(f"Independent SOC verification failed for {name} on {day}")
            states[name] = float(schedule.soc_mwh.iloc[-1])
            settled = settle_dispatch(schedule, actual, config)
            settled["strategy"] = name
            settled["delivery_day_local"] = day_key
            outputs.append(settled)
    if not outputs:
        raise ValueError("No common complete delivery days for Kumo dispatch comparison")
    return pd.concat(outputs).sort_index()


def summarize_kumo_decision_value(
    dispatch: pd.DataFrame,
    config: BatteryConfig,
    *,
    reference_strategy: str,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize revenue, utilization, risk, regret and paired inference."""
    pnl = daily_pnl(dispatch)
    if reference_strategy not in pnl:
        raise ValueError(f"Reference strategy {reference_strategy!r} is absent")
    perfect_total = (
        float(dispatch.loc[dispatch.strategy == "perfect_foresight", "net_revenue_eur"].sum())
        if "perfect_foresight" in set(dispatch.strategy)
        else float("nan")
    )
    metrics: dict[str, Any] = {}
    for name, group in dispatch.groupby("strategy"):
        base = trading_metrics(group, config.energy_mwh)
        cycles = base["equivalent_cycles"]
        metrics[name] = {
            **base,
            **daily_risk_metrics(pnl[name]),
            "revenue_per_equivalent_cycle_eur": (
                base["net_revenue_eur"] / cycles if cycles > 0 else 0.0
            ),
            "regret_vs_perfect_eur": perfect_total - base["net_revenue_eur"],
            "paired_ci_vs_reference": paired_block_bootstrap_ci(
                pnl[name],
                pnl[reference_strategy],
                block_days=min(7, max(1, len(pnl) // 2)),
                samples=bootstrap_samples,
                seed=seed,
            ),
        }
    return {"strategy_metrics": metrics, "daily_pnl": pnl}
