"""Revenue-risk frontier construction and scenario settlement."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.optimization.dispatch import optimize_stochastic_cvar


def scenario_revenues(
    dispatch: pd.DataFrame, scenarios: pd.DataFrame, config: BatteryConfig
) -> pd.Series:
    """Settle one non-anticipative schedule against every price scenario."""
    net_export = dispatch.discharge_mw - dispatch.charge_mw
    gross = scenarios.mul(net_export * dispatch.interval_hours, axis=0).sum(axis=0)
    throughput = ((dispatch.charge_mw + dispatch.discharge_mw) * dispatch.interval_hours).sum()
    return gross - config.degradation_cost_eur_per_mwh * throughput


def revenue_risk_frontier(
    scenarios: pd.DataFrame,
    config: BatteryConfig,
    risk_aversions: Iterable[float],
    *,
    alpha: float = 0.95,
    solver: str = "HIGHS",
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Solve schedules over lambda values and report expected revenue and CVaR loss."""
    rows, schedules = [], {}
    for value in risk_aversions:
        risk_aversion = float(value)
        schedule = optimize_stochastic_cvar(
            scenarios,
            config,
            risk_aversion=risk_aversion,
            alpha=alpha,
            solver=solver,
        )
        revenues = scenario_revenues(schedule, scenarios, config)
        losses = -revenues
        var = float(losses.quantile(alpha))
        rows.append(
            {
                "risk_aversion": risk_aversion,
                "expected_revenue_eur": float(revenues.mean()),
                "revenue_std_eur": float(revenues.std(ddof=1)),
                "var_loss_eur": var,
                "cvar_loss_eur": float(losses[losses >= var].mean()),
                "expected_cycles": float(
                    ((schedule.charge_mw + schedule.discharge_mw) * schedule.interval_hours).sum()
                    / (2 * config.energy_mwh)
                ),
            }
        )
        schedules[risk_aversion] = schedule
    return pd.DataFrame(rows).sort_values("risk_aversion"), schedules


def battery_sensitivity_configs(
    base: BatteryConfig,
    *,
    durations: Iterable[float] = (1.0, 2.0, 4.0),
    efficiencies: Iterable[float] = (0.85, 0.9, 0.95),
    degradation_costs: Iterable[float] = (0.0, 5.0, 15.0),
    cycle_limits: Iterable[float] = (0.5, 1.0, 1.5),
) -> dict[str, BatteryConfig]:
    """Create one-at-a-time physical/economic sensitivity configurations."""
    output: dict[str, BatteryConfig] = {}
    for duration in durations:
        output[f"duration_{duration:g}h"] = replace(base, energy_mwh=base.power_mw * duration)
    for efficiency in efficiencies:
        output[f"efficiency_{efficiency:.2f}"] = replace(
            base, charge_efficiency=efficiency, discharge_efficiency=efficiency
        )
    for cost in degradation_costs:
        output[f"degradation_{cost:g}"] = replace(base, degradation_cost_eur_per_mwh=cost)
    for limit in cycle_limits:
        output[f"cycle_limit_{limit:g}"] = replace(
            base, max_equivalent_cycles_per_day=limit
        )
    return output


def run_sensitivity_study(
    scenarios: pd.DataFrame,
    base: BatteryConfig,
    *,
    risk_aversion: float = 0.2,
    alpha: float = 0.95,
    solver: str = "HIGHS",
) -> pd.DataFrame:
    """Solve one-at-a-time battery sensitivities on a common scenario set."""
    rows = []
    for name, config in battery_sensitivity_configs(base).items():
        frontier, _ = revenue_risk_frontier(
            scenarios, config, [risk_aversion], alpha=alpha, solver=solver
        )
        rows.append({"sensitivity": name, **frontier.iloc[0].to_dict()})
    return pd.DataFrame(rows).sort_values("sensitivity").reset_index(drop=True)
