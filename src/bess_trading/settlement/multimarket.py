"""Exact multi-market value decomposition using observed settlement data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig
from bess_trading.data.regelleistung import MarketDataProvenance, require_real_reserve_data

RESERVE_SETTLEMENT_COLUMNS = {
    "fcr_capacity_price_eur_mw_h",
    "afrr_up_capacity_price_eur_mw_h",
    "afrr_down_capacity_price_eur_mw_h",
    "mfrr_up_capacity_price_eur_mw_h",
    "mfrr_down_capacity_price_eur_mw_h",
    "afrr_up_activation_mwh",
    "afrr_down_activation_mwh",
    "mfrr_up_activation_mwh",
    "mfrr_down_activation_mwh",
    "afrr_up_activation_price_eur_mwh",
    "afrr_down_activation_price_eur_mwh",
    "mfrr_up_activation_price_eur_mwh",
    "mfrr_down_activation_price_eur_mwh",
}


def settle_multimarket(
    schedule: pd.DataFrame,
    actual: pd.DataFrame,
    config: BatteryConfig,
    provenance: dict[str, MarketDataProvenance],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Settle every value component and assert interval/aggregate reconciliation."""
    if not schedule.index.equals(actual.index):
        raise ValueError("Schedule and settlement indices differ")
    require_real_reserve_data(actual, provenance, RESERVE_SETTLEMENT_COLUMNS)
    required = {
        "day_ahead_price_eur_mwh",
        "intraday_price_eur_mwh",
        "imbalance_volume_mwh",
        "imbalance_price_eur_mwh",
        "transaction_fee_eur",
        "availability_penalty_eur",
    }
    missing = required - set(actual)
    if missing or actual[list(required)].isna().any().any():
        raise ValueError(f"Missing energy settlement observations: {sorted(missing)}")
    dt = schedule.interval_hours
    up_capacity = schedule.afrr_up_mw + schedule.mfrr_up_mw
    down_capacity = schedule.afrr_down_mw + schedule.mfrr_down_mw
    up_activation = actual.afrr_up_activation_mwh + actual.mfrr_up_activation_mwh
    down_activation = actual.afrr_down_activation_mwh + actual.mfrr_down_activation_mwh
    if ((up_activation > up_capacity * dt + 1e-6) | (
        down_activation > down_capacity * dt + 1e-6
    )).any():
        raise ValueError("Observed activation exceeds awarded reserve deliverability")
    result = pd.DataFrame(index=schedule.index)
    result["day_ahead_eur"] = schedule.day_ahead_mw * dt * actual.day_ahead_price_eur_mwh
    result["intraday_eur"] = schedule.intraday_mw * dt * actual.intraday_price_eur_mwh
    result["reserve_capacity_eur"] = dt * (
        schedule.fcr_mw * actual.fcr_capacity_price_eur_mw_h
        + schedule.afrr_up_mw * actual.afrr_up_capacity_price_eur_mw_h
        + schedule.afrr_down_mw * actual.afrr_down_capacity_price_eur_mw_h
        + schedule.mfrr_up_mw * actual.mfrr_up_capacity_price_eur_mw_h
        + schedule.mfrr_down_mw * actual.mfrr_down_capacity_price_eur_mw_h
    )
    result["activation_eur"] = (
        actual.afrr_up_activation_mwh * actual.afrr_up_activation_price_eur_mwh
        + actual.afrr_down_activation_mwh * actual.afrr_down_activation_price_eur_mwh
        + actual.mfrr_up_activation_mwh * actual.mfrr_up_activation_price_eur_mwh
        + actual.mfrr_down_activation_mwh * actual.mfrr_down_activation_price_eur_mwh
    )
    result["imbalance_eur"] = (
        actual.imbalance_volume_mwh * actual.imbalance_price_eur_mwh
    )
    result["fees_eur"] = -actual.transaction_fee_eur.abs()
    result["availability_penalty_eur"] = -actual.availability_penalty_eur.abs()
    throughput = dt * (schedule.charge_mw + schedule.discharge_mw)
    result["degradation_eur"] = -config.degradation_cost_eur_per_mwh * throughput
    components = [
        "day_ahead_eur",
        "intraday_eur",
        "reserve_capacity_eur",
        "activation_eur",
        "imbalance_eur",
        "fees_eur",
        "availability_penalty_eur",
        "degradation_eur",
    ]
    result["total_value_eur"] = result[components].sum(axis=1)
    summary = {column: float(result[column].sum()) for column in components}
    summary["total_value_eur"] = float(result.total_value_eur.sum())
    if not np.isclose(summary["total_value_eur"], sum(summary[c] for c in components)):
        raise AssertionError("Market-level settlement does not reconcile")
    return result, summary


def value_metrics(
    stochastic_value: pd.Series,
    expected_value_solution: pd.Series,
    perfect_information_value: pd.Series,
    deterministic_baseline: pd.Series,
) -> dict[str, float]:
    """Calculate EVPI, VSS and regret from date-aligned strategy values."""
    aligned = pd.concat(
        [
            stochastic_value.rename("stochastic"),
            expected_value_solution.rename("expected_value"),
            perfect_information_value.rename("perfect"),
            deterministic_baseline.rename("baseline"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    means = aligned.mean()
    return {
        "evpi_eur": float(means.perfect - means.stochastic),
        "vss_eur": float(means.stochastic - means.expected_value),
        "regret_eur": float(means.perfect - means.stochastic),
        "incremental_value_vs_energy_only_eur": float(means.stochastic - means.baseline),
    }


def marginal_asset_values(
    base_value_eur: float,
    additional_power_value_eur: float,
    additional_energy_value_eur: float,
    *,
    power_increment_mw: float,
    energy_increment_mwh: float,
) -> dict[str, float]:
    """Finite-difference marginal values from otherwise identical experiments."""
    if power_increment_mw <= 0 or energy_increment_mwh <= 0:
        raise ValueError("Asset increments must be positive")
    return {
        "marginal_power_value_eur_per_mw": (
            additional_power_value_eur - base_value_eur
        )
        / power_increment_mw,
        "marginal_energy_value_eur_per_mwh": (
            additional_energy_value_eur - base_value_eur
        )
        / energy_increment_mwh,
    }
