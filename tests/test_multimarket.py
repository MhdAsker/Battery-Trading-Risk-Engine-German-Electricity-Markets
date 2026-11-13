from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_trading.audit.registry import DecisionAudit, ExperimentRegistry, canonical_hash
from bess_trading.config import BatteryConfig
from bess_trading.data.regelleistung import MarketDataProvenance, load_market_export
from bess_trading.forecasting.multimarket_scenarios import joint_activation_scenarios
from bess_trading.markets.products import Market, four_hour_block, gate_closure
from bess_trading.optimization.multimarket import (
    optimize_multimarket,
    reserve_stress_test,
    rolling_horizon_mpc,
)
from bess_trading.settlement.multimarket import (
    marginal_asset_values,
    settle_multimarket,
    value_metrics,
)


def market_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day_ahead_price_eur_mwh": np.r_[
                np.full(len(index) // 2, 20), np.full(len(index) // 2, 100)
            ],
            "intraday_price_eur_mwh": 50.0,
            "fcr_capacity_price_eur_mw_h": 20.0,
            "afrr_up_capacity_price_eur_mw_h": 0.0,
            "afrr_down_capacity_price_eur_mw_h": 0.0,
            "mfrr_up_capacity_price_eur_mw_h": 0.0,
            "mfrr_down_capacity_price_eur_mw_h": 0.0,
        },
        index=index,
    )


def test_product_rules_and_dst_safe_blocks() -> None:
    delivery = pd.Timestamp("2026-03-29T08:00:00Z")
    assert gate_closure(delivery, Market.FCR) == pd.Timestamp("2026-03-28T07:00:00Z")
    index = pd.date_range("2026-03-28T23:00:00Z", periods=92, freq="15min")
    blocks = four_hour_block(index)
    assert blocks.index.equals(index)
    assert blocks.nunique() == 6


def test_cooptimization_blocks_headroom_and_activation_stress() -> None:
    index = pd.date_range("2026-01-01", periods=16, freq="15min", tz="UTC")
    config = BatteryConfig(power_mw=1, energy_mwh=2, degradation_cost_eur_per_mwh=0)
    schedule, diagnostics = optimize_multimarket(market_frame(index), config)
    assert diagnostics.status == "optimal"
    assert (schedule.fcr_mw.round(6).isin([0.0, 1.0])).all()
    for _, block in schedule.groupby(four_hour_block(index)):
        assert block.fcr_mw.max() - block.fcr_mw.min() < 1e-6
    assert reserve_stress_test(schedule, config)["passed"] is True
    assert not ((schedule.charge_mw > 1e-7) & (schedule.discharge_mw > 1e-7)).any()


def test_outage_derates_dispatch_and_mpc_executes_first_action() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    availability = pd.Series([1.0, 0.0, 1.0, 1.0], index=index)
    schedule, diagnostics = rolling_horizon_mpc(
        market_frame(index), BatteryConfig(), horizon_intervals=2, availability=availability
    )
    assert len(schedule) == len(index) == len(diagnostics)
    assert schedule.loc[index[1], ["charge_mw", "discharge_mw"]].abs().max() < 1e-7


def test_settlement_reconciles_and_rejects_unproven_reserve_data() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    config = BatteryConfig(degradation_cost_eur_per_mwh=1)
    schedule, _ = optimize_multimarket(market_frame(index), config)
    actual = market_frame(index).assign(
        imbalance_volume_mwh=0.0,
        imbalance_price_eur_mwh=0.0,
        transaction_fee_eur=1.0,
        availability_penalty_eur=0.0,
        afrr_up_activation_mwh=0.0,
        afrr_down_activation_mwh=0.0,
        mfrr_up_activation_mwh=0.0,
        mfrr_down_activation_mwh=0.0,
        afrr_up_activation_price_eur_mwh=0.0,
        afrr_down_activation_price_eur_mwh=0.0,
        mfrr_up_activation_price_eur_mwh=0.0,
        mfrr_down_activation_price_eur_mwh=0.0,
    )
    record = MarketDataProvenance(
        "regelleistung.net", "2026-01-02T00:00:00Z", "x", "a"
    )
    provenance = {
        "reserve_capacity_awards": record,
        "reserve_activations": record,
        "activation_prices": record,
    }
    settled, summary = settle_multimarket(schedule, actual, config, provenance)
    components = [
        column
        for column in settled
        if column.endswith("_eur") and column != "total_value_eur"
    ]
    assert np.isclose(summary["total_value_eur"], settled[components].sum().sum())
    with pytest.raises(ValueError, match="provenance"):
        settle_multimarket(schedule, actual, config, {})


def test_market_export_provenance_and_decision_reconstruction(tmp_path: Path) -> None:
    export = tmp_path / "fcr.csv"
    export.write_text("timestamp_utc,value\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    frame, provenance = load_market_export(
        export,
        source="regelleistung.net",
        source_url="https://www.regelleistung.net/apps/datacenter/",
        downloaded_at_utc="2026-01-02T00:00:00Z",
    )
    assert frame.value.iloc[0] == 1
    assert len(provenance.sha256) == 64
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    config = {"horizon": 16}
    experiment_id = registry.create_experiment(config, "abc123", {"price": "v1"})
    audit = DecisionAudit(
        experiment_id=experiment_id,
        decision_time_utc="2026-01-01T00:00:00Z",
        delivery_start_utc="2026-01-01T00:15:00Z",
        market="intraday",
        model_version="v1",
        config_hash=canonical_hash(config),
        input_hashes={"market": provenance.sha256},
        solver="HIGHS",
        solver_status="optimal",
        objective_eur=10.0,
        decision={"intraday_mw": 1.0},
    )
    digest = registry.record_decision(audit)
    assert registry.reconstruct(digest) == audit


def test_evpi_vss_and_regret() -> None:
    index = pd.RangeIndex(3)
    metrics = value_metrics(
        pd.Series([10, 12, 11], index=index),
        pd.Series([8, 9, 10], index=index),
        pd.Series([13, 14, 15], index=index),
        pd.Series([7, 8, 9], index=index),
    )
    assert metrics == {
        "evpi_eur": 3.0,
        "vss_eur": 2.0,
        "regret_eur": 3.0,
        "incremental_value_vs_energy_only_eur": 3.0,
    }
    assert marginal_asset_values(100, 130, 120, power_increment_mw=0.5, energy_increment_mwh=1) == {
        "marginal_power_value_eur_per_mw": 60.0,
        "marginal_energy_value_eur_per_mwh": 20.0,
    }


def test_joint_activation_scenarios_preserve_column_pairing() -> None:
    columns = [
        f"{product}_{direction}_{field}"
        for product in ("afrr", "mfrr")
        for direction in ("up", "down")
        for field in ("activation_fraction", "activation_price_eur_mwh")
    ]
    history = pd.DataFrame({column: np.arange(40.0) for column in columns})
    horizon = pd.date_range("2026-01-01", periods=16, freq="15min", tz="UTC")
    output = joint_activation_scenarios(history, horizon, scenarios=3, seed=1)
    assert set(output) == set(columns)
    first = output[columns[0]]
    for column in columns[1:]:
        pd.testing.assert_frame_equal(first, output[column])
