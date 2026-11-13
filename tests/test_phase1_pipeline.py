import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bess_trading.experiments.phase1 import run


def test_phase1_pipeline_writes_audited_artifacts(tmp_path: Path) -> None:
    index = pd.date_range("2023-01-01", periods=330 * 24, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    prices = (
        50
        + 25 * np.sin(2 * np.pi * hour / 24)
        + 5 * np.sin(2 * np.pi * np.arange(len(index)) / (24 * 7))
    )
    input_path = tmp_path / "prices.parquet"
    pd.DataFrame({"price_eur_mwh": prices}, index=index).to_parquet(input_path)
    output = tmp_path / "results"
    config = {
        "seed": 7,
        "experiment": {
            "inputs": [str(input_path)],
            "test_start": "2023-11-22T00:00:00Z",
            "test_end": "2023-11-25T00:00:00Z",
            "output": str(output),
            "minimum_oos_days": 2,
            "bootstrap_samples": 20,
        },
        "battery": {
            "power_mw": 1.0,
            "energy_mwh": 2.0,
            "min_soc_fraction": 0.1,
            "max_soc_fraction": 0.9,
            "initial_soc_fraction": 0.5,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
            "degradation_cost_eur_per_mwh": 5.0,
            "max_equivalent_cycles_per_day": 1.5,
        },
        "forecasting": {
            "train_days": 320,
            "retrain_days": 7,
            "issue_hour_local": 11,
            "model_max_iter": 5,
        },
        "optimization": {
            "risk_aversion": 0.35,
            "terminal_value_eur_per_mwh": 50.0,
            "solver": "HIGHS",
        },
    }
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run(config_path)

    assert summary["experiment"]["complete_dispatch_days"] >= 2
    assert summary["experiment"]["causality_audit_passed"] is True
    assert set(summary["strategy_metrics"]) == {
        "zero_trade",
        "rule_based",
        "perfect_foresight",
        "persistence",
        "hour_of_week",
        "linear",
        "ml_p50",
        "quantile_robust",
    }
    assert (output / "forecasts.parquet").exists()
    assert (output / "dispatch.parquet").exists()
    assert (output / "daily_pnl.csv").exists()
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert stored["experiment"]["causality_audit_passed"] is True
    dispatch = pd.read_parquet(output / "dispatch.parquet")
    assert not ((dispatch.charge_mw > 1e-8) & (dispatch.discharge_mw > 1e-8)).any()
