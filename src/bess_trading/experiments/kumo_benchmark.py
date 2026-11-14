"""Development-gated Kumo direct, native DARR and regime-retrieval diagnostic."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from bess_trading.config import BatteryConfig, load_yaml
from bess_trading.data.time import price_publication_times
from bess_trading.experiments.kumo_decision_value import (
    dispatch_kumo_forecast_families,
    summarize_kumo_decision_value,
)
from bess_trading.experiments.phase1 import load_prices
from bess_trading.forecasting.kumo.adapter import KumoForecastAdapter, KumoRuntimeConfig
from bess_trading.forecasting.kumo.diagnostics import economic_forecast_metrics
from bess_trading.forecasting.kumo.regimes import RegimeThresholds
from bess_trading.forecasting.kumo.retrieval import (
    RegimeAwareRetriever,
    blend_direct_and_retrieval,
)
from bess_trading.forecasting.models import HourOfWeekForecaster, PersistenceForecaster


def run(config_path: str | Path) -> dict[str, Any]:
    """Run a small pre-test diagnostic; this is not the frozen 2024 evaluation."""
    settings = load_yaml(config_path)
    experiment = settings["experiment"]
    test_start = pd.Timestamp(experiment["retrospective_test_start"])
    dates = [pd.Timestamp(value).date() for value in experiment["development_dates"]]
    if any(pd.Timestamp(day, tz="Europe/Berlin").tz_convert("UTC") >= test_start for day in dates):
        raise ValueError("Kumo configuration selection dates must precede retrospective test")
    prices = load_prices([Path(value) for value in experiment["inputs"]])
    runtime = KumoRuntimeConfig(**settings["kumo"])
    adapter = KumoForecastAdapter(runtime, cache_dir=experiment["cache_dir"])
    retrieval_settings = settings["external_retrieval"]
    retriever = RegimeAwareRetriever(
        RegimeThresholds(**retrieval_settings["thresholds"]),
        context_hours=int(retrieval_settings["context_hours"]),
        k=int(retrieval_settings["k"]),
        temperature=float(retrieval_settings["temperature"]),
        minimum_regime_neighbors=int(retrieval_settings["minimum_regime_neighbors"]),
    )
    forecasts: dict[str, list[pd.Series]] = {
        "persistence": [],
        "hour_of_week": [],
        "kumo_direct": [],
        "regime_kumo": [],
    }
    if bool(settings["native_darr"]["enabled"]):
        forecasts["kumo_native_darr"] = []
    retrieval_diagnostics = []
    runtime_rows = []
    for day in dates:
        direct = adapter.forecast_delivery_day(prices, day)
        horizon = direct.forecast.index
        forecasts["kumo_direct"].append(direct.forecast)
        forecasts["persistence"].append(PersistenceForecaster().predict(prices, horizon))
        issue = direct.issue_time_utc
        known = prices[(price_publication_times(prices.index) <= issue)]
        forecasts["hour_of_week"].append(HourOfWeekForecaster().predict(known, horizon))
        external = retriever.forecast(prices, day, regime_conditioned=True)
        forecasts["regime_kumo"].append(
            blend_direct_and_retrieval(
                direct.forecast,
                external.forecast,
                float(retrieval_settings["direct_weight"]),
            )
        )
        diagnostic = external.diagnostics.copy()
        diagnostic["query_delivery_day"] = str(day)
        diagnostic["used_regime_filter"] = external.used_regime_filter
        retrieval_diagnostics.append(diagnostic)
        runtime_rows.append(
            {
                "delivery_day": str(day),
                "method": "kumo_direct",
                "inference_seconds": direct.inference_seconds,
                "cache_key": direct.cache_key,
            }
        )
        if "kumo_native_darr" in forecasts:
            darr = adapter.forecast_delivery_day(prices, day, native_darr=True)
            forecasts["kumo_native_darr"].append(darr.forecast)
            runtime_rows.append(
                {
                    "delivery_day": str(day),
                    "method": "kumo_native_darr",
                    "inference_seconds": darr.inference_seconds,
                    "cache_key": darr.cache_key,
                }
            )
    combined = {name: pd.concat(blocks).sort_index() for name, blocks in forecasts.items()}
    common = reduce(
        lambda left, right: left.intersection(right),
        [forecast.index for forecast in combined.values()],
    )
    actual = prices.reindex(common)
    if actual.isna().any():
        raise ValueError("Development diagnostic has missing settlement prices")
    spike_threshold = float(
        prices.loc[prices.index < test_start].dropna().quantile(
            float(experiment["spike_quantile"])
        )
    )
    forecast_metrics = {
        name: economic_forecast_metrics(
            actual,
            forecast.reindex(common),
            spike_threshold_eur_mwh=spike_threshold,
        )
        for name, forecast in combined.items()
    }
    battery = BatteryConfig(**settings["battery"])
    dispatch = dispatch_kumo_forecast_families(
        combined,
        prices,
        battery,
        terminal_value_eur_per_mwh=float(settings["optimization"]["terminal_value_eur_per_mwh"]),
        solver=str(settings["optimization"]["solver"]),
    )
    decision = summarize_kumo_decision_value(
        dispatch,
        battery,
        reference_strategy="kumo_direct",
        bootstrap_samples=int(experiment["bootstrap_samples"]),
        seed=int(settings["seed"]),
    )
    direct_revenue = decision["strategy_metrics"]["kumo_direct"]["net_revenue_eur"]
    regime_revenue = decision["strategy_metrics"]["regime_kumo"]["net_revenue_eur"]
    gate = {
        "diagnostic_days": len(dates),
        "minimum_gate_days": int(experiment["minimum_gate_days"]),
        "regime_retrieval_mae_improvement_eur_mwh": (
            forecast_metrics["kumo_direct"]["mae_eur_mwh"]
            - forecast_metrics["regime_kumo"]["mae_eur_mwh"]
        ),
        "regime_retrieval_net_revenue_improvement_eur": regime_revenue - direct_revenue,
    }
    gate["proceed_to_full_development"] = bool(
        len(dates) >= int(experiment["minimum_gate_days"])
        and (
            gate["regime_retrieval_mae_improvement_eur_mwh"] > 0
            or gate["regime_retrieval_net_revenue_improvement_eur"] > 0
        )
    )
    gate["decision"] = (
        "proceed" if gate["proceed_to_full_development"] else "insufficient_evidence"
    )
    summary = {
        "experiment": {
            "name": "kumo_development_diagnostic",
            "dates": [str(day) for day in dates],
            "retrospective_test_start": str(test_start),
            "not_an_untouched_test": True,
            "battery": asdict(battery),
            "kumo": asdict(runtime),
        },
        "forecast_metrics": forecast_metrics,
        "strategy_metrics": decision["strategy_metrics"],
        "research_feasibility_gate": gate,
    }
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combined).to_parquet(output / "forecasts.parquet")
    dispatch.to_parquet(output / "dispatch.parquet")
    decision["daily_pnl"].to_csv(output / "daily_pnl.csv")
    pd.concat(retrieval_diagnostics).to_parquet(output / "retrieval_diagnostics.parquet")
    pd.DataFrame(runtime_rows).to_csv(output / "runtime.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
