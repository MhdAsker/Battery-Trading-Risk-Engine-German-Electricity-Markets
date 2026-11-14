"""Frozen 24-origin Kumo development benchmark with unchanged Phase 1 baselines."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bess_trading.config import BatteryConfig, load_yaml
from bess_trading.data.time import delivery_day_index
from bess_trading.experiments.kumo_decision_value import (
    dispatch_kumo_forecast_families,
    summarize_kumo_decision_value,
)
from bess_trading.experiments.phase1 import forecast_daily, load_prices
from bess_trading.forecasting.kumo.adapter import KumoForecastAdapter, KumoRuntimeConfig
from bess_trading.forecasting.kumo.diagnostics import economic_forecast_metrics
from bess_trading.forecasting.kumo.regimes import RegimeThresholds
from bess_trading.forecasting.kumo.retrieval import (
    RegimeAwareRetriever,
    blend_direct_and_retrieval,
)
from bess_trading.metrics.risk import paired_block_bootstrap_ci


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_baselines(
    prices: pd.Series,
    dates: list[object],
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, pd.Series], float]:
    baseline = settings["frozen_baselines"]
    if not bool(baseline["enabled"]):
        return {}, 0.0
    first = min(dates)
    last = max(dates)
    start = pd.Timestamp(first, tz="Europe/Berlin").tz_convert("UTC")
    end = (pd.Timestamp(last, tz="Europe/Berlin") + pd.Timedelta(days=1)).tz_convert("UTC")
    started = time.perf_counter()
    frame = forecast_daily(
        prices,
        test_start=start,
        test_end=end,
        train_days=int(baseline["train_days"]),
        retrain_days=int(baseline["retrain_days"]),
        issue_hour_local=int(baseline["issue_hour_local"]),
        seed=seed,
        model_max_iter=int(baseline["model_max_iter"]),
    )
    selected = delivery_day_index(dates[0], "1h")
    for day in dates[1:]:
        selected = selected.append(delivery_day_index(day, "1h"))
    if not selected.isin(frame.index).all():
        missing = selected[~selected.isin(frame.index)]
        raise ValueError(f"Frozen baselines are missing {len(missing)} selected intervals")
    frame = frame.loc[selected]
    return {
        "persistence": frame.persistence.rename("persistence"),
        "hour_of_week": frame.hour_of_week.rename("hour_of_week"),
        "linear": frame.linear.rename("linear"),
        "quantile_gbm": frame.p50.rename("quantile_gbm"),
    }, time.perf_counter() - started


def _daily_forecast_comparison(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    query_labels: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    days = sorted(set(actual.index.tz_convert("Europe/Berlin").date))
    for day in days:
        horizon = delivery_day_index(day, "1h")
        observed = actual.reindex(horizon)
        direct = forecasts["kumo_direct"].reindex(horizon)
        regime = forecasts["regime_kumo"].reindex(horizon)
        direct_error = (direct - observed).abs()
        regime_error = (regime - observed).abs()
        rows.append(
            {
                "delivery_day": str(day),
                "issue_time_regimes": query_labels[str(day)],
                "direct_mae_eur_mwh": float(direct_error.mean()),
                "regime_mae_eur_mwh": float(regime_error.mean()),
                "mae_improvement_eur_mwh": float(direct_error.mean() - regime_error.mean()),
                "actual_negative_hours": int((observed < 0).sum()),
                "actual_spread_eur_mwh": float(observed.max() - observed.min()),
                "direct_spread_error_eur_mwh": float(
                    abs((direct.max() - direct.min()) - (observed.max() - observed.min()))
                ),
                "regime_spread_error_eur_mwh": float(
                    abs((regime.max() - regime.min()) - (observed.max() - observed.min()))
                ),
            }
        )
    return pd.DataFrame(rows).set_index("delivery_day")


def _condition_summary(days: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    joined = days.join(
        (pnl.regime_kumo - pnl.kumo_direct).rename("revenue_improvement_eur"), how="left"
    )
    conditions = ["all", "negative_price_risk", "high_volatility", "weekend", "weekday"]
    rows = []
    for condition in conditions:
        subset = joined if condition == "all" else joined[
            joined.issue_time_regimes.str.contains(condition, regex=False)
        ]
        rows.append(
            {
                "condition": condition,
                "days": len(subset),
                "mean_mae_improvement_eur_mwh": float(subset.mae_improvement_eur_mwh.mean()),
                "mean_revenue_improvement_eur": float(subset.revenue_improvement_eur.mean()),
                "retrieval_mae_win_rate": float((subset.mae_improvement_eur_mwh > 0).mean()),
                "retrieval_revenue_win_rate": float((subset.revenue_improvement_eur > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def run(config_path: str | Path) -> dict[str, Any]:
    """Execute the pre-registered expanded development benchmark."""
    config_path = Path(config_path)
    settings = load_yaml(config_path)
    seed = int(settings["seed"])
    np.random.seed(seed)
    experiment = settings["experiment"]
    dates = [pd.Timestamp(value).date() for value in experiment["development_dates"]]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("Frozen development origins must be unique and chronological")
    if not 20 <= len(dates) <= 30:
        raise ValueError("Expanded development benchmark requires 20-30 frozen origins")
    test_start = pd.Timestamp(experiment["retrospective_test_start"])
    if any(pd.Timestamp(day, tz="Europe/Berlin").tz_convert("UTC") >= test_start for day in dates):
        raise ValueError("Development origin crosses the retrospective-test boundary")

    prices = load_prices([Path(value) for value in experiment["inputs"]])
    runtime = KumoRuntimeConfig(**settings["kumo"])
    adapter = KumoForecastAdapter(runtime, cache_dir=experiment["cache_dir"])
    retrieval_config = settings["external_retrieval"]
    retriever = RegimeAwareRetriever(
        RegimeThresholds(**retrieval_config["thresholds"]),
        context_hours=int(retrieval_config["context_hours"]),
        k=int(retrieval_config["k"]),
        temperature=float(retrieval_config["temperature"]),
        minimum_regime_neighbors=int(retrieval_config["minimum_regime_neighbors"]),
    )

    forecasts: dict[str, list[pd.Series]] = {"kumo_direct": [], "regime_kumo": []}
    diagnostics: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    query_labels: dict[str, str] = {}
    for day in dates:
        direct = adapter.forecast_delivery_day(prices, day)
        retrieval_started = time.perf_counter()
        external = retriever.forecast(prices, day, regime_conditioned=True)
        retrieval_seconds = time.perf_counter() - retrieval_started
        forecasts["kumo_direct"].append(direct.forecast)
        forecasts["regime_kumo"].append(
            blend_direct_and_retrieval(
                direct.forecast,
                external.forecast,
                float(retrieval_config["direct_weight"]),
            )
        )
        labels = "|".join(external.query_regimes)
        query_labels[str(day)] = labels
        diagnostic = external.diagnostics.copy()
        diagnostic["query_delivery_day"] = str(day)
        diagnostic["used_regime_filter"] = external.used_regime_filter
        diagnostics.append(diagnostic)
        runtime_rows.extend(
            [
                {
                    "delivery_day": str(day),
                    "method": "kumo_direct",
                    "seconds": direct.inference_seconds,
                    "cache_key": direct.cache_key,
                    "forecast_steps": direct.forecast_steps_from_cutoff,
                },
                {
                    "delivery_day": str(day),
                    "method": "external_regime_retrieval",
                    "seconds": retrieval_seconds,
                    "cache_key": "",
                    "forecast_steps": len(direct.forecast),
                },
            ]
        )

    point = {name: pd.concat(blocks).sort_index() for name, blocks in forecasts.items()}
    baselines, baseline_seconds = _selected_baselines(prices, dates, settings, seed)
    point.update(baselines)
    common = reduce(lambda left, right: left.intersection(right), [x.index for x in point.values()])
    actual = prices.reindex(common)
    if actual.isna().any():
        raise ValueError("Frozen panel contains missing settlement prices")
    spike_threshold = float(
        prices.loc[prices.index < test_start].dropna().quantile(float(experiment["spike_quantile"]))
    )
    forecast_metrics = {
        name: economic_forecast_metrics(
            actual, forecast.reindex(common), spike_threshold_eur_mwh=spike_threshold
        )
        for name, forecast in point.items()
    }

    battery = BatteryConfig(**settings["battery"])
    dispatch = dispatch_kumo_forecast_families(
        point,
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
        seed=seed,
    )
    daily = _daily_forecast_comparison(actual, point, query_labels)
    pnl = decision["daily_pnl"]
    daily = daily.join(
        (pnl.regime_kumo - pnl.kumo_direct).rename("revenue_improvement_eur"), how="left"
    )
    condition_summary = _condition_summary(daily.drop(columns="revenue_improvement_eur"), pnl)
    mae_inference = paired_block_bootstrap_ci(
        -daily.regime_mae_eur_mwh,
        -daily.direct_mae_eur_mwh,
        block_days=7,
        samples=int(experiment["bootstrap_samples"]),
        seed=seed,
    )
    mae_inference = {
        key.replace("eur", "eur_mwh"): value for key, value in mae_inference.items()
    }
    revenue_inference = decision["strategy_metrics"]["regime_kumo"][
        "paired_ci_vs_reference"
    ]
    gate = {
        "diagnostic_days": len(dates),
        "minimum_gate_days": int(experiment["minimum_gate_days"]),
        "mae_improvement_inference": mae_inference,
        "net_revenue_improvement_inference": revenue_inference,
        "proceed_to_native_darr": bool(
            len(dates) >= int(experiment["minimum_gate_days"])
            and revenue_inference["ci_lower_eur"] > 0
            and mae_inference["ci_lower_eur_mwh"] > 0
        ),
    }
    gate["decision"] = "proceed" if gate["proceed_to_native_darr"] else "do_not_proceed"

    manifest = Path("configs/kumo_development_origins_v1.csv")
    try:
        import psutil

        process_rss_mb = psutil.Process().memory_info().rss / 1024**2
        total_ram_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:  # pragma: no cover - optional resource telemetry
        process_rss_mb = total_ram_gb = float("nan")
    summary: dict[str, Any] = {
        "experiment": {
            "name": "kumo_development_24",
            "dates": [str(day) for day in dates],
            "selection_manifest_sha256": _sha256(manifest),
            "retrospective_test_start": str(test_start),
            "not_an_untouched_test": True,
            "battery": asdict(battery),
            "kumo": asdict(runtime),
            "native_darr_evaluated": False,
            "fine_tuning_evaluated": False,
        },
        "forecast_metrics": forecast_metrics,
        "strategy_metrics": decision["strategy_metrics"],
        "paired_mae_inference": mae_inference,
        "condition_summary": condition_summary.to_dict(orient="index"),
        "research_gate": gate,
        "resources": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pytorch_device": "cpu",
            "process_rss_mb_at_completion": process_rss_mb,
            "system_ram_gb": total_ram_gb,
            "baseline_forecasting_seconds": baseline_seconds,
            "kumo_direct_total_seconds": float(
                sum(row["seconds"] for row in runtime_rows if row["method"] == "kumo_direct")
            ),
            "external_retrieval_total_seconds": float(
                sum(
                    row["seconds"]
                    for row in runtime_rows
                    if row["method"] == "external_regime_retrieval"
                )
            ),
        },
    }
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point).to_parquet(output / "forecasts.parquet")
    dispatch.to_parquet(output / "dispatch.parquet")
    pnl.to_csv(output / "daily_pnl.csv")
    daily.to_csv(output / "daily_comparison.csv")
    condition_summary.to_csv(output / "condition_summary.csv")
    pd.concat(diagnostics).to_parquet(output / "retrieval_diagnostics.parquet")
    pd.DataFrame(runtime_rows).to_csv(output / "runtime.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
