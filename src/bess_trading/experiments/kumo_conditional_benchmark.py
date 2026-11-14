"""Pre-registered conditional-retrieval Kumo development and validation study."""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from bess_trading.config import BatteryConfig, load_yaml
from bess_trading.data.time import (
    delivery_day_index,
    forecast_issue_time,
    price_publication_times,
)
from bess_trading.experiments.kumo_decision_value import (
    dispatch_kumo_forecast_families,
    summarize_kumo_decision_value,
)
from bess_trading.experiments.kumo_expanded_benchmark import _selected_baselines
from bess_trading.experiments.phase1 import load_prices
from bess_trading.forecasting.kumo.adapter import (
    KumoForecastAdapter,
    KumoRuntimeConfig,
    latest_complete_segment,
)
from bess_trading.forecasting.kumo.diagnostics import economic_forecast_metrics
from bess_trading.forecasting.kumo.policy import ConditionalRetrievalPolicy
from bess_trading.forecasting.kumo.regimes import (
    RegimeThresholds,
    build_regime_snapshot,
)
from bess_trading.forecasting.kumo.retrieval import (
    OptimizedRegimeAwareRetriever,
    blend_direct_and_retrieval,
)
from bess_trading.metrics.risk import paired_block_bootstrap_ci


def eligible_origins(
    prices: pd.Series,
    start: str,
    end: str,
    *,
    kumo_context_hours: int,
    retrieval_context_hours: int,
    maximum_forecast_steps: int,
) -> tuple[list[object], Counter[str]]:
    """Select origins using data availability only, never delivery-price outcomes."""
    publication = pd.Series(price_publication_times(prices.index), index=prices.index)
    dates: list[object] = []
    exclusions: Counter[str] = Counter()
    for stamp in pd.date_range(start, end, freq="D", inclusive="left"):
        day = stamp.date()
        issue = forecast_issue_time(day, 11)
        known = prices[(publication <= issue).to_numpy()]
        try:
            segment = latest_complete_segment(known, kumo_context_hours)
        except ValueError:
            exclusions["no_complete_kumo_context"] += 1
            continue
        horizon = delivery_day_index(day, "1h")
        path = pd.date_range(
            segment.index.max() + pd.Timedelta(hours=1), horizon.max(), freq="h", tz="UTC"
        )
        if len(path) > maximum_forecast_steps:
            exclusions["forecast_path_exceeds_limit"] += 1
            continue
        query = known.dropna().sort_index().iloc[-retrieval_context_hours:]
        expected = pd.date_range(query.index.min(), query.index.max(), freq="h", tz="UTC")
        if len(query) < retrieval_context_hours or not query.index.equals(expected):
            exclusions["retrieval_context_gap"] += 1
            continue
        if prices.reindex(horizon).isna().any():
            exclusions["incomplete_settlement"] += 1
            continue
        dates.append(day)
    return dates, exclusions


def _daily_frame(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    pnl: pd.DataFrame,
    labels: dict[str, str],
    activated: dict[str, bool],
    spike_threshold: float,
) -> pd.DataFrame:
    rows = []
    for day in sorted(set(actual.index.tz_convert("Europe/Berlin").date)):
        horizon = delivery_day_index(day, "1h")
        observed = actual.reindex(horizon)
        row: dict[str, object] = {
            "delivery_day": str(day),
            "issue_time_regimes": labels[str(day)],
            "retrieval_activated": activated[str(day)],
            "actual_negative_hours": int((observed < 0).sum()),
            "actual_spike_hours": int((observed >= spike_threshold).sum()),
            "actual_spread_eur_mwh": float(observed.max() - observed.min()),
        }
        for name, forecast in forecasts.items():
            block = forecast.reindex(horizon)
            row[f"{name}_mae_eur_mwh"] = float((block - observed).abs().mean())
            row[f"{name}_spread_error_eur_mwh"] = float(
                abs((block.max() - block.min()) - (observed.max() - observed.min()))
            )
            row[f"{name}_revenue_eur"] = float(pnl.loc[str(day), name])
        rows.append(row)
    return pd.DataFrame(rows).set_index("delivery_day")


def _paired_inference(
    daily: pd.DataFrame,
    candidate: str,
    reference: str,
    block_sizes: list[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for block in block_sizes:
        revenue = paired_block_bootstrap_ci(
            daily[f"{candidate}_revenue_eur"],
            daily[f"{reference}_revenue_eur"],
            block_days=block,
            samples=samples,
            seed=seed,
        )
        mae = paired_block_bootstrap_ci(
            -daily[f"{candidate}_mae_eur_mwh"],
            -daily[f"{reference}_mae_eur_mwh"],
            block_days=block,
            samples=samples,
            seed=seed,
        )
        output[str(block)] = {"revenue": revenue, "mae_improvement": mae}
    return output


def _regime_summary(daily: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "all": pd.Series(True, index=daily.index),
        "high_volatility": daily.issue_time_regimes.str.contains("high_volatility"),
        "negative_price_risk": daily.issue_time_regimes.str.contains("negative_price_risk"),
        "overlap": daily.issue_time_regimes.str.contains("high_volatility")
        & daily.issue_time_regimes.str.contains("negative_price_risk"),
        "ordinary": ~daily.retrieval_activated,
        "activated": daily.retrieval_activated,
        "weekend": daily.issue_time_regimes.str.contains("weekend"),
        "weekday": daily.issue_time_regimes.str.contains("weekday"),
    }
    rows = []
    for name, mask in conditions.items():
        subset = daily[mask]
        if subset.empty:
            continue
        always_mae_gain = (
            subset.kumo_direct_mae_eur_mwh - subset.regime_kumo_mae_eur_mwh
        )
        conditional_mae_gain = (
            subset.kumo_direct_mae_eur_mwh - subset.conditional_regime_kumo_mae_eur_mwh
        )
        always_revenue_gain = (
            subset.regime_kumo_revenue_eur - subset.kumo_direct_revenue_eur
        )
        conditional_revenue_gain = (
            subset.conditional_regime_kumo_revenue_eur - subset.kumo_direct_revenue_eur
        )
        rows.append(
            {
                "condition": name,
                "origins": len(subset),
                "activation_rate": float(subset.retrieval_activated.mean()),
                "always_mae_improvement_eur_mwh": float(always_mae_gain.mean()),
                "conditional_mae_improvement_eur_mwh": float(conditional_mae_gain.mean()),
                "always_revenue_improvement_eur": float(always_revenue_gain.mean()),
                "conditional_revenue_improvement_eur": float(
                    conditional_revenue_gain.mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def _physical_verification(dispatch: pd.DataFrame, battery: BatteryConfig) -> dict[str, Any]:
    low = battery.min_soc_fraction * battery.energy_mwh
    high = battery.max_soc_fraction * battery.energy_mwh
    daily_efc = dispatch.groupby(["strategy", "delivery_day_local"]).throughput_mwh.sum() / (
        2 * battery.energy_mwh
    )
    return {
        "simultaneous_charge_discharge_intervals": int(
            ((dispatch.charge_mw > 1e-7) & (dispatch.discharge_mw > 1e-7)).sum()
        ),
        "max_shared_inverter_excess_mw": float(
            (dispatch.charge_mw + dispatch.discharge_mw - battery.power_mw).clip(lower=0).max()
        ),
        "max_soc_below_minimum_mwh": float((low - dispatch.soc_mwh).clip(lower=0).max()),
        "max_soc_above_maximum_mwh": float((dispatch.soc_mwh - high).clip(lower=0).max()),
        "maximum_daily_efc": float(daily_efc.max()),
        "configured_daily_efc_limit": battery.max_equivalent_cycles_per_day,
    }


def _evaluate_split(
    name: str,
    dates: list[object],
    all_forecasts: dict[str, pd.Series],
    prices: pd.Series,
    labels: dict[str, str],
    activated: dict[str, bool],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    index = pd.DatetimeIndex([])
    for day in dates:
        index = index.append(delivery_day_index(day, "1h"))
    forecasts = {key: value.reindex(index) for key, value in all_forecasts.items()}
    actual = prices.reindex(index)
    threshold_cutoff = pd.Timestamp(min(dates), tz="Europe/Berlin").tz_convert("UTC")
    spike_threshold = float(
        prices.loc[prices.index < threshold_cutoff]
        .dropna()
        .quantile(float(settings["experiment"]["spike_quantile"]))
    )
    forecast_metrics = {
        key: economic_forecast_metrics(
            actual, forecast, spike_threshold_eur_mwh=spike_threshold
        )
        for key, forecast in forecasts.items()
    }
    battery = BatteryConfig(**settings["battery"])
    dispatch = dispatch_kumo_forecast_families(
        forecasts,
        prices,
        battery,
        terminal_value_eur_per_mwh=float(settings["optimization"]["terminal_value_eur_per_mwh"]),
        solver=str(settings["optimization"]["solver"]),
    )
    decision = summarize_kumo_decision_value(
        dispatch,
        battery,
        reference_strategy="kumo_direct",
        bootstrap_samples=int(settings["experiment"]["bootstrap_samples"]),
        seed=int(settings["seed"]),
    )
    daily = _daily_frame(
        actual,
        forecasts,
        decision["daily_pnl"],
        labels,
        activated,
        spike_threshold,
    )
    block_sizes = [int(value) for value in settings["experiment"]["bootstrap_block_origins"]]
    comparisons = {}
    for candidate, reference in (
        ("conditional_regime_kumo", "kumo_direct"),
        ("conditional_regime_kumo", "regime_kumo"),
        ("conditional_regime_kumo", "linear"),
        ("conditional_regime_kumo", "quantile_gbm"),
        ("regime_kumo", "kumo_direct"),
    ):
        comparisons[f"{candidate}_vs_{reference}"] = _paired_inference(
            daily,
            candidate,
            reference,
            block_sizes,
            int(settings["experiment"]["bootstrap_samples"]),
            int(settings["seed"]),
        )
    useful_mae = daily.regime_kumo_mae_eur_mwh < daily.kumo_direct_mae_eur_mwh
    useful_revenue = daily.regime_kumo_revenue_eur > daily.kumo_direct_revenue_eur
    activation = daily.retrieval_activated
    activation_diagnostics = {
        "activation_frequency": float(activation.mean()),
        "mae_true_positive": int((activation & useful_mae).sum()),
        "mae_false_positive": int((activation & ~useful_mae).sum()),
        "mae_missed_opportunity": int((~activation & useful_mae).sum()),
        "revenue_true_positive": int((activation & useful_revenue).sum()),
        "revenue_false_positive": int((activation & ~useful_revenue).sum()),
        "revenue_missed_opportunity": int((~activation & useful_revenue).sum()),
    }
    result = {
        "name": name,
        "origins": len(dates),
        "hours": len(index),
        "forecast_availability_coverage": 1.0,
        "spike_threshold_eur_mwh": spike_threshold,
        "actual_negative_hours": int((actual < 0).sum()),
        "actual_spike_hours": int((actual >= spike_threshold).sum()),
        "forecast_metrics": forecast_metrics,
        "strategy_metrics": decision["strategy_metrics"],
        "paired_comparisons": comparisons,
        "activation_diagnostics": activation_diagnostics,
        "regime_summary": _regime_summary(daily).to_dict(orient="index"),
        "physical_verification": _physical_verification(dispatch, battery),
    }
    return result, {"dispatch": dispatch, "daily": daily, "pnl": decision["daily_pnl"]}


def run(config_path: str | Path) -> dict[str, Any]:
    """Run development and retrospective validation without native DARR or fine-tuning."""
    settings = load_yaml(config_path)
    experiment = settings["experiment"]
    eligibility = settings["eligibility"]
    prices = load_prices([Path(value) for value in experiment["inputs"]])
    development_dates, development_exclusions = eligible_origins(
        prices,
        experiment["development_start"],
        experiment["development_end"],
        kumo_context_hours=int(eligibility["kumo_context_hours"]),
        retrieval_context_hours=int(eligibility["retrieval_context_hours"]),
        maximum_forecast_steps=int(eligibility["maximum_forecast_steps"]),
    )
    validation_dates, validation_exclusions = eligible_origins(
        prices,
        experiment["retrospective_validation_start"],
        experiment["retrospective_validation_end"],
        kumo_context_hours=int(eligibility["kumo_context_hours"]),
        retrieval_context_hours=int(eligibility["retrieval_context_hours"]),
        maximum_forecast_steps=int(eligibility["maximum_forecast_steps"]),
    )
    if len(development_dates) != int(experiment["expected_development_origins"]):
        raise ValueError("Development eligibility count differs from pre-registration")
    if len(validation_dates) != int(experiment["expected_validation_origins"]):
        raise ValueError("Validation eligibility count differs from pre-registration")
    all_dates = development_dates + validation_dates

    thresholds = RegimeThresholds(**settings["regimes"])
    retrieval_started = time.perf_counter()
    retriever = OptimizedRegimeAwareRetriever(
        prices,
        thresholds,
        context_hours=int(settings["external_retrieval"]["context_hours"]),
        k=int(settings["external_retrieval"]["k"]),
        temperature=float(settings["external_retrieval"]["temperature"]),
        minimum_regime_neighbors=int(
            settings["external_retrieval"]["minimum_regime_neighbors"]
        ),
        cache_dir=experiment["retrieval_cache_dir"],
    )
    archive_load_seconds = time.perf_counter() - retrieval_started
    adapter = KumoForecastAdapter(
        KumoRuntimeConfig(**settings["kumo"]), cache_dir=experiment["cache_dir"]
    )
    policy = ConditionalRetrievalPolicy(tuple(settings["conditional_policy"]["activate_on"]))
    blocks: dict[str, list[pd.Series]] = {
        "kumo_direct": [],
        "regime_kumo": [],
        "conditional_regime_kumo": [],
    }
    labels: dict[str, str] = {}
    activated: dict[str, bool] = {}
    diagnostics = []
    runtime_rows = []
    for day in all_dates:
        direct = adapter.forecast_delivery_day(prices, day)
        retrieved = retriever.forecast(day, regime_conditioned=True)
        blended = blend_direct_and_retrieval(
            direct.forecast,
            retrieved.forecast,
            float(settings["external_retrieval"]["direct_weight"]),
        )
        snapshot = build_regime_snapshot(prices, day, thresholds)
        conditional, decision = policy.combine(direct.forecast, blended, snapshot)
        blocks["kumo_direct"].append(direct.forecast)
        blocks["regime_kumo"].append(blended)
        blocks["conditional_regime_kumo"].append(conditional)
        labels[str(day)] = "|".join(snapshot.labels)
        activated[str(day)] = decision.activate
        diagnostic = retrieved.diagnostics.copy()
        diagnostic["query_delivery_day"] = str(day)
        diagnostic["conditional_activated"] = decision.activate
        diagnostics.append(diagnostic)
        runtime_rows.append(
            {
                "delivery_day": str(day),
                "kumo_direct_seconds": direct.inference_seconds,
                "retrieval_seconds": retrieved.runtime_seconds,
                "conditional_activated": decision.activate,
                "kumo_cache_key": direct.cache_key,
                "retrieval_cache_key": retrieved.forecast_cache_key,
                "eligible_candidates": retrieved.eligible_candidates,
            }
        )
    forecasts = {key: pd.concat(value).sort_index() for key, value in blocks.items()}
    baselines, baseline_seconds = _selected_baselines(
        prices, all_dates, settings, int(settings["seed"])
    )
    forecasts.update(baselines)
    development, development_frames = _evaluate_split(
        "policy_development_2023",
        development_dates,
        forecasts,
        prices,
        labels,
        activated,
        settings,
    )
    validation, validation_frames = _evaluate_split(
        "retrospective_validation_2024",
        validation_dates,
        forecasts,
        prices,
        labels,
        activated,
        settings,
    )
    primary_block = str(settings["decision_rule"]["primary_block_origins"])
    vs_direct = validation["paired_comparisons"][
        "conditional_regime_kumo_vs_kumo_direct"
    ][primary_block]["revenue"]
    vs_always = validation["paired_comparisons"][
        "conditional_regime_kumo_vs_regime_kumo"
    ][primary_block]["revenue"]
    supported = bool(vs_direct["ci_lower_eur"] > 0 and vs_always["ci_lower_eur"] > 0)
    runtime = pd.DataFrame(runtime_rows)
    active_runtime = runtime.loc[runtime.conditional_activated, "retrieval_seconds"].sum()
    summary: dict[str, Any] = {
        "experiment": {
            "name": "conditional_regime_kumo_113",
            "config": str(config_path),
            "development_origins": len(development_dates),
            "validation_origins": len(validation_dates),
            "native_darr_evaluated": False,
            "fine_tuning_evaluated": False,
            "retrospective_validation_not_untouched": True,
            "thresholds": asdict(thresholds),
            "policy": asdict(policy),
            "battery": asdict(BatteryConfig(**settings["battery"])),
        },
        "eligibility": {
            "development_exclusions": dict(development_exclusions),
            "validation_exclusions": dict(validation_exclusions),
        },
        "development": development,
        "validation": validation,
        "decision": {
            "conditional_policy_supported": supported,
            "rule": "primary revenue CI positive versus direct and always-on retrieval",
            "recommendation": (
                "continue_conditional_retrieval" if supported else "retain_conventional_models"
            ),
        },
        "runtime": {
            "archive_load_seconds": archive_load_seconds,
            "archive_key": retriever.archive.archive_key,
            "kumo_direct_total_seconds": float(runtime.kumo_direct_seconds.sum()),
            "optimized_retrieval_all_origins_seconds": float(runtime.retrieval_seconds.sum()),
            "conditional_operational_retrieval_seconds": float(active_runtime),
            "baseline_seconds": baseline_seconds,
            "mean_retrieval_seconds": float(runtime.retrieval_seconds.mean()),
            "platform": platform.platform(),
            "pytorch_device": "cpu",
        },
    }
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecasts).to_parquet(output / "forecasts.parquet")
    pd.concat(diagnostics).to_parquet(output / "retrieval_diagnostics.parquet")
    runtime.to_csv(output / "runtime.csv", index=False)
    pd.DataFrame(
        {
            "delivery_day": [str(day) for day in all_dates],
            "split": ["development"] * len(development_dates)
            + ["validation"] * len(validation_dates),
            "issue_time_regimes": [labels[str(day)] for day in all_dates],
            "retrieval_activated": [activated[str(day)] for day in all_dates],
        }
    ).to_csv(output / "origins.csv", index=False)
    for split, frames in (
        ("development", development_frames),
        ("validation", validation_frames),
    ):
        frames["dispatch"].to_parquet(output / f"{split}_dispatch.parquet")
        frames["daily"].to_csv(output / f"{split}_daily.csv")
        frames["pnl"].to_csv(output / f"{split}_pnl.csv")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
