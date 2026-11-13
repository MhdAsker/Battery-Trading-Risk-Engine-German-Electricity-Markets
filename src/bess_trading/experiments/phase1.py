"""Causal and physically strict Phase 1 day-ahead benchmark."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.config import BatteryConfig, load_yaml
from bess_trading.data.time import (
    delivery_day_index,
    forecast_issue_time,
    lagged_price_feature_availability_times,
    price_publication_times,
)
from bess_trading.features.build import build_features
from bess_trading.forecasting.models import (
    HourOfWeekForecaster,
    LinearForecaster,
    PersistenceForecaster,
    QuantileGBM,
)
from bess_trading.metrics.core import interval_metrics, pinball_loss, point_metrics, trading_metrics
from bess_trading.metrics.risk import daily_pnl, daily_risk_metrics, paired_block_bootstrap_ci
from bess_trading.optimization.dispatch import (
    idle_dispatch,
    optimize_dispatch,
    optimize_quantile_robust,
)
from bess_trading.optimization.rules import rule_based_dispatch

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prices(paths: list[Path]) -> pd.Series:
    """Load and de-duplicate UTC price parquet files without filling settlement data."""
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.index.tz is None:
        raise ValueError("Input timestamps must be timezone-aware")
    return frame.price_eur_mwh.astype(float).tz_convert("UTC")


def causal_feature_price(prices: pd.Series, maximum_gap_hours: int = 6) -> pd.Series:
    """Fill feature-only gaps from past observations with a strict maximum gap."""
    complete = pd.date_range(prices.index.min(), prices.index.max(), freq="h", tz="UTC")
    result = prices.reindex(complete)
    for lag in (168, 24):
        result = result.fillna(result.shift(lag))
    result = result.ffill(limit=maximum_gap_hours)
    return result


def missingness_report(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """Describe source gaps and excluded local delivery days."""
    expected = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    aligned = prices.reindex(expected)
    missing = aligned.isna()
    local_days = expected.tz_convert("Europe/Berlin").date
    day_counts = pd.Series(missing.to_numpy(), index=local_days).groupby(level=0).sum()
    excluded = [str(day) for day, count in day_counts.items() if count > 0]
    observed = pd.DataFrame({"price": aligned.to_numpy(), "day": local_days}).groupby("day").price
    daily_observed = observed.agg(["count", "mean", "std", "min", "max"])
    daily_observed["range"] = daily_observed["max"] - daily_observed["min"]
    daily_observed["excluded"] = daily_observed.index.astype(str).isin(excluded)

    def group_summary(excluded_flag: bool) -> dict[str, float]:
        sample = daily_observed[daily_observed.excluded == excluded_flag]
        return {
            "days": int(len(sample)),
            "mean_observed_price_eur_mwh": float(sample["mean"].mean())
            if len(sample)
            else float("nan"),
            "mean_observed_daily_range_eur_mwh": float(sample["range"].mean())
            if len(sample)
            else float("nan"),
        }

    runs = missing.groupby((missing != missing.shift()).cumsum()).agg(["first", "size"])
    missing_runs = runs[runs["first"]]
    return {
        "expected_intervals": len(expected),
        "missing_intervals": int(missing.sum()),
        "missing_rate": float(missing.mean()),
        "longest_missing_run_hours": int(missing_runs["size"].max()) if len(missing_runs) else 0,
        "excluded_delivery_days": excluded,
        "excluded_day_count": len(excluded),
        "observed_characteristics_complete_days": group_summary(False),
        "observed_characteristics_excluded_days": group_summary(True),
        "warning": "Excluded days may be non-random; no settlement value is imputed.",
    }


def forecast_daily(
    prices: pd.Series,
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_days: int,
    retrain_days: int,
    issue_hour_local: int,
    seed: int,
    model_max_iter: int = 100,
) -> pd.DataFrame:
    """Issue daily forecasts while refitting models on a weekly-style cadence."""
    feature_price = causal_feature_price(prices)
    features = build_features(feature_price.to_frame(name="price_eur_mwh")).dropna()
    quantile_model: QuantileGBM | None = None
    linear_model: LinearForecaster | None = None
    last_refit_day: date | None = None
    outputs: list[pd.DataFrame] = []
    start_day = test_start.tz_convert("Europe/Berlin").date()
    end_day = test_end.tz_convert("Europe/Berlin").date()
    days = pd.date_range(start_day, end_day, freq="D", inclusive="left")

    target_publication = pd.Series(price_publication_times(prices.index), index=prices.index)
    for day_timestamp in days:
        day = day_timestamp.date()
        issue = forecast_issue_time(day, issue_hour_local)
        horizon = delivery_day_index(day, "1h")
        if not horizon.isin(features.index).all():
            LOGGER.warning("Skipping %s because its feature snapshot is incomplete", day)
            continue
        refit_due = (
            quantile_model is None
            or last_refit_day is None
            or (day - last_refit_day).days >= retrain_days
        )
        eligible = prices.index[
            (prices.index >= issue - pd.Timedelta(days=train_days))
            & (target_publication <= issue).to_numpy()
            & prices.notna().to_numpy()
            & prices.index.isin(features.index)
        ]
        if len(eligible) < 24 * min(300, train_days - 7):
            raise ValueError(
                f"Insufficient causal training history for {day}: {len(eligible)} rows"
            )
        if refit_due:
            quantile_model = QuantileGBM(random_state=seed, max_iter=model_max_iter).fit(
                features.loc[eligible], prices.loc[eligible]
            )
            linear_model = LinearForecaster().fit(features.loc[eligible], prices.loc[eligible])
            last_refit_day = day

        assert quantile_model is not None and linear_model is not None
        snapshot = features.loc[horizon]
        block = quantile_model.predict(snapshot)
        block["linear"] = linear_model.predict(snapshot)
        block["persistence"] = PersistenceForecaster().predict(feature_price, horizon)
        known_history = feature_price[price_publication_times(feature_price.index) <= issue]
        block["hour_of_week"] = HourOfWeekForecaster().predict(known_history, horizon)
        available_at = lagged_price_feature_availability_times(horizon)
        block["forecast_issue_time_utc"] = issue
        block["feature_available_time_utc"] = available_at
        block["model_refit_date"] = str(last_refit_day)
        if (block.feature_available_time_utc > block.forecast_issue_time_utc).any():
            raise AssertionError(f"Feature availability violation on {day}")
        outputs.append(block)
    if not outputs:
        raise ValueError("No forecasts were generated")
    return pd.concat(outputs).sort_index()


def dispatch_strategies(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    config: BatteryConfig,
    *,
    risk_aversion: float,
    terminal_value_eur_per_mwh: float,
    solver: str,
) -> pd.DataFrame:
    """Backtest strategies on complete German days with continuous strategy SOC."""
    strategies = [
        "zero_trade",
        "rule_based",
        "perfect_foresight",
        "persistence",
        "hour_of_week",
        "linear",
        "ml_p50",
        "quantile_robust",
    ]
    states = {name: config.initial_soc_mwh for name in strategies}
    outputs: list[pd.DataFrame] = []
    days = forecasts.index.tz_convert("Europe/Berlin").date
    for day in pd.Index(days).unique():
        horizon = delivery_day_index(day, "1h")
        if not horizon.isin(forecasts.index).all():
            continue
        realized = actual.reindex(horizon)
        if realized.isna().any():
            continue
        block = forecasts.loc[horizon]
        schedules = {
            "zero_trade": idle_dispatch(horizon, config, states["zero_trade"]),
            "rule_based": rule_based_dispatch(block.p50, config, states["rule_based"]),
            "perfect_foresight": optimize_dispatch(
                realized,
                config,
                solver=solver,
                initial_soc_mwh=states["perfect_foresight"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
            "persistence": optimize_dispatch(
                block.persistence,
                config,
                solver=solver,
                initial_soc_mwh=states["persistence"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
            "hour_of_week": optimize_dispatch(
                block.hour_of_week,
                config,
                solver=solver,
                initial_soc_mwh=states["hour_of_week"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
            "linear": optimize_dispatch(
                block.linear,
                config,
                solver=solver,
                initial_soc_mwh=states["linear"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
            "ml_p50": optimize_dispatch(
                block.p50,
                config,
                solver=solver,
                initial_soc_mwh=states["ml_p50"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
            "quantile_robust": optimize_quantile_robust(
                block[["p10", "p50", "p90"]],
                config,
                risk_aversion=risk_aversion,
                solver=solver,
                initial_soc_mwh=states["quantile_robust"],
                terminal_value_eur_per_mwh=terminal_value_eur_per_mwh,
            ),
        }
        for name, schedule in schedules.items():
            states[name] = float(schedule.soc_mwh.iloc[-1])
            settled = settle_dispatch(schedule, actual, config)
            settled["strategy"] = name
            settled["delivery_day_local"] = str(day)
            settled["forecast_issue_time_utc"] = block.forecast_issue_time_utc
            if name in {"ml_p50", "quantile_robust"}:
                settled[["p10", "p50", "p90"]] = block[["p10", "p50", "p90"]]
            outputs.append(settled)
    if not outputs:
        raise ValueError("No complete delivery days were available for dispatch")
    return pd.concat(outputs).sort_index()


def summarize(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    dispatch: pd.DataFrame,
    config: BatteryConfig,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    aligned = forecasts.join(actual.rename("actual"), how="inner").dropna()
    forecast_metrics = {
        name: point_metrics(aligned.actual, aligned[name])
        for name in ("persistence", "hour_of_week", "linear", "p50")
    }
    forecast_metrics["quantiles"] = {
        "p10_pinball": pinball_loss(aligned.actual, aligned.p10, 0.1),
        "p50_pinball": pinball_loss(aligned.actual, aligned.p50, 0.5),
        "p90_pinball": pinball_loss(aligned.actual, aligned.p90, 0.9),
        **interval_metrics(aligned.actual, aligned.p10, aligned.p90),
    }
    pnl = daily_pnl(dispatch)
    strategy_metrics: dict[str, Any] = {}
    for name, group in dispatch.groupby("strategy"):
        strategy_metrics[name] = {
            **trading_metrics(group, config.energy_mwh),
            **daily_risk_metrics(pnl[name]),
        }
    perfect = strategy_metrics["perfect_foresight"]["net_revenue_eur"]
    for name, metrics in strategy_metrics.items():
        metrics["regret_vs_perfect_eur"] = perfect - metrics["net_revenue_eur"]
        metrics["revenue_per_mw_day_eur"] = metrics["net_revenue_eur"] / config.power_mw / len(pnl)
        metrics["paired_ci_vs_ml_p50"] = paired_block_bootstrap_ci(
            pnl[name],
            pnl["ml_p50"],
            block_days=min(7, max(1, len(pnl) // 2)),
            samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "forecast_metrics": forecast_metrics,
        "strategy_metrics": strategy_metrics,
        "daily_pnl_summary": pnl.describe().to_dict(),
    }


def run(config_path: str | Path) -> dict[str, Any]:
    """Run Phase 1 from YAML and persist an auditable artifact bundle."""
    config_path = Path(config_path)
    settings = load_yaml(config_path)
    seed = int(settings.get("seed", 42))
    np.random.seed(seed)
    input_paths = [Path(value) for value in settings["experiment"]["inputs"]]
    prices = load_prices(input_paths)
    test_start = pd.Timestamp(settings["experiment"]["test_start"])
    test_end = pd.Timestamp(settings["experiment"]["test_end"])
    battery = BatteryConfig(**settings["battery"])
    forecast_settings = settings["forecasting"]
    optimization = settings["optimization"]
    output = Path(settings["experiment"]["output"])

    forecasts = forecast_daily(
        prices,
        test_start=test_start,
        test_end=test_end,
        train_days=int(forecast_settings["train_days"]),
        retrain_days=int(forecast_settings["retrain_days"]),
        issue_hour_local=int(forecast_settings["issue_hour_local"]),
        seed=seed,
        model_max_iter=int(forecast_settings.get("model_max_iter", 100)),
    )
    dispatch = dispatch_strategies(
        forecasts,
        prices,
        battery,
        risk_aversion=float(optimization["risk_aversion"]),
        terminal_value_eur_per_mwh=float(optimization["terminal_value_eur_per_mwh"]),
        solver=str(optimization["solver"]),
    )
    complete_days = int(dispatch.delivery_day_local.nunique())
    minimum_days = int(settings["experiment"].get("minimum_oos_days", 300))
    if complete_days < minimum_days:
        raise ValueError(f"Only {complete_days} complete OOS days; required {minimum_days}")
    summary = summarize(
        forecasts,
        prices,
        dispatch,
        battery,
        int(settings["experiment"].get("bootstrap_samples", 2000)),
        seed,
    )
    summary["experiment"] = {
        "phase": 1,
        "test_start": str(test_start),
        "test_end_exclusive": str(test_end),
        "complete_dispatch_days": complete_days,
        "battery": asdict(battery),
        "config": settings,
        "config_path": str(config_path),
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "seed": seed,
        "causality_audit_passed": bool(
            (forecasts.feature_available_time_utc <= forecasts.forecast_issue_time_utc).all()
        ),
        "missingness": missingness_report(prices, test_start, test_end),
        "note": "Historical energy-only simulation; not live or commercial trading.",
    }
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output / "forecasts.parquet")
    dispatch.to_parquet(output / "dispatch.parquet")
    daily_pnl(dispatch).to_csv(output / "daily_pnl.csv")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "effective_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return summary
