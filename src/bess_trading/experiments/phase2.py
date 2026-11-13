"""Leakage-safe probabilistic value experiment for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.config import BatteryConfig, load_yaml
from bess_trading.data.time import delivery_day_index
from bess_trading.experiments.phase1 import forecast_daily, load_prices
from bess_trading.forecasting.probabilistic import (
    ConformalQuantileCalibrator,
    ensemble_crps,
    pit_values,
    residual_block_scenarios,
    scenario_diagnostics,
    weighted_interval_score,
)
from bess_trading.metrics.research import (
    attribute_dispatch_errors,
    classify_regimes,
    conditional_interval_coverage,
)
from bess_trading.metrics.risk import daily_pnl, paired_block_bootstrap_ci
from bess_trading.optimization.dispatch import optimize_dispatch, optimize_stochastic_cvar
from bess_trading.optimization.frontier import revenue_risk_frontier, run_sensitivity_study


def calibrate_without_test_leakage(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    test_start: pd.Timestamp,
    alpha: float,
) -> tuple[pd.DataFrame, ConformalQuantileCalibrator]:
    """Fit conformal correction before the immutable test boundary."""
    calibration_index = forecasts.index[forecasts.index < test_start]
    test_index = forecasts.index[forecasts.index >= test_start]
    if calibration_index.empty or test_index.empty:
        raise ValueError("Both pre-test calibration and test forecasts are required")
    calibration = forecasts.loc[calibration_index, ["p10", "p50", "p90"]].join(
        actual.rename("actual"), how="inner"
    ).dropna()
    calibrator = ConformalQuantileCalibrator(alpha=alpha).fit(
        calibration.actual,
        calibration.p10,
        calibration.p90,
    )
    test = forecasts.loc[test_index, ["p10", "p50", "p90"]].join(
        actual.rename("actual"), how="inner"
    ).dropna()
    return calibrator.transform(test[["p10", "p50", "p90"]]), calibrator


def _settle_strategy(
    schedule: pd.DataFrame,
    actual: pd.Series,
    config: BatteryConfig,
    strategy: str,
    day: object,
) -> pd.DataFrame:
    result = settle_dispatch(schedule, actual, config)
    result["strategy"] = strategy
    result["delivery_day_local"] = str(day)
    return result


def dispatch_probabilistic_strategies(
    calibrated: pd.DataFrame,
    actual: pd.Series,
    residual_history: pd.Series,
    config: BatteryConfig,
    *,
    scenario_count: int,
    block_hours: int,
    risk_aversion: float,
    alpha: float,
    solver: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare P50 and stochastic CVaR schedules with continuous strategy SOC."""
    states = {"p50": config.initial_soc_mwh, "stochastic_cvar": config.initial_soc_mwh}
    dispatches, all_scenarios = [], []
    local_days = pd.Index(calibrated.index.tz_convert("Europe/Berlin").date).unique()
    for offset, day in enumerate(local_days):
        horizon = delivery_day_index(day, "1h")
        if not horizon.isin(calibrated.index).all() or actual.reindex(horizon).isna().any():
            continue
        block = calibrated.loc[horizon]
        scenarios = residual_block_scenarios(
            block.p50,
            residual_history,
            scenarios=scenario_count,
            block_hours=block_hours,
            seed=seed + offset,
        )
        all_scenarios.append(scenarios)
        schedules = {
            "p50": optimize_dispatch(
                block.p50, config, initial_soc_mwh=states["p50"], solver=solver
            ),
            "stochastic_cvar": optimize_stochastic_cvar(
                scenarios,
                config,
                initial_soc_mwh=states["stochastic_cvar"],
                risk_aversion=risk_aversion,
                alpha=alpha,
                solver=solver,
            ),
        }
        for name, schedule in schedules.items():
            states[name] = float(schedule.soc_mwh.iloc[-1])
            dispatches.append(_settle_strategy(schedule, actual, config, name, day))
    if not dispatches:
        raise ValueError("No complete test delivery days")
    return pd.concat(dispatches).sort_index(), pd.concat(all_scenarios).sort_index()


def _regime_analysis(
    forecasts: pd.DataFrame, actual: pd.Series, fundamentals: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    frame = forecasts.join(actual.rename("actual"), how="inner")
    frame["volatility"] = frame.actual.rolling(24, min_periods=1).std().fillna(0)
    if fundamentals is not None and "renewable_share_forecast" in fundamentals:
        frame = frame.join(fundamentals[["renewable_share_forecast"]], how="left")
    thresholds = {
        "spike": float(frame.actual.quantile(0.95)),
        "volatility": float(frame.volatility.quantile(0.75)),
        "renewable_share": 0.7,
    }
    regimes = classify_regimes(frame, thresholds)
    coverage = conditional_interval_coverage(frame.actual, frame.p10, frame.p90, regimes)
    return regimes, coverage


def run(config_path: str | Path) -> dict[str, Any]:
    """Run Phase 2 and persist forecasts, scenarios, dispatch, diagnostics, and inference."""
    settings = load_yaml(config_path)
    seed = int(settings.get("seed", 42))
    np.random.seed(seed)
    experiment = settings["experiment"]
    prices = load_prices([Path(value) for value in experiment["inputs"]])
    calibration_start = pd.Timestamp(experiment["calibration_start"])
    test_start, test_end = pd.Timestamp(experiment["test_start"]), pd.Timestamp(
        experiment["test_end"]
    )
    forecasting = settings["forecasting"]
    forecasts = forecast_daily(
        prices,
        test_start=calibration_start,
        test_end=test_end,
        train_days=int(forecasting["train_days"]),
        retrain_days=int(forecasting["retrain_days"]),
        issue_hour_local=int(forecasting["issue_hour_local"]),
        seed=seed,
        model_max_iter=int(forecasting.get("model_max_iter", 100)),
    )
    alpha = float(settings["probabilistic"]["interval_alpha"])
    calibrated, calibrator = calibrate_without_test_leakage(forecasts, prices, test_start, alpha)
    calibration_index = forecasts.index[
        (forecasts.index < test_start) & (forecasts.index >= calibration_start)
    ]
    residuals = prices.reindex(calibration_index) - forecasts.loc[calibration_index, "p50"]
    residuals = residuals.dropna()
    battery = BatteryConfig(**settings["battery"])
    optimization = settings["optimization"]
    dispatch, scenarios = dispatch_probabilistic_strategies(
        calibrated,
        prices,
        residuals,
        battery,
        scenario_count=int(settings["probabilistic"]["scenario_count"]),
        block_hours=int(settings["probabilistic"]["block_hours"]),
        risk_aversion=float(optimization["risk_aversion"]),
        alpha=float(optimization["cvar_alpha"]),
        solver=str(optimization["solver"]),
        seed=seed,
    )
    actual = prices.reindex(calibrated.index)
    scenario_actual = actual.reindex(scenarios.index)
    regimes, conditional_coverage = _regime_analysis(calibrated, actual, None)
    pnl = daily_pnl(dispatch)
    comparison = paired_block_bootstrap_ci(
        pnl.stochastic_cvar,
        pnl.p50,
        samples=int(experiment.get("bootstrap_samples", 2000)),
        seed=seed,
    )
    first_day = scenarios.index[0].tz_convert("Europe/Berlin").date()
    representative = scenarios.loc[delivery_day_index(first_day, "1h")]
    frontier, _ = revenue_risk_frontier(
        representative,
        battery,
        optimization["frontier_risk_aversions"],
        alpha=float(optimization["cvar_alpha"]),
        solver=str(optimization["solver"]),
    )
    sensitivity = run_sensitivity_study(
        representative,
        battery,
        risk_aversion=float(optimization["risk_aversion"]),
        alpha=float(optimization["cvar_alpha"]),
        solver=str(optimization["solver"]),
    )
    cvar_schedule = dispatch[dispatch.strategy == "stochastic_cvar"]
    attribution = attribute_dispatch_errors(
        actual,
        calibrated.p50,
        cvar_schedule.charge_mw.reindex(actual.index),
        cvar_schedule.discharge_mw.reindex(actual.index),
    )
    metrics: dict[str, Any] = {
        "experiment": {
            "phase": 2,
            "calibration_start": str(calibration_start),
            "test_start": str(test_start),
            "test_end_exclusive": str(test_end),
            "test_hours": len(calibrated),
            "test_days": int(dispatch.delivery_day_local.nunique()),
            "conformal_correction_eur_mwh": calibrator.correction_,
            "test_isolation_passed": bool(calibration_index.max() < test_start),
        },
        "probabilistic": {
            "p10_p90_coverage": float(
                ((actual >= calibrated.p10) & (actual <= calibrated.p90)).mean()
            ),
            "wis": weighted_interval_score(actual, calibrated),
            "crps": ensemble_crps(scenario_actual, scenarios),
            "pit_mean": float(pit_values(actual, calibrated).mean()),
            "scenario_diagnostics": scenario_diagnostics(scenario_actual, scenarios),
            "conditional_coverage": conditional_coverage,
        },
        "revenue_comparison": comparison,
        "research_answers": {
            "probabilistic_value": (
                "positive" if comparison["ci_lower_eur"] > 0 else "not statistically established"
            ),
            "negative_results_reported": comparison["ci_lower_eur"] <= 0,
        },
    }
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(output / "calibrated_forecasts.parquet")
    scenarios.to_parquet(output / "scenarios.parquet")
    dispatch.to_parquet(output / "dispatch.parquet")
    regimes.to_parquet(output / "regimes.parquet")
    attribution.to_parquet(output / "error_attribution.parquet")
    frontier.to_csv(output / "efficient_frontier.csv", index=False)
    sensitivity.to_csv(output / "sensitivity_study.csv", index=False)
    pnl.to_csv(output / "daily_pnl.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
