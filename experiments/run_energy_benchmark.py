"""Run a causal, reproducible energy-only benchmark on local ENTSO-E parquet files.

This script deliberately keeps data acquisition separate. It refits every seven days
on a trailing 365-day window, optimizes each complete UTC day, and settles schedules
against realized prices. Results are written to a versioned public results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bess_trading.backtesting.engine import settle_dispatch
from bess_trading.config import BatteryConfig
from bess_trading.features.build import build_features
from bess_trading.forecasting.models import (
    HourOfWeekForecaster,
    PersistenceForecaster,
    QuantileGBM,
)
from bess_trading.metrics.core import interval_metrics, pinball_loss, point_metrics, trading_metrics
from bess_trading.optimization.dispatch import optimize_dispatch, optimize_quantile_robust


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prices(paths: list[Path]) -> pd.Series:
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.index.tz is None:
        raise ValueError("Input timestamps must be timezone-aware")
    return frame.price_eur_mwh.astype(float)


def causal_feature_price(prices: pd.Series) -> pd.Series:
    """Fill sparse feature-history gaps without filling realized settlement prices.

    Values use the same UTC hour one week earlier, then one day earlier. Remaining
    early-history gaps use only forward-propagated observations. No future value is
    used and the original target series remains unchanged.
    """
    complete_index = pd.date_range(prices.index.min(), prices.index.max(), freq="h", tz="UTC")
    feature_price = prices.reindex(complete_index)
    for lag in (168, 24):
        feature_price = feature_price.fillna(feature_price.shift(lag))
    return feature_price.ffill()


def forecast_walk_forward(
    prices: pd.Series, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> pd.DataFrame:
    feature_price = causal_feature_price(prices)
    features = build_features(feature_price.to_frame(name="price_eur_mwh")).dropna()
    predictions: list[pd.DataFrame] = []
    cursor = test_start
    while cursor < test_end:
        block_end = min(cursor + pd.Timedelta(days=7), test_end)
        train_start = cursor - pd.Timedelta(days=365)
        train_idx = features.index[
            (features.index >= train_start)
            & (features.index < cursor)
            & features.index.isin(prices.dropna().index)
        ]
        test_idx = features.index[(features.index >= cursor) & (features.index < block_end)]
        if len(train_idx) < 24 * 300 or test_idx.empty:
            raise ValueError("Insufficient complete history for walk-forward block")
        model = QuantileGBM().fit(features.loc[train_idx], prices.loc[train_idx])
        block = model.predict(features.loc[test_idx])
        # Each daily forecast may use the previous day's realized price even though
        # the ML model itself is refitted only weekly.
        block["persistence"] = PersistenceForecaster().predict(feature_price, test_idx)
        history = feature_price[feature_price.index < cursor]
        block["hour_of_week"] = HourOfWeekForecaster().predict(history, test_idx)
        block["forecast_issue_time_utc"] = cursor
        predictions.append(block)
        cursor = block_end
    return pd.concat(predictions).sort_index()


def dispatch_strategies(
    forecasts: pd.DataFrame, actual: pd.Series, config: BatteryConfig
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    joined = forecasts.join(actual.rename("actual"), how="inner").dropna()
    for day, block in joined.groupby(joined.index.floor("D")):
        if len(block) != 24:
            continue
        strategies = {
            "perfect_foresight": optimize_dispatch(block.actual, config),
            "persistence": optimize_dispatch(block.persistence, config),
            "hour_of_week": optimize_dispatch(block.hour_of_week, config),
            "ml_p50": optimize_dispatch(block.p50, config),
            "quantile_robust": optimize_quantile_robust(block[["p10", "p50", "p90"]], config),
        }
        for name, dispatch in strategies.items():
            settled = settle_dispatch(dispatch, actual, config)
            settled["strategy"] = name
            settled["delivery_day_utc"] = day
            if name in {"ml_p50", "quantile_robust"}:
                settled[["p10", "p50", "p90"]] = block[["p10", "p50", "p90"]]
            outputs.append(settled)
    if not outputs:
        raise ValueError("No complete 24-hour test days were available")
    return pd.concat(outputs).sort_index()


def summarize(
    forecasts: pd.DataFrame, actual: pd.Series, dispatch: pd.DataFrame, config: BatteryConfig
) -> dict[str, object]:
    aligned = forecasts.join(actual.rename("actual"), how="inner").dropna()
    forecast_metrics: dict[str, object] = {}
    for name in ("persistence", "hour_of_week", "p50"):
        forecast_metrics[name] = point_metrics(aligned.actual, aligned[name])
    forecast_metrics["quantiles"] = {
        "p10_pinball": pinball_loss(aligned.actual, aligned.p10, 0.1),
        "p50_pinball": pinball_loss(aligned.actual, aligned.p50, 0.5),
        "p90_pinball": pinball_loss(aligned.actual, aligned.p90, 0.9),
        **interval_metrics(aligned.actual, aligned.p10, aligned.p90),
    }
    strategy_metrics = {
        name: trading_metrics(group, config.energy_mwh)
        for name, group in dispatch.groupby("strategy")
    }
    upper = strategy_metrics["perfect_foresight"]["net_revenue_eur"]
    for metrics in strategy_metrics.values():
        metrics["regret_vs_perfect_eur"] = upper - metrics["net_revenue_eur"]
        metrics["revenue_per_mw_eur"] = metrics["net_revenue_eur"] / config.power_mw
        metrics["revenue_per_mwh_capacity_eur"] = metrics["net_revenue_eur"] / config.energy_mwh

    weekly = aligned.assign(week=aligned.index.to_period("W").astype(str))
    ml_dispatch = dispatch[dispatch.strategy == "ml_p50"].copy()
    ml_dispatch["week"] = ml_dispatch.index.to_period("W").astype(str)
    weekly_mae = weekly.groupby("week").apply(
        lambda x: float((x.actual - x.p50).abs().mean()), include_groups=False
    )
    weekly_revenue = ml_dispatch.groupby("week").net_revenue_eur.sum()
    common = weekly_mae.index.intersection(weekly_revenue.index)
    correlation = spearmanr(weekly_mae.loc[common], weekly_revenue.loc[common])
    return {
        "forecast_metrics": forecast_metrics,
        "strategy_metrics": strategy_metrics,
        "mae_revenue_spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "n_weeks": len(common),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--test-start", default="2024-11-01T00:00:00Z")
    parser.add_argument("--test-end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--output", type=Path, default=Path("results/energy_benchmark_2024"))
    args = parser.parse_args()

    np.random.seed(42)
    prices = load_prices(args.inputs)
    test_start, test_end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
    config = BatteryConfig()
    forecasts = forecast_walk_forward(prices, test_start, test_end)
    dispatch = dispatch_strategies(forecasts, prices, config)
    summary = summarize(forecasts, prices, dispatch, config)
    summary["experiment"] = {
        "test_start": str(test_start),
        "test_end_exclusive": str(test_end),
        "complete_dispatch_days": int(dispatch.delivery_day_utc.nunique()),
        "battery": asdict(config),
        "inputs": {str(path): _sha256(path) for path in args.inputs},
        "seed": 42,
        "note": "Historical simulation; energy-only; not live or commercial trading.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(args.output / "forecasts.parquet")
    dispatch.to_parquet(args.output / "dispatch.parquet")
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["strategy_metrics"]).T.to_csv(args.output / "strategy_metrics.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
