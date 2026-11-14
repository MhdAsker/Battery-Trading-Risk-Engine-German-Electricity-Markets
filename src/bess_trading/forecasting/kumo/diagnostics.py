"""Forecast and dispatch diagnostics focused on economically important price ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bess_trading.data.time import delivery_day_groups


def _classification(actual: pd.Series, predicted: pd.Series, threshold: float) -> dict[str, float]:
    observed = actual >= threshold
    forecast = predicted >= threshold
    true_positive = int((observed & forecast).sum())
    false_positive = int((~observed & forecast).sum())
    false_negative = int((observed & ~forecast).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def economic_forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    spike_threshold_eur_mwh: float,
    top_hours: int = 4,
    arbitrage_threshold_eur_mwh: float = 12.0,
) -> dict[str, float]:
    """Evaluate level, event, rank, spread and transaction-window forecast quality."""
    frame = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if frame.empty or frame.index.tz is None:
        raise ValueError("Economic metrics require aligned timezone-aware forecasts")
    negative = _classification(-frame.actual, -frame.predicted, 0.0)
    spike = _classification(frame.actual, frame.predicted, spike_threshold_eur_mwh)
    rows = []
    for _, index in delivery_day_groups(frame.index).items():
        block = frame.loc[index]
        if len(block) < top_hours * 2:
            continue
        actual_order = block.actual.rank(method="average")
        predicted_order = block.predicted.rank(method="average")
        correlation = spearmanr(actual_order, predicted_order).statistic
        actual_low = set(block.actual.nsmallest(top_hours).index)
        predicted_low = set(block.predicted.nsmallest(top_hours).index)
        actual_high = set(block.actual.nlargest(top_hours).index)
        predicted_high = set(block.predicted.nlargest(top_hours).index)
        actual_spread = float(block.actual.max() - block.actual.min())
        predicted_spread = float(block.predicted.max() - block.predicted.min())
        rows.append(
            {
                "rank_correlation": float(correlation),
                "charge_hour_overlap": len(actual_low & predicted_low) / top_hours,
                "discharge_hour_overlap": len(actual_high & predicted_high) / top_hours,
                "minimum_timing_error_hours": abs(
                    block.index.get_loc(block.actual.idxmin())
                    - block.index.get_loc(block.predicted.idxmin())
                ),
                "maximum_timing_error_hours": abs(
                    block.index.get_loc(block.actual.idxmax())
                    - block.index.get_loc(block.predicted.idxmax())
                ),
                "spread_absolute_error": abs(predicted_spread - actual_spread),
                "false_arbitrage": predicted_spread >= arbitrage_threshold_eur_mwh
                and actual_spread < arbitrage_threshold_eur_mwh,
                "missed_arbitrage": predicted_spread < arbitrage_threshold_eur_mwh
                and actual_spread >= arbitrage_threshold_eur_mwh,
            }
        )
    daily = pd.DataFrame(rows)
    error = frame.predicted - frame.actual
    return {
        "mae_eur_mwh": float(error.abs().mean()),
        "rmse_eur_mwh": float(np.sqrt((error**2).mean())),
        "negative_price_precision": negative["precision"],
        "negative_price_recall": negative["recall"],
        "negative_price_f1": negative["f1"],
        "spike_precision": spike["precision"],
        "spike_recall": spike["recall"],
        "spike_f1": spike["f1"],
        **{f"mean_{column}": float(daily[column].mean()) for column in daily},
    }


def dispatch_influence(
    direct_dispatch: pd.DataFrame, retrieval_dispatch: pd.DataFrame
) -> dict[str, float]:
    """Quantify how retrieval changes feasible charge/discharge actions."""
    if not direct_dispatch.index.equals(retrieval_dispatch.index):
        raise ValueError("Dispatch indices differ")
    charge_delta = (retrieval_dispatch.charge_mw - direct_dispatch.charge_mw).abs()
    discharge_delta = (retrieval_dispatch.discharge_mw - direct_dispatch.discharge_mw).abs()
    return {
        "changed_charge_intervals": int((charge_delta > 1e-7).sum()),
        "changed_discharge_intervals": int((discharge_delta > 1e-7).sum()),
        "absolute_charge_change_mwh": float(
            (charge_delta * direct_dispatch.interval_hours).sum()
        ),
        "absolute_discharge_change_mwh": float(
            (discharge_delta * direct_dispatch.interval_hours).sum()
        ),
    }
