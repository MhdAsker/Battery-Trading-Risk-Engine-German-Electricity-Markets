"""Metrics used by the experiment layer and dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    return {"mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean()))}


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    error = actual - predicted
    return float(np.maximum(quantile * error, (quantile - 1) * error).mean())


def interval_metrics(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> dict[str, float]:
    return {
        "coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "mean_width": float((upper - lower).mean()),
    }


def trading_metrics(
    result: pd.DataFrame, config_energy_mwh: float, alpha: float = 0.05
) -> dict[str, float]:
    revenue = result.net_revenue_eur
    cumulative = revenue.cumsum()
    drawdown = cumulative - cumulative.cummax()
    losses = -revenue
    var = losses.quantile(1 - alpha)
    tail = losses[losses >= var]
    throughput = float(result.throughput_mwh.sum())
    return {
        "gross_revenue_eur": float(result.gross_revenue_eur.sum()),
        "net_revenue_eur": float(revenue.sum()),
        "degradation_cost_eur": float(result.degradation_cost_eur.sum()),
        "throughput_mwh": throughput,
        "equivalent_cycles": throughput / (2 * config_energy_mwh),
        "max_drawdown_eur": float(drawdown.min()),
        "revenue_volatility_eur": float(revenue.std()),
        "cvar_loss_eur": float(tail.mean()) if len(tail) else 0.0,
    }
