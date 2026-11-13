"""Regime diagnostics and economically weighted forecast-error attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regimes(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Create overlapping market-regime flags using training-fitted thresholds."""
    result = pd.DataFrame(index=frame.index)
    result["negative_price"] = frame.actual < 0
    result["price_spike"] = frame.actual >= thresholds["spike"]
    result["high_volatility"] = frame.volatility >= thresholds["volatility"]
    if "renewable_share_forecast" in frame:
        result["renewable_surplus"] = (
            frame.renewable_share_forecast >= thresholds["renewable_share"]
        )
    return result


def conditional_interval_coverage(
    actual: pd.Series, lower: pd.Series, upper: pd.Series, regimes: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Report interval coverage, width and count by possibly overlapping regime."""
    output: dict[str, dict[str, float]] = {}
    for name in regimes:
        mask = regimes[name].fillna(False)
        output[name] = {
            "count": int(mask.sum()),
            "coverage": float(
                ((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])).mean()
            ),
            "mean_width": float((upper[mask] - lower[mask]).mean()),
        }
    return output


def attribute_dispatch_errors(
    actual: pd.Series, forecast: pd.Series, charge_mw: pd.Series, discharge_mw: pd.Series
) -> pd.DataFrame:
    """Assign economically relevant error categories at scheduled decision intervals."""
    error = forecast - actual
    result = pd.DataFrame(index=actual.index)
    result["forecast_error_eur_mwh"] = error
    result["dispatch_weighted_error_eur"] = error * (discharge_mw - charge_mw)
    actual_high = actual >= actual.quantile(0.9)
    forecast_high = forecast >= forecast.quantile(0.9)
    actual_low = actual <= actual.quantile(0.1)
    forecast_low = forecast <= forecast.quantile(0.1)
    result["missed_discharge_spike"] = actual_high & ~forecast_high
    result["false_spread"] = (forecast_high & ~actual_high) | (forecast_low & ~actual_low)
    result["reversed_ranking"] = np.sign(actual.diff()) != np.sign(forecast.diff())
    result["peak_underestimation"] = actual_high & (error < 0)
    result["incorrect_charge_timing"] = (charge_mw > 0) & ~actual_low
    return result
