"""Feature creation with explicit publication-time assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _local_day_lag(series: pd.Series, days: int) -> pd.Series:
    """Lag by German calendar day and local hour rather than a fixed row count."""
    local = series.index.tz_convert("Europe/Berlin")
    source_keys = pd.MultiIndex.from_arrays([local.date, local.hour], names=["date", "hour"])
    # Repeated autumn hours are both legitimately known; their mean gives one stable
    # previous-day local-hour value for the next day.
    lookup = pd.Series(series.to_numpy(), index=source_keys).groupby(level=[0, 1]).mean()
    target_dates = (pd.to_datetime(local.date) - pd.Timedelta(days=days)).date
    target_keys = pd.MultiIndex.from_arrays([target_dates, local.hour])
    return pd.Series(lookup.reindex(target_keys).to_numpy(), index=series.index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features for a day-ahead forecast made before the previous day clears.

    Price-derived features are lagged at least 24 hours. Fundamental columns in the
    input must be day-ahead forecasts, not realized load/generation. Their names must
    end in ``_forecast_mw`` to make that contract visible.
    """
    if "price_eur_mwh" not in frame:
        raise ValueError("price_eur_mwh is required")
    out = pd.DataFrame(index=frame.index)
    local = frame.index.tz_convert("Europe/Berlin")
    out["hour"] = local.hour
    out["weekday"] = local.weekday
    out["weekend"] = (local.weekday >= 5).astype(int)
    out["month"] = local.month
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    price = frame["price_eur_mwh"]
    for lag, days in ((24, 1), (48, 2), (168, 7)):
        out[f"price_lag_{lag}h"] = _local_day_lag(price, days)
    known_price = out["price_lag_24h"]
    out["price_rolling_24h"] = known_price.rolling(24, min_periods=24).mean()
    out["price_rolling_168h"] = known_price.rolling(168, min_periods=168).mean()
    allowed = [c for c in frame if c.endswith("_forecast_mw")]
    out[allowed] = frame[allowed]
    load, wind, solar = "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"
    if {load, wind, solar}.issubset(out.columns):
        out["renewables_forecast_mw"] = out[wind] + out[solar]
        out["residual_load_forecast_mw"] = out[load] - out["renewables_forecast_mw"]
        out["renewable_share_forecast"] = out["renewables_forecast_mw"] / out[load].replace(
            0, np.nan
        )
    return out
