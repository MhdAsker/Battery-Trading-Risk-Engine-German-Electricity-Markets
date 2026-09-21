"""Feature creation with explicit publication-time assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    for lag in (24, 48, 168):
        out[f"price_lag_{lag}h"] = price.shift(lag)
    known_price = price.shift(24)
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
