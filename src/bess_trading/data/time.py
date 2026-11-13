"""German electricity-market delivery calendar helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd

MARKET_TIMEZONE = "Europe/Berlin"


def delivery_day(index: pd.DatetimeIndex) -> pd.Index:
    """Return the German local delivery date for every UTC timestamp."""
    if index.tz is None:
        raise ValueError("Delivery timestamps must be timezone-aware")
    return pd.Index(index.tz_convert(MARKET_TIMEZONE).date, name="delivery_day_local")


def delivery_day_index(day: date | str, frequency: str = "1h") -> pd.DatetimeIndex:
    """Build one German delivery day in UTC, including 23/25-hour DST days."""
    local_start = pd.Timestamp(day).tz_localize(MARKET_TIMEZONE)
    local_end = local_start + pd.DateOffset(days=1)
    return pd.date_range(local_start, local_end, freq=frequency, inclusive="left").tz_convert("UTC")


def delivery_day_groups(index: pd.DatetimeIndex) -> dict[date, pd.DatetimeIndex]:
    """Group an index by German local delivery date while preserving chronology."""
    keys = delivery_day(index)
    return {day: index[keys == day] for day in keys.unique()}


def forecast_issue_time(day: date | str, hour_local: int = 11) -> pd.Timestamp:
    """Return the configured day-ahead forecast issue time on the previous local day."""
    if not 0 <= hour_local <= 23:
        raise ValueError("hour_local must be between 0 and 23")
    delivery_start = pd.Timestamp(day).tz_localize(MARKET_TIMEZONE)
    issue_local = delivery_start - pd.DateOffset(days=1) + pd.Timedelta(hours=hour_local)
    return issue_local.tz_convert("UTC")


def price_publication_times(
    index: pd.DatetimeIndex, publication_hour_local: int = 13
) -> pd.DatetimeIndex:
    """Approximate publication times for already-cleared day-ahead prices.

    The default is deliberately later than the configured 11:00 forecast snapshot,
    so a delivery day's prices only become eligible for subsequent forecast days.
    """
    local_dates = pd.to_datetime(index.tz_convert(MARKET_TIMEZONE).date)
    published_naive = (
        local_dates - pd.Timedelta(days=1) + pd.Timedelta(hours=publication_hour_local)
    )
    return published_naive.tz_localize(MARKET_TIMEZONE).tz_convert("UTC")


def lagged_price_feature_availability_times(
    index: pd.DatetimeIndex,
    lag_days: int = 1,
    publication_hour_local: int = 13,
) -> pd.DatetimeIndex:
    """Publication time of a local-day-lagged price feature across DST changes."""
    current_dates = pd.to_datetime(index.tz_convert(MARKET_TIMEZONE).date)
    published_naive = (
        current_dates - pd.Timedelta(days=lag_days + 1) + pd.Timedelta(hours=publication_hour_local)
    )
    return published_naive.tz_localize(MARKET_TIMEZONE).tz_convert("UTC")


def interval_hours(index: pd.DatetimeIndex) -> float:
    """Infer a regular market-time-unit duration in hours."""
    if len(index) < 2:
        if index.freq is None:
            raise ValueError("A single timestamp requires an explicit index frequency")
        value = pd.Timedelta(index.freq).total_seconds() / 3600
        if value not in {0.25, 0.5, 1.0}:
            raise ValueError(f"Unsupported interval duration: {value} hours")
        return float(value)
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 3600
    if deltas.nunique() != 1:
        raise ValueError("Dispatch index must have a regular UTC frequency")
    value = float(deltas.iloc[0])
    if value not in {0.25, 0.5, 1.0}:
        raise ValueError(f"Unsupported interval duration: {value} hours")
    return value
