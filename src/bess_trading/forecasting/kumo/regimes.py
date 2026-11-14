"""Forecast-time electricity-market regime assignment using known information only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from bess_trading.data.time import delivery_day_index, forecast_issue_time, price_publication_times


@dataclass(frozen=True)
class RegimeThresholds:
    """Thresholds fitted on development snapshots, never final-test outcomes."""

    negative_price_share: float
    volatility_eur_mwh: float
    residual_load_mw: float | None = None
    renewable_share: float | None = None


@dataclass(frozen=True)
class RegimeSnapshot:
    issue_time_utc: pd.Timestamp
    delivery_day_local: str
    labels: tuple[str, ...]
    features: dict[str, float]


def fit_regime_thresholds(snapshots: pd.DataFrame, cutoff: pd.Timestamp) -> RegimeThresholds:
    """Fit robust regime thresholds using snapshots strictly before a cutoff."""
    if snapshots.index.tz is None:
        raise ValueError("Snapshot issue times must be timezone-aware")
    development = snapshots[snapshots.index < cutoff]
    if development.empty:
        raise ValueError("No pre-cutoff regime snapshots")
    return RegimeThresholds(
        negative_price_share=float(development.recent_negative_share.quantile(0.75)),
        volatility_eur_mwh=float(development.recent_volatility_eur_mwh.quantile(0.75)),
        residual_load_mw=(
            float(development.residual_load_forecast_mw.quantile(0.75))
            if "residual_load_forecast_mw" in development
            else None
        ),
        renewable_share=(
            float(development.renewable_share_forecast.quantile(0.75))
            if "renewable_share_forecast" in development
            else None
        ),
    )


def build_regime_snapshot(
    prices: pd.Series,
    delivery_day: date | str,
    thresholds: RegimeThresholds,
    *,
    issue_hour_local: int = 11,
    fundamental_forecasts: pd.DataFrame | None = None,
    fundamental_available_at: pd.Series | None = None,
) -> RegimeSnapshot:
    """Assign regimes from published prices and optional archived forecasts."""
    issue = forecast_issue_time(delivery_day, issue_hour_local)
    publications = pd.Series(price_publication_times(prices.index), index=prices.index)
    known = prices[(publications <= issue).to_numpy()].dropna().sort_index()
    if len(known) < 168:
        raise ValueError("Regime assignment requires 168 published hourly prices")
    recent = known.iloc[-168:]
    features: dict[str, float] = {
        "recent_negative_share": float((recent < 0).mean()),
        "recent_volatility_eur_mwh": float(recent.diff().std(ddof=1)),
        "recent_mean_price_eur_mwh": float(recent.mean()),
        "recent_daily_spread_eur_mwh": float(recent.iloc[-24:].max() - recent.iloc[-24:].min()),
    }
    if fundamental_forecasts is not None:
        horizon = delivery_day_index(delivery_day, "1h")
        if fundamental_available_at is None:
            raise ValueError("Fundamental forecasts require availability timestamps")
        availability = fundamental_available_at.reindex(horizon)
        if availability.isna().any() or (availability > issue).any():
            raise ValueError("Fundamental forecast was unavailable at regime assignment time")
        block = fundamental_forecasts.reindex(horizon)
        if block.isna().any().any():
            raise ValueError("Fundamental forecast block is incomplete")
        for column in ("residual_load_forecast_mw", "renewable_share_forecast"):
            if column in block:
                features[column] = float(block[column].mean())

    labels: list[str] = []
    if features["recent_negative_share"] >= thresholds.negative_price_share:
        labels.append("negative_price_risk")
    if features["recent_volatility_eur_mwh"] >= thresholds.volatility_eur_mwh:
        labels.append("high_volatility")
    if thresholds.residual_load_mw is not None and (
        features.get("residual_load_forecast_mw", -np.inf) >= thresholds.residual_load_mw
    ):
        labels.append("scarcity_risk")
    if thresholds.renewable_share is not None and (
        features.get("renewable_share_forecast", -np.inf) >= thresholds.renewable_share
    ):
        labels.append("renewable_surplus")
    local_day = pd.Timestamp(delivery_day)
    labels.append("weekend" if local_day.weekday() >= 5 else "weekday")
    labels.append(f"season_{((local_day.month % 12) // 3) + 1}")
    return RegimeSnapshot(issue, str(delivery_day), tuple(labels), features)


def snapshot_frame(snapshots: list[RegimeSnapshot]) -> pd.DataFrame:
    """Convert auditable snapshots to a threshold-fitting frame."""
    return pd.DataFrame(
        [snapshot.features for snapshot in snapshots],
        index=pd.DatetimeIndex([snapshot.issue_time_utc for snapshot in snapshots]),
    ).sort_index()
