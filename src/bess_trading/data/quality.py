"""Data-contract checks for time-indexed market data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    """Machine-readable validation outcome."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_market_frame(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...] = ("price_eur_mwh",),
    expected_frequency: str = "1h",
) -> QualityReport:
    """Check schema, UTC, duplicates, gaps, frequency, and implausible values."""
    report = QualityReport()
    if frame.empty:
        report.errors.append("Frame is empty")
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        report.errors.append(f"Missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        report.errors.append("Index must be a DatetimeIndex")
        return report
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        report.errors.append("Index must be timezone-aware UTC")
    if frame.index.has_duplicates:
        report.errors.append(f"Duplicated timestamps: {int(frame.index.duplicated().sum())}")
    if not frame.index.is_monotonic_increasing:
        report.errors.append("Timestamps are not sorted")
    if len(frame.index) > 1:
        expected = pd.date_range(frame.index.min(), frame.index.max(), freq=expected_frequency)
        # UTC date ranges avoid local DST gaps and repetitions.
        actual_deltas = frame.index.to_series().diff().dropna()
        step = pd.Timedelta(expected_frequency)
        gaps = int((actual_deltas > step).sum())
        odd = int(((actual_deltas != step) & (actual_deltas <= step)).sum())
        if gaps:
            report.warnings.append(f"Missing intervals/gaps: {gaps}")
        if odd or len(expected) != len(frame):
            report.warnings.append("Unexpected frequency or irregular timestamps")
    numeric = frame.select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy()).any():
        report.errors.append("Infinite values found")
    if (
        "price_eur_mwh" in frame
        and ((frame["price_eur_mwh"] < -1000) | (frame["price_eur_mwh"] > 10000)).any()
    ):
        report.warnings.append("Prices outside conservative plausibility range [-1000, 10000]")
    for col in ("load_mw", "wind_mw", "solar_mw"):
        if col in frame and (frame[col] < 0).any():
            report.errors.append(f"Negative physical quantity in {col}")
    return report


def localize_german_timestamp(values: pd.Series) -> pd.DatetimeIndex:
    """Convert local German clock times to UTC, rejecting ambiguous DST input.

    Local CSVs should ideally include offsets. Naive autumn clock-change rows are
    intrinsically ambiguous and therefore rejected instead of silently guessed.
    """
    parsed = pd.to_datetime(values)
    if getattr(parsed.dt, "tz", None) is not None:
        return pd.DatetimeIndex(parsed.dt.tz_convert("UTC"))
    try:
        localized = parsed.dt.tz_localize("Europe/Berlin", ambiguous="raise", nonexistent="raise")
    except Exception as exc:
        raise ValueError(
            "Naive local timestamps contain an ambiguous or nonexistent DST time; "
            "provide explicit UTC offsets"
        ) from exc
    return pd.DatetimeIndex(localized.dt.tz_convert("UTC"))
