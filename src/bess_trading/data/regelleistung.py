"""Strict adapters for authoritative German balancing-market exports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ALLOWED_SOURCES = {"regelleistung.net", "netztransparenz.de", "entsoe.eu"}


@dataclass(frozen=True)
class MarketDataProvenance:
    source: str
    downloaded_at_utc: str
    source_url: str
    sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_market_export(
    path: str | Path,
    *,
    source: str,
    source_url: str,
    downloaded_at_utc: str,
    timestamp_column: str = "timestamp_utc",
) -> tuple[pd.DataFrame, MarketDataProvenance]:
    """Load a CSV export and attach immutable source provenance."""
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"Unsupported market-data source: {source}")
    target = Path(path)
    frame = pd.read_csv(target)
    if timestamp_column not in frame:
        raise ValueError(f"Missing {timestamp_column}")
    timestamps = pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise")
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp_utc")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Market export timestamps must be unique and sorted")
    provenance = MarketDataProvenance(
        source=source,
        downloaded_at_utc=downloaded_at_utc,
        source_url=source_url,
        sha256=file_sha256(target),
    )
    return frame, provenance


def require_real_reserve_data(
    frame: pd.DataFrame,
    provenance: dict[str, MarketDataProvenance],
    required_columns: set[str],
) -> None:
    """Fail closed when reserve settlement lacks authoritative observations."""
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reserve observations: {sorted(missing)}")
    if frame[list(required_columns)].isna().any().any():
        raise ValueError("Reserve observations contain missing values")
    required_records = {"reserve_capacity_awards", "reserve_activations", "activation_prices"}
    missing_records = required_records - set(provenance)
    if missing_records:
        raise ValueError(
            "Reserve data requires award, activation and price provenance: "
            f"{sorted(missing_records)}"
        )
    sources = {item.source for item in provenance.values()}
    if not sources or not sources <= ALLOWED_SOURCES:
        raise ValueError("Reserve data requires approved source provenance")
