"""Minimal, auditable ENTSO-E Transparency Platform client.

The API uses UTC query boundaries. XML periods may have 15, 30, or 60 minute
resolution; observations are expanded from each period start and then normalized.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

LOGGER = logging.getLogger(__name__)
API_URL = "https://web-api.tp.entsoe.eu/api"
GERMANY_LUXEMBOURG = "10Y1001A1001A82H"
GERMANY_CONTROL_AREA = "10Y1001A1001A83F"


class EntsoeError(RuntimeError):
    """Raised for ENTSO-E transport or response errors."""


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _api_time(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d%H%M")


def _duration(value: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(value)
    except ValueError as exc:
        raise EntsoeError(f"Unsupported ENTSO-E resolution: {value}") from exc


@dataclass
class EntsoeClient:
    """Credential-safe client for a focused set of ENTSO-E endpoints."""

    token: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        # httpx INFO logs include full query strings, including ENTSO-E credentials.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.token = self.token or os.getenv("ENTSOE_API_TOKEN")
        if not self.token:
            raise ValueError("Set ENTSOE_API_TOKEN; never commit the token")

    def _request(self, params: dict[str, str]) -> bytes:
        safe_params = {**params, "securityToken": self.token or ""}
        response = httpx.get(API_URL, params=safe_params, timeout=self.timeout_seconds)
        if response.status_code != 200:
            # Never include the requested URL because it contains the credential.
            raise EntsoeError(
                f"ENTSO-E returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.content

    def day_ahead_prices(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        zone: str = GERMANY_LUXEMBOURG,
    ) -> pd.DataFrame:
        """Download day-ahead prices in EUR/MWh (document type A44)."""
        return self._series_request(
            {
                "documentType": "A44",
                "contract_MarketAgreement.Type": "A01",
                "in_Domain": zone,
                "out_Domain": zone,
            },
            start,
            end,
            "price_eur_mwh",
            "price.amount",
            classification_position="1",
        )

    def actual_load(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        zone: str = GERMANY_CONTROL_AREA,
    ) -> pd.DataFrame:
        """Download actual total load in MW (document type A65)."""
        return self._series_request(
            {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": zone},
            start,
            end,
            "load_mw",
            "quantity",
        )

    def generation(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        zone: str = GERMANY_CONTROL_AREA,
    ) -> pd.DataFrame:
        """Download actual generation by production type (document type A75)."""
        raw = self._request_with_dates(
            {"documentType": "A75", "processType": "A16", "in_Domain": zone}, start, end
        )
        long = parse_timeseries(raw, value_tag="quantity", include_psr=True)
        if long.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp_utc"))
        result = long.pivot_table(
            index="timestamp_utc", columns="psr_type", values="value", aggfunc="sum"
        )
        return result.sort_index().rename_axis(columns=None)

    def _request_with_dates(self, params: dict[str, str], start: object, end: object) -> bytes:
        start_utc, end_utc = _utc(start), _utc(end)
        if end_utc <= start_utc:
            raise ValueError("end must be after start")
        return self._request(
            {**params, "periodStart": _api_time(start_utc), "periodEnd": _api_time(end_utc)}
        )

    def _series_request(
        self,
        params: dict[str, str],
        start: object,
        end: object,
        column: str,
        tag: str,
        classification_position: str | None = None,
    ) -> pd.DataFrame:
        raw = self._request_with_dates(params, start, end)
        frame = parse_timeseries(
            raw,
            value_tag=tag,
            classification_position=classification_position,
        )
        start_utc, end_utc = _utc(start), _utc(end)
        if not frame.empty:
            frame = frame[
                (frame["timestamp_utc"] >= start_utc) & (frame["timestamp_utc"] < end_utc)
            ]
        if frame.empty:
            return pd.DataFrame(
                columns=[column], index=pd.DatetimeIndex([], tz="UTC", name="timestamp_utc")
            )
        return frame.groupby("timestamp_utc")["value"].mean().to_frame(column).sort_index()


def parse_timeseries(
    xml: bytes,
    value_tag: str,
    include_psr: bool = False,
    classification_position: str | None = None,
) -> pd.DataFrame:
    """Parse ENTSO-E XML without depending on namespace version strings."""
    root = ET.fromstring(xml)
    rows: list[dict[str, object]] = []
    for series in root.findall(".//{*}TimeSeries"):
        classification = series.find(
            "./{*}classificationSequence_AttributeInstanceComponent.position"
        )
        if classification_position is not None and (
            classification is None or classification.text != classification_position
        ):
            continue
        psr_node = series.find(".//{*}MktPSRType/{*}psrType")
        psr = psr_node.text if psr_node is not None else "unknown"
        for period in series.findall("./{*}Period"):
            start_node = period.find("./{*}timeInterval/{*}start")
            resolution_node = period.find("./{*}resolution")
            if start_node is None or resolution_node is None:
                continue
            start = _utc(start_node.text or "")
            step = _duration(resolution_node.text or "")
            for point in period.findall("./{*}Point"):
                pos = point.find("./{*}position")
                val = point.find(f"./{{*}}{value_tag}")
                if pos is None or val is None or pos.text is None or val.text is None:
                    continue
                row: dict[str, object] = {
                    "timestamp_utc": start + (int(pos.text) - 1) * step,
                    "value": float(val.text),
                }
                if include_psr:
                    row["psr_type"] = psr
                rows.append(row)
    return pd.DataFrame(rows)


def save_raw(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist immutable API output as parquet using an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(target)
    LOGGER.info("Saved %s rows to %s", len(frame), target)
    return target
