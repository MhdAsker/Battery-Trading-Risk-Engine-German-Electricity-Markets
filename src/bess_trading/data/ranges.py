"""Date-range chunking for reliable multi-year ENTSO-E acquisition."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def annual_chunks(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield half-open UTC chunks no longer than one calendar year."""
    cursor = pd.Timestamp(start)
    limit = pd.Timestamp(end)
    cursor = cursor.tz_localize("UTC") if cursor.tzinfo is None else cursor.tz_convert("UTC")
    limit = limit.tz_localize("UTC") if limit.tzinfo is None else limit.tz_convert("UTC")
    if limit <= cursor:
        raise ValueError("end must be after start")
    while cursor < limit:
        chunk_end = min(cursor + pd.DateOffset(years=1), limit)
        yield cursor, chunk_end
        cursor = chunk_end
