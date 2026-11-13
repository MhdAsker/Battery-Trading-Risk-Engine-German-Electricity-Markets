import pandas as pd
import pytest

from bess_trading.data.ranges import annual_chunks


def test_annual_chunks_cover_range_without_overlap() -> None:
    chunks = list(annual_chunks("2021-01-01", "2024-06-01"))
    assert len(chunks) == 4
    assert chunks[0][0] == pd.Timestamp("2021-01-01", tz="UTC")
    assert chunks[-1][1] == pd.Timestamp("2024-06-01", tz="UTC")
    assert all(chunks[position][1] == chunks[position + 1][0] for position in range(3))


def test_annual_chunks_reject_invalid_range() -> None:
    with pytest.raises(ValueError):
        list(annual_chunks("2024-01-01", "2024-01-01"))
