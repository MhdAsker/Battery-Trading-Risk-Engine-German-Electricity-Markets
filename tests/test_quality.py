import pandas as pd
import pytest

from bess_trading.data.quality import localize_german_timestamp, validate_market_frame


def test_quality_rejects_empty_frame() -> None:
    report = validate_market_frame(
        pd.DataFrame(columns=["price_eur_mwh"], index=pd.DatetimeIndex([], tz="UTC"))
    )
    assert not report.valid
    assert "empty" in report.errors[0].lower()


def test_quality_detects_duplicate() -> None:
    idx = pd.DatetimeIndex(["2024-01-01T00:00Z", "2024-01-01T00:00Z"])
    report = validate_market_frame(pd.DataFrame({"price_eur_mwh": [1.0, 2.0]}, index=idx))
    assert not report.valid
    assert "Duplicated" in report.errors[0]


def test_naive_ambiguous_dst_rejected() -> None:
    with pytest.raises(ValueError):
        localize_german_timestamp(pd.Series(["2024-10-27 02:30"]))
