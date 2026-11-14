import numpy as np
import pandas as pd

from bess_trading.forecasting.kumo.regimes import RegimeThresholds
from bess_trading.forecasting.kumo.retrieval import (
    RegimeAwareRetriever,
    blend_direct_and_retrieval,
)


def prices() -> pd.Series:
    index = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    values = 40 + 15 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    values += 5 * np.sin(2 * np.pi * np.arange(len(index)) / (24 * 30))
    return pd.Series(values, index=index)


def test_external_retrieval_continuations_are_available_before_query() -> None:
    retriever = RegimeAwareRetriever(
        RegimeThresholds(negative_price_share=0.5, volatility_eur_mwh=20),
        context_hours=168,
        k=3,
    )
    result = retriever.forecast(prices(), "2023-12-15", regime_conditioned=True)
    assert len(result.forecast) == 24
    issue = pd.Timestamp("2023-12-14 10:00:00Z")
    assert (pd.to_datetime(result.diagnostics.continuation_available_at_utc) <= issue).all()
    assert (pd.to_datetime(result.diagnostics.candidate_issue_time_utc) < issue).all()
    assert np.isclose(result.diagnostics.weight.sum(), 1.0)


def test_external_blend_is_explicit_and_index_safe() -> None:
    index = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
    direct = pd.Series(10.0, index=index)
    retrieved = pd.Series(20.0, index=index)
    blended = blend_direct_and_retrieval(direct, retrieved, 0.25)
    assert (blended == 17.5).all()
