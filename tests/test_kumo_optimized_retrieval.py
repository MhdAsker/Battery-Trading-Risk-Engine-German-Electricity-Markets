from __future__ import annotations

import numpy as np
import pandas as pd

from bess_trading.forecasting.kumo.policy import ConditionalRetrievalPolicy
from bess_trading.forecasting.kumo.regimes import RegimeSnapshot, RegimeThresholds
from bess_trading.forecasting.kumo.retrieval import (
    OptimizedRegimeAwareRetriever,
    RegimeAwareRetriever,
)


def _prices() -> pd.Series:
    index = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    values = 60 + 25 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    values += 5 * np.sin(2 * np.pi * np.arange(len(index)) / (24 * 7))
    values += np.random.default_rng(7).normal(0, 0.01, len(index))
    return pd.Series(values, index=index, name="price_eur_mwh")


def test_optimized_retrieval_is_numerically_equivalent_to_reference() -> None:
    prices = _prices()
    thresholds = RegimeThresholds(negative_price_share=0.02, volatility_eur_mwh=20)
    reference = RegimeAwareRetriever(thresholds, context_hours=168, k=5)
    optimized = OptimizedRegimeAwareRetriever(
        prices, thresholds, context_hours=168, k=5, cache_dir=None
    )
    expected = reference.forecast(prices, "2023-06-15", regime_conditioned=True)
    actual = optimized.forecast("2023-06-15", regime_conditioned=True)
    np.testing.assert_allclose(actual.forecast, expected.forecast, rtol=0, atol=1e-10)
    assert actual.diagnostics.candidate_delivery_day.tolist() == (
        expected.diagnostics.candidate_delivery_day.tolist()
    )
    np.testing.assert_allclose(actual.diagnostics.weight, expected.diagnostics.weight, atol=1e-12)
    assert actual.used_regime_filter == expected.used_regime_filter


def test_optimized_archive_rejects_future_candidate_injection() -> None:
    prices = _prices()
    changed = prices.copy()
    changed.loc[changed.index >= "2023-06-15"] = 999999.0
    thresholds = RegimeThresholds(0.02, 20)
    first = OptimizedRegimeAwareRetriever(prices, thresholds, cache_dir=None).forecast(
        "2023-06-15"
    )
    second = OptimizedRegimeAwareRetriever(changed, thresholds, cache_dir=None).forecast(
        "2023-06-15"
    )
    np.testing.assert_allclose(first.forecast, second.forecast, atol=1e-10)
    assert (
        first.diagnostics.continuation_available_at_utc
        <= pd.Timestamp("2023-06-14 09:00Z")
    ).all()


def test_conditional_policy_uses_only_snapshot_labels() -> None:
    policy = ConditionalRetrievalPolicy()
    issue = pd.Timestamp("2023-06-14 09:00Z")
    ordinary = RegimeSnapshot(issue, "2023-06-15", ("weekday", "season_3"), {})
    risky = RegimeSnapshot(
        issue,
        "2023-06-15",
        ("negative_price_risk", "weekday", "season_3"),
        {},
    )
    assert policy.decide(ordinary).activate is False
    decision = policy.decide(risky)
    assert decision.activate is True
    assert decision.reasons == ("negative_price_risk",)
