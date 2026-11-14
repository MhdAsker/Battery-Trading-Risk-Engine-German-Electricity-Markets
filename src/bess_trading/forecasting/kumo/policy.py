"""Pre-registered forecast-time policies for conditional external retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bess_trading.forecasting.kumo.regimes import RegimeSnapshot


@dataclass(frozen=True)
class ConditionalRetrievalDecision:
    activate: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalRetrievalPolicy:
    """Activate retrieval only for issue-time risk labels fixed before validation."""

    activate_on: tuple[str, ...] = ("high_volatility", "negative_price_risk")

    def decide(self, snapshot: RegimeSnapshot) -> ConditionalRetrievalDecision:
        reasons = tuple(label for label in self.activate_on if label in snapshot.labels)
        return ConditionalRetrievalDecision(bool(reasons), reasons)

    def combine(
        self,
        direct: pd.Series,
        retrieval_blend: pd.Series,
        snapshot: RegimeSnapshot,
    ) -> tuple[pd.Series, ConditionalRetrievalDecision]:
        if not direct.index.equals(retrieval_blend.index):
            raise ValueError("Conditional-policy forecast indices differ")
        decision = self.decide(snapshot)
        selected = retrieval_blend if decision.activate else direct
        return selected.rename("conditional_regime_kumo"), decision
