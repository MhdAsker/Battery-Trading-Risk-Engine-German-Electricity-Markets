"""External electricity-regime-aware historical continuation retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from bess_trading.data.time import delivery_day_index, forecast_issue_time, price_publication_times
from bess_trading.forecasting.kumo.regimes import (
    RegimeSnapshot,
    RegimeThresholds,
    build_regime_snapshot,
)


@dataclass(frozen=True)
class RetrievalResult:
    forecast: pd.Series
    diagnostics: pd.DataFrame
    query_regimes: tuple[str, ...]
    used_regime_filter: bool


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(values.std(ddof=0))
    return (values - values.mean()) / (scale if scale > 1e-8 else 1.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def _published_known(prices: pd.Series, issue: pd.Timestamp) -> pd.Series:
    publications = pd.Series(price_publication_times(prices.index), index=prices.index)
    return prices[(publications <= issue).to_numpy()].dropna().sort_index()


class RegimeAwareRetriever:
    """Retrieve causal historical day continuations with auditable regime filters."""

    def __init__(
        self,
        thresholds: RegimeThresholds,
        *,
        context_hours: int = 168,
        k: int = 10,
        temperature: float = 0.1,
        minimum_regime_neighbors: int = 3,
    ) -> None:
        if context_hours < 24 or k < 1 or temperature <= 0:
            raise ValueError("Invalid external retrieval configuration")
        self.thresholds = thresholds
        self.context_hours = context_hours
        self.k = k
        self.temperature = temperature
        self.minimum_regime_neighbors = minimum_regime_neighbors

    def _candidate(
        self,
        prices: pd.Series,
        candidate_day: date,
        query_issue: pd.Timestamp,
        query_hours: int,
    ) -> tuple[np.ndarray, np.ndarray, RegimeSnapshot, dict[str, object]] | None:
        candidate_issue = forecast_issue_time(candidate_day, 11)
        if candidate_issue >= query_issue:
            return None
        known_then = _published_known(prices, candidate_issue)
        if len(known_then) < self.context_hours:
            return None
        context = known_then.iloc[-self.context_hours :]
        expected = pd.date_range(context.index.min(), context.index.max(), freq="h", tz="UTC")
        if not context.index.equals(expected):
            return None
        continuation_index = delivery_day_index(candidate_day, "1h")
        if len(continuation_index) != query_hours:
            return None
        continuation = prices.reindex(continuation_index)
        if continuation.isna().any():
            return None
        continuation_publication = price_publication_times(continuation_index).max()
        if continuation_publication > query_issue:
            return None
        snapshot = build_regime_snapshot(prices, candidate_day, self.thresholds)
        metadata: dict[str, object] = {
            "candidate_delivery_day": str(candidate_day),
            "candidate_issue_time_utc": candidate_issue,
            "context_start_utc": context.index.min(),
            "context_end_utc": context.index.max(),
            "continuation_available_at_utc": continuation_publication,
        }
        return context.to_numpy(dtype=float), continuation.to_numpy(dtype=float), snapshot, metadata

    def forecast(
        self,
        prices: pd.Series,
        delivery_day: date | str,
        *,
        regime_conditioned: bool = True,
        issue_hour_local: int = 11,
    ) -> RetrievalResult:
        query_issue = forecast_issue_time(delivery_day, issue_hour_local)
        query_horizon = delivery_day_index(delivery_day, "1h")
        query_known = _published_known(prices, query_issue)
        if len(query_known) < self.context_hours:
            raise ValueError("Query has insufficient published context")
        query_context = query_known.iloc[-self.context_hours :]
        expected = pd.date_range(
            query_context.index.min(), query_context.index.max(), freq="h", tz="UTC"
        )
        if not query_context.index.equals(expected):
            raise ValueError("Query context contains gaps")
        query_snapshot = build_regime_snapshot(prices, delivery_day, self.thresholds)
        start_day = prices.index.min().tz_convert("Europe/Berlin").date()
        last_candidate = pd.Timestamp(delivery_day) - pd.Timedelta(days=1)
        candidate_days = pd.date_range(start_day, last_candidate, freq="D")
        candidates = []
        for stamp in candidate_days:
            candidate = self._candidate(
                prices, stamp.date(), query_issue, len(query_horizon)
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise ValueError("No causal retrieval candidates")

        economic_labels = {
            "negative_price_risk",
            "high_volatility",
            "scarcity_risk",
            "renewable_surplus",
        }
        query_economic = economic_labels.intersection(query_snapshot.labels)
        regime_candidates = [
            candidate
            for candidate in candidates
            if query_economic
            and query_economic.intersection(candidate[2].labels)
        ]
        use_filter = regime_conditioned and len(regime_candidates) >= self.minimum_regime_neighbors
        pool = regime_candidates if use_filter else candidates
        query_vector = _zscore(query_context.to_numpy(dtype=float))
        rows = []
        continuations = []
        for context, continuation, snapshot, metadata in pool:
            similarity = _cosine(query_vector, _zscore(context))
            level_shift = float(query_context.mean() - context.mean())
            continuations.append(continuation + level_shift)
            rows.append(
                {
                    **metadata,
                    "similarity": similarity,
                    "level_shift_eur_mwh": level_shift,
                    "candidate_regimes": "|".join(snapshot.labels),
                    "regime_match": bool(set(query_snapshot.labels).intersection(snapshot.labels)),
                }
            )
        diagnostics = pd.DataFrame(rows).sort_values("similarity", ascending=False).head(self.k)
        selected = diagnostics.index.to_numpy()
        similarities = diagnostics.similarity.to_numpy(dtype=float)
        logits = (similarities - similarities.max()) / self.temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        curves = np.asarray([continuations[position] for position in selected])
        forecast = pd.Series(weights @ curves, index=query_horizon, name="retrieval_p50")
        diagnostics = diagnostics.copy()
        diagnostics["weight"] = weights
        diagnostics["query_regimes"] = "|".join(query_snapshot.labels)
        diagnostics["rank"] = np.arange(1, len(diagnostics) + 1)
        return RetrievalResult(
            forecast,
            diagnostics.reset_index(drop=True),
            query_snapshot.labels,
            use_filter,
        )


def blend_direct_and_retrieval(
    direct: pd.Series, retrieval: pd.Series, direct_weight: float
) -> pd.Series:
    """External research blend; this is not NVIDIA native DARR."""
    if not direct.index.equals(retrieval.index):
        raise ValueError("Direct and retrieval forecast indices differ")
    if not 0 <= direct_weight <= 1:
        raise ValueError("direct_weight must be in [0, 1]")
    return (direct_weight * direct + (1 - direct_weight) * retrieval).rename(
        "regime_kumo_p50"
    )
