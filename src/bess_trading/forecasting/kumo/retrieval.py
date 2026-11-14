"""External electricity-regime-aware historical continuation retrieval."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

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
    archive_key: str | None = None
    forecast_cache_key: str | None = None
    eligible_candidates: int | None = None
    runtime_seconds: float | None = None


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(values.std(ddof=0))
    return (values - values.mean()) / (scale if scale > 1e-8 else 1.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def _published_known(prices: pd.Series, issue: pd.Timestamp) -> pd.Series:
    publications = pd.Series(price_publication_times(prices.index), index=prices.index)
    return prices[(publications <= issue).to_numpy()].dropna().sort_index()


def _series_fingerprint(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


@dataclass(frozen=True)
class RetrievalArchive:
    """Reusable candidate windows with explicit information-time provenance."""

    contexts: np.ndarray
    normalized_contexts: np.ndarray
    context_means: np.ndarray
    continuations: np.ndarray
    continuation_hours: np.ndarray
    candidate_days: np.ndarray
    candidate_issue_ns: np.ndarray
    context_start_ns: np.ndarray
    context_end_ns: np.ndarray
    continuation_available_ns: np.ndarray
    labels: np.ndarray
    archive_key: str
    data_cutoff_utc: str

    @classmethod
    def build(
        cls,
        prices: pd.Series,
        thresholds: RegimeThresholds,
        *,
        context_hours: int,
        issue_hour_local: int = 11,
        cache_dir: str | Path | None = "artifacts/kumo_retrieval_cache",
    ) -> RetrievalArchive:
        """Build or load a vintage-keyed archive without relaxing candidate eligibility."""
        if prices.index.tz is None or not prices.index.is_monotonic_increasing:
            raise ValueError("Archive prices must have an ordered timezone-aware index")
        payload = {
            "version": 1,
            "price_fingerprint": _series_fingerprint(prices),
            "price_start": str(prices.index.min()),
            "price_cutoff": str(prices.index.max()),
            "thresholds": asdict(thresholds),
            "context_hours": context_hours,
            "issue_hour_local": issue_hour_local,
        }
        archive_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        target = Path(cache_dir) / f"{archive_key}.npz" if cache_dir is not None else None
        if target is not None and target.exists():
            cached = np.load(target, allow_pickle=False)
            return cls(
                **{name: cached[name] for name in cached.files if name != "archive_key"},
                archive_key=str(cached["archive_key"].item()),
            )

        valid = prices.dropna().sort_index()
        valid_publication = price_publication_times(valid.index).asi8
        values = valid.to_numpy(dtype=float)
        timestamps = valid.index.asi8
        full_index = prices.index
        full_values = prices.to_numpy(dtype=float)
        day_start = prices.index.min().tz_convert("Europe/Berlin").date()
        day_end = prices.index.max().tz_convert("Europe/Berlin").date()
        rows: list[tuple[object, ...]] = []
        for stamp in pd.date_range(day_start, day_end, freq="D"):
            day = stamp.date()
            issue = forecast_issue_time(day, issue_hour_local)
            stop = int(np.searchsorted(valid_publication, issue.value, side="right"))
            if stop < context_hours:
                continue
            context_values = values[stop - context_hours : stop]
            context_times = timestamps[stop - context_hours : stop]
            if np.any(np.diff(context_times) != pd.Timedelta(hours=1).value):
                continue
            horizon = delivery_day_index(day, "1h")
            positions = full_index.get_indexer(horizon)
            if (positions < 0).any():
                continue
            continuation = full_values[positions]
            if np.isnan(continuation).any():
                continue
            continuation_available = price_publication_times(horizon).max()
            recent = context_values[-168:]
            features = {
                "negative": float(np.mean(recent < 0)),
                "volatility": float(np.std(np.diff(recent), ddof=1)),
            }
            labels: list[str] = []
            if features["negative"] >= thresholds.negative_price_share:
                labels.append("negative_price_risk")
            if features["volatility"] >= thresholds.volatility_eur_mwh:
                labels.append("high_volatility")
            labels.append("weekend" if stamp.weekday() >= 5 else "weekday")
            labels.append(f"season_{((stamp.month % 12) // 3) + 1}")
            padded = np.full(25, np.nan, dtype=float)
            padded[: len(continuation)] = continuation
            rows.append(
                (
                    context_values,
                    padded,
                    len(continuation),
                    str(day),
                    issue.value,
                    context_times[0],
                    context_times[-1],
                    continuation_available.value,
                    "|".join(labels),
                )
            )
        if not rows:
            raise ValueError("No valid historical retrieval candidates")
        contexts = np.stack([row[0] for row in rows])
        means = contexts.mean(axis=1)
        scales = contexts.std(axis=1)
        scales[scales <= 1e-8] = 1.0
        normalized = (contexts - means[:, None]) / scales[:, None]
        archive = cls(
            contexts=contexts,
            normalized_contexts=normalized,
            context_means=means,
            continuations=np.stack([row[1] for row in rows]),
            continuation_hours=np.asarray([row[2] for row in rows], dtype=np.int16),
            candidate_days=np.asarray([row[3] for row in rows]),
            candidate_issue_ns=np.asarray([row[4] for row in rows], dtype=np.int64),
            context_start_ns=np.asarray([row[5] for row in rows], dtype=np.int64),
            context_end_ns=np.asarray([row[6] for row in rows], dtype=np.int64),
            continuation_available_ns=np.asarray([row[7] for row in rows], dtype=np.int64),
            labels=np.asarray([row[8] for row in rows]),
            archive_key=archive_key,
            data_cutoff_utc=str(prices.index.max()),
        )
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **asdict(archive))
        return archive


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


class OptimizedRegimeAwareRetriever:
    """Vectorized retriever backed by a reusable, vintage-keyed causal archive."""

    def __init__(
        self,
        prices: pd.Series,
        thresholds: RegimeThresholds,
        *,
        context_hours: int = 168,
        k: int = 10,
        temperature: float = 0.1,
        minimum_regime_neighbors: int = 3,
        issue_hour_local: int = 11,
        cache_dir: str | Path | None = "artifacts/kumo_retrieval_cache",
    ) -> None:
        if context_hours < 168 or k < 1 or temperature <= 0:
            raise ValueError("Invalid optimized retrieval configuration")
        self.prices = prices
        self.thresholds = thresholds
        self.context_hours = context_hours
        self.k = k
        self.temperature = temperature
        self.minimum_regime_neighbors = minimum_regime_neighbors
        self.issue_hour_local = issue_hour_local
        self.archive = RetrievalArchive.build(
            prices,
            thresholds,
            context_hours=context_hours,
            issue_hour_local=issue_hour_local,
            cache_dir=cache_dir,
        )
        valid = prices.dropna().sort_index()
        self._valid_values = valid.to_numpy(dtype=float)
        self._valid_timestamps = valid.index.asi8
        self._valid_publication = price_publication_times(valid.index).asi8

    def _query_context(self, issue: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        stop = int(np.searchsorted(self._valid_publication, issue.value, side="right"))
        if stop < self.context_hours:
            raise ValueError("Query has insufficient published context")
        values = self._valid_values[stop - self.context_hours : stop]
        timestamps = self._valid_timestamps[stop - self.context_hours : stop]
        if np.any(np.diff(timestamps) != pd.Timedelta(hours=1).value):
            raise ValueError("Query context contains gaps")
        return values, timestamps

    def forecast(
        self,
        delivery_day: date | str,
        *,
        regime_conditioned: bool = True,
    ) -> RetrievalResult:
        """Return an equivalent forecast while filtering the archive at the query cutoff."""
        started = time.perf_counter()
        query_issue = forecast_issue_time(delivery_day, self.issue_hour_local)
        horizon = delivery_day_index(delivery_day, "1h")
        query_context, query_timestamps = self._query_context(query_issue)
        query_snapshot = build_regime_snapshot(
            self.prices,
            delivery_day,
            self.thresholds,
            issue_hour_local=self.issue_hour_local,
        )
        archive = self.archive
        eligible = (
            (archive.candidate_issue_ns < query_issue.value)
            & (archive.continuation_available_ns <= query_issue.value)
            & (archive.continuation_hours == len(horizon))
            & (archive.candidate_days < str(pd.Timestamp(delivery_day).date()))
        )
        if not eligible.any():
            raise ValueError("No causal retrieval candidates")
        economic_labels = {
            "negative_price_risk",
            "high_volatility",
            "scarcity_risk",
            "renewable_surplus",
        }
        query_economic = economic_labels.intersection(query_snapshot.labels)
        economic_match = np.asarray(
            [
                bool(query_economic.intersection(label.split("|")))
                for label in archive.labels
            ]
        )
        regime_pool = eligible & economic_match
        use_filter = regime_conditioned and int(regime_pool.sum()) >= self.minimum_regime_neighbors
        pool_mask = regime_pool if use_filter else eligible
        pool = np.flatnonzero(pool_mask)

        query_vector = _zscore(query_context)
        candidates = archive.normalized_contexts[pool]
        denominators = np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vector)
        similarities = np.divide(
            candidates @ query_vector,
            denominators,
            out=np.zeros(len(pool), dtype=float),
            where=denominators > 0,
        )
        level_shifts = query_context.mean() - archive.context_means[pool]
        query_labels = set(query_snapshot.labels)
        rows = pd.DataFrame(
            {
                "candidate_delivery_day": archive.candidate_days[pool],
                "candidate_issue_time_utc": pd.to_datetime(
                    archive.candidate_issue_ns[pool], utc=True
                ),
                "context_start_utc": pd.to_datetime(archive.context_start_ns[pool], utc=True),
                "context_end_utc": pd.to_datetime(archive.context_end_ns[pool], utc=True),
                "continuation_available_at_utc": pd.to_datetime(
                    archive.continuation_available_ns[pool], utc=True
                ),
                "similarity": similarities,
                "level_shift_eur_mwh": level_shifts,
                "candidate_regimes": archive.labels[pool],
                "regime_match": [
                    bool(query_labels.intersection(label.split("|")))
                    for label in archive.labels[pool]
                ],
                "archive_position": pool,
            }
        )
        diagnostics = rows.sort_values("similarity", ascending=False).head(self.k).copy()
        logits = (
            diagnostics.similarity.to_numpy(dtype=float)
            - float(diagnostics.similarity.max())
        ) / self.temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        selected = diagnostics.archive_position.to_numpy(dtype=int)
        curves = archive.continuations[selected, : len(horizon)]
        shifts = diagnostics.level_shift_eur_mwh.to_numpy(dtype=float)
        forecast = pd.Series(
            weights @ (curves + shifts[:, None]),
            index=horizon,
            name="retrieval_p50",
        )
        eligibility_hash = hashlib.sha256(
            archive.candidate_days[np.flatnonzero(eligible)].tobytes()
        ).hexdigest()
        forecast_payload = {
            "archive_key": archive.archive_key,
            "delivery_day": str(delivery_day),
            "query_issue": str(query_issue),
            "query_context_cutoff": str(pd.Timestamp(query_timestamps[-1], tz="UTC")),
            "regime_conditioned": regime_conditioned,
            "k": self.k,
            "temperature": self.temperature,
            "minimum_regime_neighbors": self.minimum_regime_neighbors,
            "eligibility_hash": eligibility_hash,
        }
        forecast_cache_key = hashlib.sha256(
            json.dumps(forecast_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        diagnostics.drop(columns="archive_position", inplace=True)
        diagnostics["weight"] = weights
        diagnostics["query_regimes"] = "|".join(query_snapshot.labels)
        diagnostics["rank"] = np.arange(1, len(diagnostics) + 1)
        diagnostics["archive_key"] = archive.archive_key
        diagnostics["forecast_cache_key"] = forecast_cache_key
        return RetrievalResult(
            forecast=forecast,
            diagnostics=diagnostics.reset_index(drop=True),
            query_regimes=query_snapshot.labels,
            used_regime_filter=use_filter,
            archive_key=archive.archive_key,
            forecast_cache_key=forecast_cache_key,
            eligible_candidates=int(eligible.sum()),
            runtime_seconds=time.perf_counter() - started,
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
