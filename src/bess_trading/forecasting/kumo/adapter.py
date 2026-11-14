"""Leakage-safe adapter around NVIDIA Kumo Forecast's official DataFrame API."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from bess_trading.data.time import (
    delivery_day_index,
    forecast_issue_time,
    price_publication_times,
)

OFFICIAL_REPOSITORY = "https://github.com/NVIDIA/Kumo-TS"
AUDITED_COMMIT = "ad459475d1302fc8365e7117e39837a8ecb6a7e3"
DEFAULT_CHECKPOINT_REPOSITORY = "nvidia/Kumo-Forecast"


@dataclass(frozen=True)
class KumoRuntimeConfig:
    """Project-owned subset of the verified Kumo Forecast inference configuration."""

    seq_len: int = 512
    model_horizon: int = 72
    checkpoint: str = "run8_best_model_cr.pt"
    standardizer: str = "standardizer.pkl"
    model_name: str = "AutonLab/MOMENT-1-large"
    use_cross_channel: bool = True
    device: str | None = None
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 0
    alpha: float = 0.2
    k: int = 16
    temperature: float = 0.05
    context_stride: int = 24
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.seq_len < 24 or self.model_horizon <= 0:
            raise ValueError("Kumo context and model horizon must be positive")
        if not 0 <= self.alpha <= 1 or self.k < 1 or self.temperature <= 0:
            raise ValueError("Invalid DARR configuration")


@dataclass(frozen=True)
class KumoForecastResult:
    """A point forecast with temporal, model and resource metadata."""

    forecast: pd.Series
    issue_time_utc: pd.Timestamp
    delivery_day_local: str
    mode: str
    input_cutoff_utc: pd.Timestamp
    archive_cutoff_utc: pd.Timestamp | None
    forecast_steps_from_cutoff: int
    inference_seconds: float
    metadata: dict[str, Any]
    cache_key: str


class OfficialKumoBackend:
    """Lazy importer for the official optional Kumo package."""

    def __call__(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
        context_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        try:
            module = importlib.import_module("sdk.forecasting")
        except ImportError as exc:
            raise ImportError(
                "Kumo Forecast is optional. Clone NVIDIA/Kumo-TS at the audited revision, "
                "install Kumo-Forecast in a compatible environment, and ensure sdk.forecasting "
                "is importable. No checkpoint is vendored by this project."
            ) from exc
        official_config = module.ForecastingConfig(**config)
        return module.perform_forecasting(
            df=frame,
            config=official_config,
            context_df=context_frame,
        )


def _series_hash(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _regular_complete(series: pd.Series, name: str) -> None:
    if series.empty or series.index.tz is None:
        raise ValueError(f"{name} must be a non-empty timezone-aware series")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be unique and increasing")
    if series.isna().any():
        raise ValueError(
            f"{name} contains missing values; standard Kumo/DARR zero-fills missing values, "
            "which is forbidden for this price experiment"
        )
    expected = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    if not series.index.equals(expected):
        raise ValueError(f"{name} must be a contiguous hourly UTC series")


def latest_complete_segment(series: pd.Series, minimum_rows: int) -> pd.Series:
    """Return the latest contiguous complete hourly segment without imputing prices."""
    if series.index.tz is None:
        raise ValueError("Price timestamps must be timezone-aware")
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ValueError("No observed prices are available")
    breaks = clean.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    segments = [group for _, group in clean.groupby(breaks)]
    eligible = [segment for segment in segments if len(segment) >= minimum_rows]
    if not eligible:
        raise ValueError(f"No complete hourly segment contains at least {minimum_rows} rows")
    return eligible[-1]


class KumoForecastAdapter:
    """Map project price vintages and German delivery days to official Kumo inference."""

    def __init__(
        self,
        config: KumoRuntimeConfig | None = None,
        *,
        backend: Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame | None], pd.DataFrame]
        | None = None,
        cache_dir: str | Path = "artifacts/kumo_cache",
    ) -> None:
        self.config = config or KumoRuntimeConfig()
        self.backend = backend or OfficialKumoBackend()
        self.cache_dir = Path(cache_dir)

    def _known_prices(
        self, prices: pd.Series, issue_time: pd.Timestamp, required_rows: int
    ) -> pd.Series:
        publication = pd.Series(price_publication_times(prices.index), index=prices.index)
        known = prices[(publication <= issue_time).to_numpy()]
        return latest_complete_segment(known, required_rows)

    def _sdk_config(self, forecast_horizon: int) -> dict[str, Any]:
        return {
            "timestamp_column": "timestamp",
            "target_column": "target",
            "standardizer_pkl": self.config.standardizer,
            "ckpt": self.config.checkpoint,
            "seq_len": self.config.seq_len,
            "forecast_horizon": forecast_horizon,
            "model_horizon": self.config.model_horizon,
            "alpha": self.config.alpha,
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "context_stride": self.config.context_stride,
            "seed": self.config.seed,
            "k": self.config.k,
            "temperature": self.config.temperature,
            "device": self.config.device,
            "local_files_only": self.config.local_files_only,
            "use_cross_channel": self.config.use_cross_channel,
            "return_all_channels": False,
            "handle_missingness": False,
        }

    @staticmethod
    def _sdk_frame(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": series.index.tz_convert("UTC").tz_localize(None),
                "target": series.to_numpy(dtype=float),
            }
        )

    def forecast_delivery_day(
        self,
        prices: pd.Series,
        delivery_day: date | str,
        *,
        issue_hour_local: int = 11,
        native_darr: bool = False,
        use_cache: bool = True,
    ) -> KumoForecastResult:
        """Forecast the full physical delivery day from a publication-safe vintage."""
        issue = forecast_issue_time(delivery_day, issue_hour_local)
        horizon = delivery_day_index(delivery_day, "1h")
        known = self._known_prices(prices, issue, self.config.seq_len)
        input_series = known.iloc[-self.config.seq_len :]
        input_cutoff = input_series.index.max()
        if input_cutoff >= horizon.min():
            raise ValueError("Kumo input overlaps the delivery horizon")
        full_horizon = pd.date_range(
            input_cutoff + pd.Timedelta(hours=1), horizon.max(), freq="h", tz="UTC"
        )
        if not horizon.isin(full_horizon).all():
            raise ValueError("Forecast path does not cover the complete delivery day")
        if len(full_horizon) > 512:
            raise ValueError("Required Kumo path exceeds the verified 512-step SDK limit")

        context_series: pd.Series | None = None
        context_frame: pd.DataFrame | None = None
        if native_darr:
            required = self.config.seq_len + len(full_horizon)
            context_series = self._known_prices(prices, issue, required)
            context_frame = self._sdk_frame(context_series)

        key_payload = {
            "config": asdict(self.config),
            "delivery_day": str(delivery_day),
            "issue": str(issue),
            "mode": "native_darr" if native_darr else "direct",
            "input_hash": _series_hash(input_series),
            "context_hash": _series_hash(context_series) if context_series is not None else None,
            "audited_kumo_commit": AUDITED_COMMIT,
        }
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        metadata_path = self.cache_dir / f"{cache_key}.json"
        if use_cache and cache_path.exists() and metadata_path.exists():
            cached = pd.read_parquet(cache_path).iloc[:, 0]
            cached.name = "kumo_p50"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return KumoForecastResult(
                cached,
                issue,
                str(delivery_day),
                key_payload["mode"],
                input_cutoff,
                context_series.index.max() if context_series is not None else None,
                len(full_horizon),
                float(metadata["inference_seconds"]),
                metadata,
                cache_key,
            )

        sdk_config = self._sdk_config(len(full_horizon))
        started = time.perf_counter()
        raw = self.backend(self._sdk_frame(input_series), sdk_config, context_frame)
        elapsed = time.perf_counter() - started
        if "target_forecast" not in raw or len(raw) != len(full_horizon):
            raise ValueError("Official Kumo backend returned an unexpected forecast shape")
        values = pd.Series(
            raw.target_forecast.to_numpy(dtype=float), index=full_horizon, name="kumo_p50"
        )
        forecast = values.reindex(horizon)
        if forecast.isna().any():
            raise ValueError("Kumo forecast is incomplete for the German delivery day")
        metadata = {
            **key_payload,
            "official_repository": OFFICIAL_REPOSITORY,
            "checkpoint_repository": DEFAULT_CHECKPOINT_REPOSITORY,
            "inference_seconds": elapsed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "forecast_steps_from_cutoff": len(full_horizon),
            "delivery_hours": len(horizon),
            "input_rows": len(input_series),
            "archive_rows": len(context_series) if context_series is not None else 0,
        }
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            forecast.to_frame().to_parquet(cache_path)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return KumoForecastResult(
            forecast,
            issue,
            str(delivery_day),
            key_payload["mode"],
            input_cutoff,
            context_series.index.max() if context_series is not None else None,
            len(full_horizon),
            elapsed,
            metadata,
            cache_key,
        )
