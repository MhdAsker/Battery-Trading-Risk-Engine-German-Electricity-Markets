from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from bess_trading.forecasting.kumo.adapter import KumoForecastAdapter, KumoRuntimeConfig


@dataclass
class FakeKumoBackend:
    calls: list[tuple[pd.DataFrame, dict[str, object], pd.DataFrame | None]] = field(
        default_factory=list
    )

    def __call__(
        self,
        frame: pd.DataFrame,
        config: dict[str, object],
        context: pd.DataFrame | None,
    ) -> pd.DataFrame:
        self.calls.append(
            (frame.copy(), config.copy(), None if context is None else context.copy())
        )
        horizon = int(config["forecast_horizon"])
        return pd.DataFrame({"target_forecast": np.arange(horizon, dtype=float)})


def price_history() -> pd.Series:
    index = pd.date_range("2021-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    values = 50 + 20 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    return pd.Series(values, index=index, name="price_eur_mwh")


@pytest.mark.parametrize(
    ("day", "hours"), [("2024-03-31", 23), ("2024-10-27", 25)]
)
def test_kumo_adapter_extracts_physical_dst_delivery_day(day: str, hours: int) -> None:
    backend = FakeKumoBackend()
    adapter = KumoForecastAdapter(
        KumoRuntimeConfig(seq_len=168), backend=backend, cache_dir="unused"
    )
    result = adapter.forecast_delivery_day(price_history(), day, use_cache=False)
    assert len(result.forecast) == hours
    assert result.forecast.index.is_unique
    assert result.forecast.index.tz is not None
    assert backend.calls[0][1]["forecast_horizon"] == result.forecast_steps_from_cutoff


def test_kumo_input_is_invariant_to_future_price_injection() -> None:
    prices = price_history()
    changed = prices.copy()
    changed.loc[changed.index >= "2024-06-15"] = 999999.0
    first_backend, second_backend = FakeKumoBackend(), FakeKumoBackend()
    config = KumoRuntimeConfig(seq_len=168)
    first = KumoForecastAdapter(config, backend=first_backend).forecast_delivery_day(
        prices, "2024-06-15", use_cache=False
    )
    second = KumoForecastAdapter(config, backend=second_backend).forecast_delivery_day(
        changed, "2024-06-15", use_cache=False
    )
    pd.testing.assert_series_equal(first.forecast, second.forecast)
    pd.testing.assert_frame_equal(first_backend.calls[0][0], second_backend.calls[0][0])
    assert first.cache_key == second.cache_key


def test_native_darr_archive_is_cut_off_before_issue() -> None:
    backend = FakeKumoBackend()
    adapter = KumoForecastAdapter(
        KumoRuntimeConfig(seq_len=168, k=4), backend=backend
    )
    result = adapter.forecast_delivery_day(
        price_history(), "2024-06-15", native_darr=True, use_cache=False
    )
    context = backend.calls[0][2]
    assert context is not None
    assert len(context) >= 168 + result.forecast_steps_from_cutoff
    assert result.archive_cutoff_utc is not None
    assert result.archive_cutoff_utc < result.forecast.index.min()


def test_kumo_uses_older_complete_context_instead_of_zero_filling_gap() -> None:
    prices = price_history().loc["2024-01-01":].copy()
    prices.loc["2024-06-13 12:00:00+00:00"] = np.nan
    adapter = KumoForecastAdapter(KumoRuntimeConfig(seq_len=168), backend=FakeKumoBackend())
    result = adapter.forecast_delivery_day(prices, "2024-06-15", use_cache=False)
    assert result.forecast_steps_from_cutoff > len(result.forecast)
    assert not adapter.backend.calls[0][0].target.isna().any()
