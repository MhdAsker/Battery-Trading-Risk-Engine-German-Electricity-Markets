import numpy as np
import pandas as pd

from bess_trading.features.build import build_features


def test_price_features_do_not_use_recent_target() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    base = pd.DataFrame({"price_eur_mwh": np.arange(200.0)}, index=idx)
    changed = base.copy()
    changed.iloc[-24:, 0] = 99999
    assert build_features(base).iloc[-1].equals(build_features(changed).iloc[-1])


def test_residual_load_and_renewable_share_features() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "price_eur_mwh": np.arange(200.0),
            "load_forecast_mw": 100.0,
            "wind_forecast_mw": 30.0,
            "solar_forecast_mw": 20.0,
        },
        index=idx,
    )
    features = build_features(frame)
    assert features.residual_load_forecast_mw.iloc[-1] == 50.0
    assert features.renewable_share_forecast.iloc[-1] == 0.5
