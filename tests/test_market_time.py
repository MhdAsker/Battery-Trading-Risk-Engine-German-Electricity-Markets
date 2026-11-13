from bess_trading.data.time import (
    delivery_day_index,
    forecast_issue_time,
    lagged_price_feature_availability_times,
)


def test_german_spring_dst_day_has_23_hours() -> None:
    index = delivery_day_index("2024-03-31")
    assert len(index) == 23
    assert index.is_monotonic_increasing


def test_german_autumn_dst_day_has_25_hours() -> None:
    index = delivery_day_index("2024-10-27")
    assert len(index) == 25
    assert index.is_unique


def test_price_features_are_available_before_daily_issue() -> None:
    horizon = delivery_day_index("2024-06-15")
    issue = forecast_issue_time("2024-06-15", hour_local=11)
    available = lagged_price_feature_availability_times(horizon)
    assert (available <= issue).all()


def test_price_feature_availability_survives_autumn_dst() -> None:
    horizon = delivery_day_index("2024-10-27")
    issue = forecast_issue_time("2024-10-27", hour_local=11)
    available = lagged_price_feature_availability_times(horizon)
    assert len(available) == 25
    assert (available <= issue).all()
