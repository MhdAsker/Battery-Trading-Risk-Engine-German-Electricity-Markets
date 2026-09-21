import pandas as pd

from bess_trading.forecasting.validation import walk_forward_splits


def test_walk_forward_has_no_overlap() -> None:
    idx = pd.date_range("2023-01-01", periods=500 * 24, freq="h", tz="UTC")
    train, test = next(walk_forward_splits(idx, train_days=365, test_days=7))
    assert train.max() < test.min()
    assert len(test) == 7 * 24
