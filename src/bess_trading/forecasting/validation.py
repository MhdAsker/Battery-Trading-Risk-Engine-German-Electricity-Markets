"""Walk-forward split generation."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    index: pd.DatetimeIndex, train_days: int = 365, test_days: int = 7, step_days: int = 7
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield fixed-width rolling train/test indices without temporal overlap."""
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted")
    cursor = index.min() + pd.Timedelta(days=train_days)
    end = index.max() + pd.Timedelta(microseconds=1)
    while cursor + pd.Timedelta(days=test_days) <= end:
        train = index[(index >= cursor - pd.Timedelta(days=train_days)) & (index < cursor)]
        test = index[(index >= cursor) & (index < cursor + pd.Timedelta(days=test_days))]
        if len(train) and len(test):
            yield train, test
        cursor += pd.Timedelta(days=step_days)
