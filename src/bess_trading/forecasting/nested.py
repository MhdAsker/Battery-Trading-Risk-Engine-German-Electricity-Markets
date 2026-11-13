"""Nested temporal model selection without test-period tuning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


def nested_rolling_select(
    x: pd.DataFrame,
    y: pd.Series,
    candidates: Iterable[dict[str, Any]],
    factory: Callable[[dict[str, Any]], Any],
    validation_days: int = 28,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Select hyperparameters on trailing validation blocks, then refit on all data."""
    cutoff = x.index.max() - pd.Timedelta(days=validation_days)
    train, validation = x.index[x.index < cutoff], x.index[x.index >= cutoff]
    if train.empty or validation.empty:
        raise ValueError("Nested validation requires non-empty train and validation windows")
    rows = []
    best_score, best_params = np.inf, None
    for params in candidates:
        model = factory(params).fit(x.loc[train], y.loc[train])
        prediction = np.asarray(model.predict(x.loc[validation]))
        score = float(np.mean(np.abs(prediction - y.loc[validation].to_numpy())))
        rows.append({**params, "validation_mae": score})
        if score < best_score:
            best_score, best_params = score, params
    assert best_params is not None
    return factory(best_params).fit(x, y), best_params, pd.DataFrame(rows)
