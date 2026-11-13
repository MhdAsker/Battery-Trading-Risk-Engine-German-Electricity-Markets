import json
from pathlib import Path

import pandas as pd
import pytest

from bess_trading.visualization.dashboard_data import (
    cumulative_revenue,
    select_strategy,
    strategy_revenue,
)


def test_dashboard_revenue_matches_stored_benchmark_metrics() -> None:
    root = Path("results/energy_benchmark_2024")
    data = pd.read_parquet(root / "dispatch.parquet")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    totals = strategy_revenue(data)
    for strategy, value in totals.items():
        assert value == pytest.approx(metrics["strategy_metrics"][strategy]["net_revenue_eur"])


def test_cumulative_revenue_requires_one_strategy() -> None:
    data = pd.DataFrame(
        {"strategy": ["a", "b"], "net_revenue_eur": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
    )
    with pytest.raises(ValueError, match="one strategy"):
        cumulative_revenue(data)
    assert cumulative_revenue(select_strategy(data, "a")).iloc[-1] == 1.0
