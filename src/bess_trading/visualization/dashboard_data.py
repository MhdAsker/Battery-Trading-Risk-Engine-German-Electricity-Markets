"""Pure dashboard transformations kept testable outside Streamlit."""

from __future__ import annotations

import pandas as pd


def strategies(data: pd.DataFrame) -> list[str]:
    """Return stable strategy choices from an artifact."""
    return sorted(data.strategy.dropna().unique()) if "strategy" in data else ["default"]


def select_strategy(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Filter an artifact before calculating any path-dependent metric."""
    if "strategy" not in data:
        return data.copy()
    result = data[data.strategy == strategy].copy()
    if result.empty:
        raise ValueError(f"Unknown strategy: {strategy}")
    return result.sort_index()


def strategy_revenue(data: pd.DataFrame) -> pd.Series:
    """Aggregate total net revenue exactly once per strategy."""
    if not {"strategy", "net_revenue_eur"}.issubset(data):
        raise ValueError("Artifact requires strategy and net_revenue_eur")
    return data.groupby("strategy").net_revenue_eur.sum().sort_values(ascending=False)


def cumulative_revenue(data: pd.DataFrame) -> pd.Series:
    """Calculate one strategy's chronological cumulative P&L."""
    if "strategy" in data and data.strategy.nunique() > 1:
        raise ValueError("Filter to one strategy before calculating cumulative revenue")
    return data.sort_index().net_revenue_eur.cumsum()
