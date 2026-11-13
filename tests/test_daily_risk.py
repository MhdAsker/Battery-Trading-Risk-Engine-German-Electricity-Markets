import pandas as pd

from bess_trading.metrics.risk import daily_pnl, daily_risk_metrics, paired_block_bootstrap_ci


def test_daily_risk_aggregates_before_tail_statistics() -> None:
    frame = pd.DataFrame(
        {
            "delivery_day_local": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "strategy": ["a", "a", "b", "b"] * 2,
            "net_revenue_eur": [5, 5, 2, 2, -4, -6, 1, 1],
        }
    )
    pnl = daily_pnl(frame)
    assert pnl.loc["2024-01-01", "a"] == 10
    assert pnl.loc["2024-01-02", "a"] == -10
    risk = daily_risk_metrics(pnl.a)
    assert risk["loss_probability"] == 0.5
    assert risk["daily_mean_eur"] == 0.0


def test_paired_block_bootstrap_is_reproducible() -> None:
    candidate = pd.Series(range(30), dtype=float)
    reference = candidate - 2
    first = paired_block_bootstrap_ci(candidate, reference, samples=100, seed=7)
    second = paired_block_bootstrap_ci(candidate, reference, samples=100, seed=7)
    assert first == second
    assert first["mean_daily_difference_eur"] == 2.0
