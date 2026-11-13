"""Daily P&L risk and paired block-bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_pnl(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Aggregate settled interval revenues to strategy-level local delivery days."""
    required = {"strategy", "delivery_day_local", "net_revenue_eur"}
    if not required.issubset(dispatch):
        raise ValueError(f"Dispatch requires {sorted(required)}")
    return (
        dispatch.groupby(["delivery_day_local", "strategy"], observed=True)
        .net_revenue_eur.sum()
        .unstack("strategy")
        .sort_index()
    )


def daily_risk_metrics(values: pd.Series, alpha: float = 0.95) -> dict[str, float]:
    """Calculate downside statistics from daily strategy P&L."""
    losses = -values
    var = float(losses.quantile(alpha))
    tail = losses[losses >= var]
    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax()
    negative = values[values < 0]
    return {
        "daily_mean_eur": float(values.mean()),
        "daily_volatility_eur": float(values.std(ddof=1)),
        "loss_probability": float((values < 0).mean()),
        "downside_deviation_eur": float(np.sqrt(np.mean(np.minimum(values, 0) ** 2))),
        "daily_var_95_loss_eur": var,
        "daily_cvar_95_loss_eur": float(tail.mean()),
        "maximum_drawdown_eur": float(drawdown.min()),
        "mean_loss_eur": float(-negative.mean()) if len(negative) else 0.0,
    }


def paired_block_bootstrap_ci(
    candidate: pd.Series,
    reference: pd.Series,
    *,
    block_days: int = 7,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Confidence interval for mean daily revenue difference using circular blocks."""
    paired = pd.concat([candidate, reference], axis=1).dropna()
    differences = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy(dtype=float)
    n = len(differences)
    if n < block_days * 2:
        raise ValueError("At least two bootstrap blocks are required")
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    draws: list[float] = []
    blocks_needed = int(np.ceil(n / block_days))
    for _ in range(samples):
        selected: list[int] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            selected.extend((start + np.arange(block_days)) % n)
        draws.append(float(differences[np.asarray(selected[:n])].mean()))
    tail = (1 - confidence) / 2
    standard_deviation = float(differences.std(ddof=1))
    return {
        "mean_daily_difference_eur": float(differences.mean()),
        "ci_lower_eur": float(np.quantile(draws, tail)),
        "ci_upper_eur": float(np.quantile(draws, 1 - tail)),
        "confidence": confidence,
        "block_days": block_days,
        "bootstrap_samples": samples,
        "paired_effect_size_dz": (
            float(differences.mean() / standard_deviation) if standard_deviation > 0 else 0.0
        ),
        "probability_of_improvement": float(np.mean(np.asarray(draws) > 0)),
    }
