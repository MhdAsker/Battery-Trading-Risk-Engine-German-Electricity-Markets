"""Joint activation-volume and activation-price scenario generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def joint_activation_scenarios(
    history: pd.DataFrame,
    horizon: pd.DatetimeIndex,
    *,
    scenarios: int = 100,
    block_intervals: int = 16,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Sample common historical blocks so volumes and prices remain dependent."""
    required = {
        "afrr_up_activation_fraction",
        "afrr_down_activation_fraction",
        "mfrr_up_activation_fraction",
        "mfrr_down_activation_fraction",
        "afrr_up_activation_price_eur_mwh",
        "afrr_down_activation_price_eur_mwh",
        "mfrr_up_activation_price_eur_mwh",
        "mfrr_down_activation_price_eur_mwh",
    }
    missing = required - set(history)
    if missing:
        raise ValueError(f"Missing activation scenario history: {sorted(missing)}")
    clean = history[list(sorted(required))].dropna()
    if len(clean) < block_intervals:
        raise ValueError("Activation history is shorter than one sampling block")
    rng = np.random.default_rng(seed)
    output = {
        column: np.empty((len(horizon), scenarios), dtype=float) for column in clean.columns
    }
    values = clean.to_numpy()
    for scenario in range(scenarios):
        rows: list[np.ndarray] = []
        while len(rows) < len(horizon):
            start = int(rng.integers(0, len(values)))
            rows.extend(values[(start + np.arange(block_intervals)) % len(values)])
        draw = np.asarray(rows[: len(horizon)])
        for position, column in enumerate(clean.columns):
            output[column][:, scenario] = draw[:, position]
    return {
        column: pd.DataFrame(
            values,
            index=horizon,
            columns=[f"scenario_{number:03d}" for number in range(scenarios)],
        )
        for column, values in output.items()
    }
