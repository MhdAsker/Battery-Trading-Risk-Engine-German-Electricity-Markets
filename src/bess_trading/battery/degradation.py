"""Piecewise throughput and rainflow-style SOC-cycle degradation proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def piecewise_degradation_cost(
    throughput_mwh: pd.Series, low_rate: float, high_rate: float, breakpoint_mwh: float
) -> float:
    total = float(throughput_mwh.sum())
    return low_rate * min(total, breakpoint_mwh) + high_rate * max(0.0, total - breakpoint_mwh)


def rainflow_equivalent_cycles(soc_mwh: pd.Series, energy_mwh: float) -> float:
    """Approximate full-cycle equivalents from SOC turning-point ranges."""
    values = soc_mwh.to_numpy(dtype=float)
    if len(values) < 3:
        return 0.0
    direction = np.sign(np.diff(values))
    turns = np.r_[values[0], values[1:-1][direction[:-1] * direction[1:] < 0], values[-1]]
    return float(np.abs(np.diff(turns)).sum() / (2 * energy_mwh))
