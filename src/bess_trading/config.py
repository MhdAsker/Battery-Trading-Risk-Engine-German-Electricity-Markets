"""Typed configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and economic battery parameters."""

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    min_soc_fraction: float = 0.10
    max_soc_fraction: float = 0.90
    initial_soc_fraction: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    degradation_cost_eur_per_mwh: float = 5.0
    max_equivalent_cycles_per_day: float = 1.5

    def __post_init__(self) -> None:
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ValueError("Power and energy must be positive")
        if not 0 <= self.min_soc_fraction < self.max_soc_fraction <= 1:
            raise ValueError("SOC bounds must satisfy 0 <= min < max <= 1")
        if not self.min_soc_fraction <= self.initial_soc_fraction <= self.max_soc_fraction:
            raise ValueError("Initial SOC must lie inside SOC bounds")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Efficiencies must be in (0, 1]")
        if self.max_equivalent_cycles_per_day <= 0:
            raise ValueError("Cycle limit must be positive")

    @property
    def initial_soc_mwh(self) -> float:
        return self.initial_soc_fraction * self.energy_mwh


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without evaluating arbitrary Python objects."""
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return value
