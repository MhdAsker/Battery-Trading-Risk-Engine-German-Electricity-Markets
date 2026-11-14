"""Development-only residual calibration for Kumo point forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegimeResidualCalibrator:
    """Empirical residual intervals with split-conformal and sparse-regime fallback."""

    alpha: float = 0.2
    minimum_regime_samples: int = 200
    parameters_: dict[str, tuple[float, float, float]] = field(default_factory=dict, init=False)

    def fit(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        regimes: pd.Series,
        *,
        distribution_end: pd.Timestamp,
        test_start: pd.Timestamp,
    ) -> RegimeResidualCalibrator:
        """Fit residual quantiles before distribution_end and conformalize afterward."""
        frame = pd.concat(
            [actual.rename("actual"), predicted.rename("predicted"), regimes.rename("regime")],
            axis=1,
        ).dropna()
        if frame.index.tz is None or frame.index.max() >= test_start:
            raise ValueError("Calibration data must be timezone-aware and strictly pre-test")
        distribution = frame[frame.index < distribution_end]
        conformal = frame[frame.index >= distribution_end]
        if len(distribution) < 100 or len(conformal) < 20:
            raise ValueError(
                "Residual calibration requires separate distribution/conformal samples"
            )

        global_q = self._quantiles(distribution.actual - distribution.predicted)
        global_correction = self._correction(conformal, global_q)
        self.parameters_["global"] = (*global_q, global_correction)
        for label, group in distribution.groupby("regime"):
            conformal_group = conformal[conformal.regime == label]
            if len(group) < self.minimum_regime_samples or len(conformal_group) < 20:
                continue
            quantiles = self._quantiles(group.actual - group.predicted)
            self.parameters_[str(label)] = (
                *quantiles,
                self._correction(conformal_group, quantiles),
            )
        return self

    def _quantiles(self, residuals: pd.Series) -> tuple[float, float]:
        return float(residuals.quantile(self.alpha / 2)), float(
            residuals.quantile(1 - self.alpha / 2)
        )

    def _correction(self, frame: pd.DataFrame, quantiles: tuple[float, float]) -> float:
        lower = frame.predicted + quantiles[0]
        upper = frame.predicted + quantiles[1]
        scores = np.maximum(lower - frame.actual, frame.actual - upper)
        level = min(1.0, np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
        return max(0.0, float(np.quantile(scores, level, method="higher")))

    def transform(self, predicted: pd.Series, regimes: pd.Series) -> pd.DataFrame:
        if "global" not in self.parameters_:
            raise RuntimeError("Residual calibrator is not fitted")
        regimes = regimes.reindex(predicted.index)
        rows = []
        for timestamp, value in predicted.items():
            label = str(regimes.loc[timestamp])
            source = label if label in self.parameters_ else "global"
            lower_residual, upper_residual, correction = self.parameters_[source]
            rows.append(
                {
                    "p10": value + lower_residual - correction,
                    "p50": value,
                    "p90": value + upper_residual + correction,
                    "calibration_source": source,
                }
            )
        return pd.DataFrame(rows, index=predicted.index)


def regime_residual_scenarios(
    point_forecast: pd.Series,
    residual_history: pd.Series,
    historical_regimes: pd.Series,
    query_regime: str,
    *,
    scenarios: int = 100,
    block_hours: int = 24,
    minimum_regime_hours: int = 240,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Generate coherent residual-block paths with documented global fallback."""
    aligned = pd.concat(
        [residual_history.rename("residual"), historical_regimes.rename("regime")], axis=1
    ).dropna()
    selected = aligned.loc[aligned.regime == query_regime, "residual"]
    source = query_regime
    if len(selected) < max(minimum_regime_hours, block_hours):
        selected = aligned.residual
        source = "global"
    if len(selected) < block_hours:
        raise ValueError("Residual history is shorter than one scenario block")
    values = selected.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    output = np.empty((len(point_forecast), scenarios), dtype=float)
    for scenario in range(scenarios):
        draw: list[float] = []
        while len(draw) < len(point_forecast):
            start = int(rng.integers(0, len(values)))
            draw.extend(values[(start + np.arange(block_hours)) % len(values)])
        output[:, scenario] = point_forecast.to_numpy() + np.asarray(draw[: len(point_forecast)])
    return (
        pd.DataFrame(
            output,
            index=point_forecast.index,
            columns=[f"scenario_{number:03d}" for number in range(scenarios)],
        ),
        source,
    )
