# Phase 2: probabilistic value research

Phase 2 asks a narrower and more defensible question than “which model has the lowest MAE?”:
does a calibrated probabilistic price forecast improve realized battery revenue after risk and
degradation are considered?

## Experimental protocol

The default experiment uses July–December 2023 only for conformal calibration and residual
scenario construction. Calendar year 2024 remains untouched until final evaluation. Forecasts
are issued at 11:00 Europe/Berlin, and Phase 1 publication-time checks remain active. Model
selection utilities use trailing rolling validation and never accept test observations.

Run the complete configured experiment with:

```powershell
bess-research run-phase2 --config configs/phase2.yaml
```

Download archived ENTSO-E day-ahead fundamentals with:

```powershell
bess-research fetch-fundamentals --start 2023-01-01 --end 2025-01-01
```

`build_features` accepts only explicitly named `*_forecast_mw` fundamentals and creates
residual-load and renewable-share features. It deliberately rejects the convention of silently
using realized load or generation in a day-ahead model.

## Methods implemented

- Regularized linear, LightGBM, XGBoost and quantile gradient-boosting estimators.
- Trailing nested temporal model selection.
- Split conformal P10–P90 calibration fitted before the test boundary.
- WIS, empirical-ensemble CRPS, PIT, conditional coverage, interval width and counts.
- Circular residual-block scenarios that preserve multi-hour dependence.
- Explicit non-anticipative stochastic mixed-integer CVaR optimization.
- Revenue/CVaR efficient frontiers and duration, efficiency, degradation, cycle-limit and
  risk-aversion sensitivity APIs.
- Piecewise-throughput and SOC rainflow-style degradation proxies.
- Paired circular block-bootstrap confidence intervals, paired effect sizes and probability of
  improvement.
- Economic forecast-error attribution for missed spikes, false spreads, reversed rankings,
  incorrect charge timing and peak underestimation.

## Artifact contract

`results/phase2_2024` contains calibrated forecasts, coherent scenarios, settled dispatch,
daily P&L, regime labels, error attribution, an efficient frontier and `metrics.json`. The latter
records test isolation, unconditional and regime coverage, scenario correlation/tail checks,
confidence intervals, effect sizes, and an explicit positive or negative research answer.

The experiment is historical energy-only research. It excludes intraday re-trading, imbalance
settlement, ancillary-service stacking, bid acceptance uncertainty, taxes and grid constraints;
its revenue is not a live-trading claim.
