# Phase 1 implementation and acceptance record

## Scope implemented

| Requirement | Implementation | Verification |
|---|---|---|
| Daily causal forecasts | One snapshot per German delivery day at configured 11:00 local issue time | Forecast artifact stores issue and feature-availability timestamps |
| Weekly refitting | Models are reused for seven delivery days while features refresh daily | `model_refit_date` in forecast artifact |
| Interval-aware energy | SOC, revenue, throughput and degradation multiply by inferred Δt | Parametrized 15/30/60-minute tests |
| Charge/discharge exclusivity | HiGHS MILP binary mode plus shared inverter constraint | Extreme negative-price test |
| German delivery calendar | Europe/Berlin local days with 23/24/25 intervals | Spring/autumn DST tests |
| Continuous SOC | Each strategy's final SOC initializes the next complete day | Continuity test and dispatch fields |
| Terminal energy value | Configurable salvage value; no forced daily reset | `configs/phase1.yaml` |
| Multi-year data | Annual ENTSO-E acquisition command and 2020–2024 input history | Input hashes in `metrics.json` |
| Missingness bias | Missing runs, excluded dates and observed complete/excluded-day characteristics | `experiment.missingness` |
| Daily risk | Daily P&L volatility, loss probability, downside deviation, VaR/CVaR and drawdown | `daily_pnl.csv` and metrics |
| Baselines | Zero, rule, persistence, hour-of-week, linear, P50, robust and perfect foresight | Strategy metrics |
| One-command experiment | YAML-driven `bess-research run-phase1` | Synthetic end-to-end CI test |
| Dashboard correctness | Explicit strategy filter and pure tested aggregation functions | Stored-metric consistency test |
| Container demo | Docker image includes checked-in result artifacts | CI Docker build |

## Published run

- Data history: 2020–2024 ENTSO-E DE-LU day-ahead prices.
- Out-of-sample replay period: calendar year 2024.
- Complete settled delivery days: 327.
- Causality audit: passed for every forecast row.
- Test suite: 27 tests with approximately 79% package coverage.
- Reproduction: `bess-research run-phase1 --config configs/phase1.yaml`.

## Interpretation boundary

Phase 1 is still energy-only. The quantile-tail policy is explicitly a conservative
heuristic, not a scenario or CVaR optimizer. Reserve, activation, intraday, imbalance,
fees and detailed battery aging remain Phase 2/3 work and are not synthesized.
