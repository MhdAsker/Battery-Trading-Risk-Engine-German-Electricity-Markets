# November 2024 energy-only benchmark

This checked-in experiment evaluates forecast-driven day-ahead dispatch for a
simulated 1 MW / 2 MWh battery over 27 complete UTC delivery days between 1 and
30 November 2024. Three source-incomplete days are excluded from dispatch and revenue.

## Design

- Real ENTSO-E German/Luxembourg day-ahead clearing prices.
- Trailing 365-day training window and seven-day model refits.
- Calendar, 24/48/168-hour lag, and causal rolling price features.
- P10/P50/P90 histogram gradient-boosting quantile models with seed 42.
- 10–90% SOC, 95% one-way efficiencies, €5/MWh throughput degradation, 1.5 daily
  equivalent-cycle limit, and 50% initial/terminal SOC.
- Every schedule is frozen from forecast information and settled at observed prices.

## Main results

| Strategy | Net revenue (€) | Gross revenue (€) | Throughput (MWh) | Equivalent cycles | Regret (€) |
|---|---:|---:|---:|---:|---:|
| Perfect foresight | 3,407.38 | 4,048.94 | 128.31 | 32.08 | 0.00 |
| Persistence | 2,316.74 | 2,975.71 | 131.79 | 32.95 | 1,090.65 |
| Hour-of-week | 785.16 | 1,547.41 | 152.45 | 38.11 | 2,622.22 |
| ML P50 | 2,209.59 | 2,812.69 | 120.62 | 30.15 | 1,197.79 |
| Quantile robust | **2,340.13** | 2,784.69 | **88.91** | **22.23** | **1,067.25** |

ML P50 achieved the lowest forecast MAE (€29.06/MWh) but did not produce the highest
realistic strategy revenue. Persistence earned €2,316.74, while robust quantile
dispatch earned €2,340.13 with materially lower cycling and downside metrics. This
supports the project's central premise that price accuracy and operational value are
related but not interchangeable objectives.

P10–P90 coverage was only 53.4%, well below its nominal 80%, indicating that the
interval was too narrow or otherwise miscalibrated. The robust-policy improvement is
therefore promising but not proof that probabilistic forecasting generalizes.

## Files

- `metrics.json`: complete forecast, strategy, risk, configuration, and provenance data.
- `strategy_metrics.csv`: compact strategy comparison.
- `forecasts.parquet`: timestamped P10/P50/P90 and baseline forecasts.
- `dispatch.parquet`: hourly schedules, SOC, realized-price settlement, and strategy.

The evaluation window is short and selected for the first reproducible milestone.
Results must not be annualized or interpreted as commercial returns.

