# Phase 1 corrected 2024 benchmark

This artifact bundle is the causal and physically strict Phase 1 benchmark for a
simulated 1 MW / 2 MWh battery. The data history spans 2020–2024 and the 2024
out-of-sample replay contains 327 complete German delivery days.

## Corrections relative to the legacy pilot

- Daily 11:00 Europe/Berlin forecast snapshots with weekly model refits.
- Audited feature publication times; the causality audit passes for every row.
- Local-day price lags that remain valid across DST.
- German 23/24/25-hour delivery days.
- Interval-aware SOC, revenue, throughput and degradation equations.
- Mixed-integer charge/discharge exclusivity and shared inverter limits.
- Continuous interday SOC with terminal energy value rather than daily resets.
- Daily P&L risk measures and paired seven-day circular block-bootstrap intervals.
- Zero-trade, rule-based, persistence, hour-of-week, linear, P50, robust and
  perfect-foresight strategies.

## Forecast results

| Forecast | MAE (€/MWh) | RMSE (€/MWh) |
|---|---:|---:|
| Persistence | 27.71 | 44.50 |
| Hour of week | 30.88 | 47.73 |
| Regularized linear | 23.91 | **37.78** |
| Quantile GBM P50 | **22.75** | 39.59 |

P10–P90 coverage is 70.8% with mean width €56.66/MWh, below the nominal 80% target.

## Strategy results

| Strategy | Net revenue (€) | Revenue/MW-day (€) | Daily CVaR loss (€) | Regret (€) |
|---|---:|---:|---:|---:|
| Perfect foresight | 50,399 | 154.1 | 0.1 | 0 |
| Linear | **42,207** | **129.1** | 11.8 | 8,192 |
| ML P50 | 41,661 | 127.4 | 23.3 | 8,738 |
| Persistence | 39,404 | 120.5 | 33.0 | 10,995 |
| Quantile robust | 38,686 | 118.3 | **4.5** | 11,713 |
| Hour of week | 36,542 | 111.8 | 37.9 | 13,857 |
| Rule based | 25,074 | 76.7 | 92.9 | 25,325 |
| Zero trade | 0 | 0 | 0 | 50,399 |

The linear-versus-P50 revenue difference is not statistically established because its
paired bootstrap confidence interval includes zero. Robust dispatch reduces downside
risk and throughput but does not maximize revenue.

## Data quality

The source contains 51 missing hourly intervals. Settlement values are never imputed;
affected local delivery days are excluded. `metrics.json` records the excluded dates,
longest gap, missing rate and observed price/range characteristics for complete versus
excluded days.

## Files

- `forecasts.parquet`: forecasts, issue times, feature availability and refit dates.
- `dispatch.parquet`: interval decisions, SOC, strategies and realized settlement.
- `daily_pnl.csv`: strategy P&L at the primary risk-accounting frequency.
- `metrics.json`: forecast, economic, daily-risk, bootstrap and audit results.
- `effective_config.yaml`: exact configuration used for the run.

These are historical simulations, not commercial or expected annual returns.
