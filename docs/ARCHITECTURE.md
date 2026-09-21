# Technical architecture

The repository separates immutable market inputs, causal transformations, forecasts,
decisions, and ex-post settlement. That boundary is the central research control.

```text
ENTSO-E / local CSV
        |
        v
data/raw (immutable) -> validation -> data/processed (UTC, joined)
                                            |
                                            v
                            leakage-safe feature snapshots
                                            |
                         walk-forward forecast models
                               |              |
                            P50 point      P10/P50/P90
                               |              |
                         deterministic     robust dispatch
                               \              /
                                realized-price settlement
                                         |
                              metrics + dashboard
```

## Module boundaries

- `data`: API clients, XML/CSV parsing, and data contracts. API output is not modified.
- `features`: only fields available at the documented forecast issue time.
- `forecasting`: models and rolling temporal splits; no random train/test split.
- `battery`: independent physical schedule validation.
- `optimization`: transparent convex programs. Forecasts determine schedules only.
- `backtesting`: realized-price settlement and strategy orchestration.
- `metrics`: forecast accuracy, calibration, revenue, risk, and regret inputs.
- `dashboard`: reads experiment artifacts and never trains models implicitly.

## Experiment artifact contract

An experiment should save `forecasts.parquet`, `dispatch.parquet`, `metrics.json`, and
the effective configuration. Every timestamp is UTC. Forecast rows include
`forecast_issue_time_utc` so causal availability can be audited.

Multi-market optimization is deliberately not activated until reserve-capacity and
activation data are ingested and their product-specific constraints are represented.
Energy-only results must not be labelled multi-market results.

