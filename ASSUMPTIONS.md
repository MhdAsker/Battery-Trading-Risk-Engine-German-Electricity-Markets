# Assumptions and research boundaries

## Market

- The simulated asset is a price taker in the German/Luxembourg day-ahead market.
- The Phase 1 optimizer is interval-aware for 15-, 30-, and 60-minute products. The
  published 2024 experiment remains hourly because its source product is hourly.
- Scheduled volumes clear fully at the published day-ahead price. Bid curves, partial
  fills, fees, imbalance exposure, network charges, taxes, and intraday re-trading are
  outside the first milestone.
- Revenue is always settled at realized historical clearing prices. Forecast prices
  appear only in the decision optimization.

## Information timing

- Each Phase 1 forecast has an explicit issue timestamp at 11:00 Europe/Berlin on the
  preceding day. Models may be refitted weekly, but prediction snapshots are daily.
- Realized load and generation are analysis variables, never predictors.
- Fundamental predictors must be historically archived forecasts published before
  issuance. Renaming realized values to `*_forecast_mw` violates the experiment.
- Price lags are at least 24 hours. DST is handled in UTC; local clock-hour features can
  repeat in autumn and skip in spring, matching the physical timeline.

## Battery

- Default asset: 1 MW / 2 MWh, 10–90% SOC, 95% charge and discharge efficiencies.
- Power denotes grid-side energy per one-hour interval. Auxiliary losses and standby
  consumption are omitted.
- Linear throughput degradation is an economic proxy, not a cell-aging model.
- Phase 1 uses a mixed-integer operating mode and a shared inverter constraint, so
  charging and discharging cannot occur simultaneously even under negative prices.
- SOC is carried continuously between German delivery days. A configurable terminal
  energy value replaces an artificial daily reset to 50%.
- German delivery days are constructed in Europe/Berlin and may contain 23, 24, or 25
  hourly intervals across daylight-saving transitions.

## Risk-aware decision

The robust strategy charges against a blend of P50 and P90 and discharges against a
blend of P50 and P10. `risk_aversion` controls the tail weight. This is an interpretable
quantile-robust policy, not a claim of full CVaR optimization. Forecast calibration is
therefore reported alongside trading outcomes.

## Multi-market limitation

No reserve revenues are synthesized. A valid multi-market experiment requires actual
capacity prices, product durations, symmetric/asymmetric power requirements,
availability obligations, activation volumes, energy settlement, and gate closure
rules. Until those data and rules are implemented, the project reports energy-only
results and exposes the limitation.

