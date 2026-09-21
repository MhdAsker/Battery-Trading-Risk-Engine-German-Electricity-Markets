# Assumptions and research boundaries

## Market

- The simulated asset is a price taker in the German/Luxembourg day-ahead market.
- Dispatch is hourly in the initial experiment. ENTSO-E sub-hourly inputs are retained
  but must be explicitly aggregated before use.
- Scheduled volumes clear fully at the published day-ahead price. Bid curves, partial
  fills, fees, imbalance exposure, network charges, taxes, and intraday re-trading are
  outside the first milestone.
- Revenue is always settled at realized historical clearing prices. Forecast prices
  appear only in the decision optimization.

## Information timing

- A forecast is treated as issued before the applicable day-ahead auction cutoff.
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
- The optimizer is convex. With positive degradation and round-trip losses,
  simultaneous charge/discharge is dominated in ordinary prices; schedules are still
  checked for numerical artifacts.
- Daily experiments return to 50% SOC, preventing end-of-horizon energy dumping.

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

