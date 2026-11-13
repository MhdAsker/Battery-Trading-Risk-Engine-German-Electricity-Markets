# Experiment design

## Primary comparison

Phase 1 issues a snapshot at 11:00 Europe/Berlin before each delivery day, while
refitting models every seven days. Every row stores issue time and latest feature
availability time. Dispatch uses German local delivery days, carries SOC between days,
freezes each schedule, and settles it against observed prices.

Sparse source gaps are never imputed for target evaluation or revenue settlement. For
lagged price features only, the benchmark uses the same UTC hour one week earlier,
then one day earlier, and finally causal forward fill. A delivery day is traded only
when every realized interval in its 23/24/25-hour German delivery day exists.

Strategies:

1. Zero-trade and rule-based daily low/high quantile policies.
2. Perfect-foresight optimizer using realized prices (unattainable upper bound).
3. Historical hour-of-week and persistence forecast dispatch.
4. Regularized linear and ML P50 dispatch.
5. Quantile-robust P10/P50/P90 dispatch.
6. Multi-market only after real reserve data pass quality checks.

## Forecast evaluation

Report MAE and RMSE; P10/P50/P90 pinball loss; P10–P90 coverage and width. Slice by
negative prices, top 1% price spikes, top quartile renewable share, and top quartile
rolling 24-hour volatility. Thresholds are fitted on each training window.

## Economic evaluation

Report gross/net revenue, revenue per MW-day, throughput, equivalent cycles,
degradation cost, daily volatility, loss probability, downside deviation, daily
VaR/CVaR, drawdown, and regret. Strategy differences use paired circular
block-bootstrap confidence intervals.

To test whether MAE implies value, calculate per-window forecast MAE and trading
revenue, then Spearman correlation and bootstrap its confidence interval. Attribute
expensive errors by whether they reverse the forecast spread ranking at scheduled
charge/discharge hours. Avoid causal claims: both accuracy and revenue depend on regime.

## Stress tests

Run the same frozen test dates over durations 1/2/4 h, round-trip efficiencies,
degradation costs, power-to-energy ratios, risk aversion, and controlled forecast-noise
levels. Only the synthetic noise stress test is synthetic and must be labelled so;
base prices remain observed.
