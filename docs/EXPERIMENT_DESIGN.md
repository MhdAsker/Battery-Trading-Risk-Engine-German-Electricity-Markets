# Experiment design

## Primary comparison

Use at least two complete years for training/validation and a later contiguous test
period. For each rolling origin: build features using only records available at the
issue time, refit on a trailing window, produce the next delivery block, freeze the
schedule, and settle it against observed prices.

Sparse source gaps are never imputed for target evaluation or revenue settlement. For
lagged price features only, the benchmark uses the same UTC hour one week earlier,
then one day earlier, and finally causal forward fill. A delivery day is traded only
when all 24 realized hourly prices exist.

Strategies:

1. Rule-based daily low/high quantile policy (baseline to add to the experiment runner).
2. Perfect-foresight optimizer using realized prices (unattainable upper bound).
3. Historical hour-of-week and persistence forecast dispatch.
4. ML P50 dispatch.
5. Quantile-robust P10/P50/P90 dispatch.
6. Multi-market only after real reserve data pass quality checks.

## Forecast evaluation

Report MAE and RMSE; P10/P50/P90 pinball loss; P10–P90 coverage and width. Slice by
negative prices, top 1% price spikes, top quartile renewable share, and top quartile
rolling 24-hour volatility. Thresholds are fitted on each training window.

## Economic evaluation

Report gross/net revenue, revenue per MW, revenue per MWh of energy capacity,
throughput, equivalent cycles, degradation cost, revenue volatility, maximum drawdown,
tail loss/CVaR, and regret against perfect foresight.

To test whether MAE implies value, calculate per-window forecast MAE and trading
revenue, then Spearman correlation and bootstrap its confidence interval. Attribute
expensive errors by whether they reverse the forecast spread ranking at scheduled
charge/discharge hours. Avoid causal claims: both accuracy and revenue depend on regime.

## Stress tests

Run the same frozen test dates over durations 1/2/4 h, round-trip efficiencies,
degradation costs, power-to-energy ratios, risk aversion, and controlled forecast-noise
levels. Only the synthetic noise stress test is synthetic and must be labelled so;
base prices remain observed.
