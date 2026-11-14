# Conditional Kumo retrieval pre-registration

This document freezes the policy and evaluation before new 2024 Kumo predictions are examined.
It follows the exploratory 24-origin result at commit `c610569` and is not an untouched study:
the repository's 2024 conventional-model outcomes are already public. The 2024 Kumo forecasts
are new, but the period is described as retrospective validation.

## Fixed hypothesis and policy

External regime retrieval is activated when the issuance-time snapshot contains either
`high_volatility` or `negative_price_risk`. Otherwise the forecast is Kumo Direct. The thresholds
remain the exploratory values: recent negative-price share >= 2% and recent hourly-change
volatility >= EUR 20/MWh over the last 168 published prices. No 2024 outcome is used to change
these thresholds, the 50/50 direct/retrieval blend, neighbor count, temperature, or fallback.

## Frozen chronological samples

- Policy-development/exploratory period: all strictly eligible 2023 origins (expected n=69).
- Retrospective-validation period: all strictly eligible 2024 origins (expected n=44).
- Combined execution panel: 113 origins.

Eligibility is mechanical: a complete 512-hour causal Kumo context, no more than 512 forecast
steps to delivery-day end, a contiguous latest 168-hour retrieval context, and a complete
settlement curve. No price level, event, forecast error, revenue, or preferred strategy is used
to select an otherwise complete day. Missing measurements are rejected, never filled.

## Comparators and endpoints

Forecasts: persistence, regularized linear, Quantile GBM P50, Kumo Direct, always-on external
retrieval, and conditional external retrieval. Battery assets and settlement conventions are
identical. Quantile GBM and linear are both pre-registered conventional comparators; validation
results will not be used to choose between them.

Primary economic endpoint: paired daily net-revenue difference on the 2024 retrospective
validation origins. Primary uncertainty uses a seven-origin circular block bootstrap, with
three- and fourteen-origin sensitivity intervals. Forecast MAE differences use the same paired
origins. Confidence intervals crossing zero are not superiority.

Conditional retrieval is considered supported only if its primary revenue confidence interval
is positive versus both Kumo Direct and always-on retrieval. Comparisons with linear and
Quantile GBM determine whether Kumo supplies incremental value; they do not alter the policy.

## Evidence labels

- 2023 results: policy-development/exploratory evidence.
- 2024 results: retrospective chronological validation, not untouched confirmation.
- Native DARR and Kumo fine-tuning: disabled and outside scope.
