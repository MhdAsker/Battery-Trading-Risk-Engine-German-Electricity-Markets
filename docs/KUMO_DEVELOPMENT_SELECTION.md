# Frozen Kumo development-origin selection v1

This document freezes the expanded Kumo development panel before any new checkpoint inference.
The panel contains 24 delivery days in 2023 and remains strictly before the retrospective 2024
boundary.

## Eligibility

An origin was eligible only when, at the 11:00 Europe/Berlin D-1 issue time:

1. at least one complete 512-hour published-price context existed;
2. the path from that context to delivery-day end did not exceed Kumo's verified 512-step limit;
3. the most recent 168 published observations used by external retrieval were contiguous;
4. the delivery day had a complete settlement curve; and
5. all regime features were computed from already-published prices.

No delivery-day price level, spread, spike, negative-price event, or battery revenue was used to
select an origin. Settlement completeness is a data-availability screen, not an outcome screen.

## Stratification

Selection covers spring, summer, autumn and one meteorological-winter origin, weekdays and
weekends, the 25-hour autumn DST day, ordinary contexts, negative-price-risk contexts and
high-volatility contexts. Regime labels use the already-frozen thresholds: recent negative-price
share at least 2% and recent hourly-change volatility at least EUR 20/MWh. These statistics use
only the latest 168 prices available at issuance.

The strict missing-data policy materially limits calendar coverage. Eligible 2023 origins occur
only in May, June, August, September, October, November and one December date. January through
April and July cannot supply the required recent complete context without imputation. This
selection bias must be reported with results.

The authoritative manifest is `configs/kumo_development_origins_v1.csv`; the execution config is
`configs/kumo_development_24.yaml`. Fine-tuning and native DARR are outside this experiment.
