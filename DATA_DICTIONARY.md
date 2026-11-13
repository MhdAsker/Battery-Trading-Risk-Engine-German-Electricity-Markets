# Data dictionary

All canonical timestamps are timezone-aware UTC. German local time is derived only
for presentation and calendar features.

| Field | Unit | Source / meaning | Availability used in forecasting |
|---|---:|---|---|
| `timestamp_utc` | UTC | Delivery interval start; primary index | Known |
| `price_eur_mwh` | EUR/MWh | ENTSO-E A44 German/Luxembourg day-ahead clearing price | Available after day-ahead auction; target, never a same-day feature |
| `load_mw` | MW | ENTSO-E A65 realized total load | Ex-post analysis only |
| `load_forecast_mw` | MW | Published day-ahead load forecast | Must have publication time before forecast cutoff |
| `wind_mw` | MW | Aggregated realized onshore/offshore wind | Ex-post analysis only |
| `solar_mw` | MW | Realized solar generation | Ex-post analysis only |
| `wind_forecast_mw` | MW | Published day-ahead wind forecast | Must have publication time before cutoff |
| `solar_forecast_mw` | MW | Published day-ahead solar forecast | Must have publication time before cutoff |
| `residual_load_forecast_mw` | MW | Load forecast minus wind and solar forecasts | Derived at issue time |
| `price_lag_24h` | EUR/MWh | Target shifted 24 UTC hours | Known under the conservative cutoff assumption |
| `p10`, `p50`, `p90` | EUR/MWh | Conditional price quantiles | Produced at forecast issue time |
| `charge_mw` | MW | Grid-side battery charging power | Decision |
| `discharge_mw` | MW | Grid-side battery discharging power | Decision |
| `soc_mwh` | MWh | Stored energy after the interval | Simulated state |
| `soc_start_mwh` | MWh | Stored energy before the interval | Simulated state |
| `interval_hours` | h | Market time unit used in energy accounting | Known from product |
| `forecast_issue_time_utc` | UTC | Daily forecast snapshot timestamp | Audit field |
| `feature_available_time_utc` | UTC | Latest input publication time | Must not exceed issue time |
| `delivery_day_local` | date | Europe/Berlin market delivery date | Settlement grouping |
| `gross_revenue_eur` | EUR | Realized price × (discharge − charge) | Ex-post settlement |
| `throughput_mwh` | MWh | Charge plus discharge energy per hourly interval | Ex-post accounting |
| `net_revenue_eur` | EUR | Gross revenue minus linear degradation cost | Ex-post settlement |

ENTSO-E production type codes are kept as source codes in raw generation files;
semantic labels should be added during processing with a versioned mapping. Reserve
market fields will be added only after a real regelleistung.net schema is acquired.

Area identifiers are product-specific: day-ahead prices use the DE-LU bidding zone
`10Y1001A1001A82H`, while German load and generation use the German control-area code
`10Y1001A1001A83F`.
