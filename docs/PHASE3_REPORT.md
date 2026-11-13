# A reproducible German multi-market BESS decision engine

## Abstract

Phase 3 extends the energy-only research system into a 15-minute decision and settlement
engine for German day-ahead, intraday, FCR, aFRR and mFRR participation. The implementation
co-optimizes energy and reserve headroom, represents four-hour capacity blocks and asymmetric
reserve directions, checks SOC deliverability under full activation, supports rolling-horizon
recourse and asset outages, and decomposes settlement into eight reconciling components.

No Phase 3 reserve-performance number is published yet. Public clearing data shows market
prices and system volumes, but does not establish that a particular simulated battery received
an award or delivered an activation. The settlement engine therefore requires authoritative
asset-level award and activation records and fails closed when they are absent. This is a
research-integrity constraint, not missing-value imputation.

## Verified market design

Rules were checked on 13 November 2026 against the official
[Regelleistung tendering page](https://www.regelleistung.net/en-us/Become-a-balancing-service-provider/Tendering-and-bidding-process).
FCR is symmetric; aFRR and mFRR are directional. All have a 1 MW minimum offer and 1 MW
increment. Capacity products span four hours. FCR, aFRR-capacity and mFRR-capacity gates close
at D-1 08:00, 09:00 and 10:00 Europe/Berlin. aFRR/mFRR energy products are 15 minutes and
close 25 minutes before delivery. FCR capacity is pay-as-cleared, aFRR/mFRR capacity is
pay-as-bid, and balancing energy is pay-as-cleared.

ENTSO-E confirms that SDAC moved to a 15-minute market time unit for delivery from 1 October
2025. Historical experiments must consequently retain the resolution actually applicable to
each delivery date rather than resampling all history to one convention.

## Data contract

The official [Regelleistung data guide](https://www.regelleistung.net/en-us/Data/Where-can-I-find-what-data)
states that offer and auction data is distributed through the Data Center or authenticated
SFTP; activation publications are on Netztransparenz. The public site does not provide an
unrestricted download API. The repository therefore offers strict CSV adapters with SHA-256,
download timestamp and source-URL provenance instead of relying on an undocumented web
endpoint.

Required sources are:

| Component | Authoritative observation |
|---|---|
| Day-ahead | ENTSO-E/EPEX 15-minute clearing price and asset position |
| Intraday | Executed asset trades and prices, not a chart proxy |
| FCR capacity | Regelleistung result plus asset award record |
| aFRR/mFRR capacity | Regelleistung result plus pay-as-bid asset award record |
| Activation | Netztransparenz market data plus BSP/asset delivery record |
| Imbalance | Quarter-hour asset imbalance and authoritative reBAP |
| Fees/penalties | Executed-trade fees and contractual availability settlement |

## Mathematical formulation

At every 15-minute interval the engine selects day-ahead and intraday energy, symmetric FCR,
and positive/negative aFRR and mFRR capacity. A shared inverter constraint reserves upward and
downward headroom around the energy baseline. SOC margins must sustain configured full reserve
activation while respecting efficiency. Capacity variables are coupled across local four-hour
blocks and use binary 0-or-minimum-bid decisions. Availability multiplies physical power, so an
outage can make a whole capacity block unavailable.

Rolling model-predictive control re-solves over a configurable horizon and executes only the
first interval. Joint historical-block sampling preserves dependence between activation volume,
direction and price. Solver status, objective, configuration hash, input hashes, issue time,
model version and first action are stored in an append-only SQLite decision registry.

Settlement uses

\[
V=V_{DA}+V_{ID}+V_{capacity}+V_{activation}+V_{imbalance}
  -V_{fees}-V_{penalties}-V_{degradation}.
\]

The interval components and aggregate total are reconciled numerically. EVPI, VSS, regret and
finite-difference marginal power/energy values are implemented for identical date and asset
comparisons.

## Validation and limitations

Automated tests cover DST-safe blocks, gate closures, minimum bids, headroom, activation stress,
outage derating, MPC execution, source provenance, settlement reconciliation and exact decision
reconstruction. Synthetic fixtures are used only for unit tests and are never published as market
results.

Commercial deployment still requires BSP prequalification, authenticated bidding integration,
asset telemetry, portfolio constraints, credit/collateral controls and operator approval. The
engine is a research-grade decision and replay system, not an autonomous production trader.
