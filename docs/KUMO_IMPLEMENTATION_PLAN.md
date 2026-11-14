# Kumo Forecast and domain-aware retrieval implementation plan

## Audit snapshot

- Project branch: `feature/kumo-darr-research`.
- Frozen evidence: `results/phase1_2024` and `results/phase2_2024` are read-only inputs.
- Forecast issue: 11:00 Europe/Berlin on D-1.
- Price-vintage contract: a price can enter context only when its publication timestamp is not
  later than the issue time.
- Delivery contract: physical UTC indices representing 23-, 24- or 25-hour German days.
- Battery contract: 1 MW / 2 MWh, 10%-90% SOC, 95% one-way efficiencies, EUR 5/MWh
  throughput degradation, 1.5 EFC/day, continuous strategy-specific SOC.
- Settlement contract: schedules are frozen on forecasts and settled only against observed
  prices; missing settlement intervals exclude the delivery day.
- Hardware: NVIDIA GTX 1650 with 4 GB VRAM; CPU fallback is available.

## Verified NVIDIA capability

The official source inspected was NVIDIA/Kumo-TS commit
`ad459475d1302fc8365e7117e39837a8ecb6a7e3` (audit date 2026-11-14).

- Package: `kumo-forecast` 1.0.1, Python >=3.10, PyTorch >=2.7.
- License: Apache-2.0 for NVIDIA-authored source, with third-party notices.
- API: `sdk.forecasting.ForecastingConfig` and `perform_forecasting`.
- Checkpoints: `nvidia/Kumo-Forecast`; default cross-channel checkpoint
  `run8_best_model_cr.pt`, base checkpoint `moment_head_512_6hr.pt`.
- Default context length: 512 observations.
- Maximum requested forecast horizon: 512 observations.
- Longer-than-native requests use autoregressive rollout.
- DARR builds embeddings from historical windows, retrieves cosine-nearest neighbors and blends
  `alpha * direct + (1 - alpha) * kNN`.
- DARR parameters exposed by the SDK include `alpha`, `k`, `temperature` and
  `context_stride`.
- Standard inference fills numeric missing values with zero. The separate missingness-aware path
  exists, but the current SDK rejects its use with DARR.
- Numeric columns are model channels, not verified future exogenous covariates. The integration
  therefore begins univariate and does not claim future weather support.
- The public DARR API returns the forecast, not neighbor identities. Electricity-regime
  retrieval and auditable neighbors are implemented externally and labelled as project research.
- Kumo does not provide calibrated P10/P50/P90 output or battery optimization. Both remain
  external project components.

## Execution gates

1. Implement a backend-injected optional adapter and temporal contracts.
2. Pass direct, DARR, DST, missingness and deliberate-leakage tests without model weights.
3. Run one official direct inference smoke origin and record device/time/memory if the checkpoint
   fits available resources.
4. Run a small 2023 development diagnostic spanning negative-price, volatile, scarcity-like and
   ordinary days.
5. Continue to native DARR only if direct inference is operational. Stop a broad DARR search if
   the small diagnostic shows no forecast or dispatch evidence.
6. Freeze checkpoint, direct/DARR settings, external retrieval, calibration and optimization
   parameters before any retrospective 2024 evaluation.
7. Publish positive and negative results with paired revenue inference and compute cost.

The 2024 study is a retrospective extension because its benchmark outcomes are already public;
it is not described as a genuinely untouched test.
