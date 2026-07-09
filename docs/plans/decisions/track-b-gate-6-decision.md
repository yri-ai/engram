# Track B Gate 6 Decision — Phase 0 Forecast Harness

Gate 6 status: PASS

## Thresholds
- Checked-in synthetic fixture baseline Brier reproduces expected artifact within ±0.001.
- Leakage canary is green.
- Baseline, hazard, and GBM models are present on the scoreboard.

## Observed
- Baseline Brier: `0.437500`
- Expected baseline Brier: `0.4375`
- Baseline Brier matches expected: `True`
- Leakage canary: `passed`
- Harness filter canary path detected: `True`
- Gate 6 leakage canary passed: `True`
- Hazard on board: `True`
- GBM on board: `True`
- Local Ginnie events present: `False`

## Artifacts
- Scoreboard JSON: `outputs/results/forecast_scoreboard_v1.json`
- Scoreboard summary: `outputs/results/forecast_scoreboard_v1.md`

## Decision
Proceed to Phase 1 only if this document says PASS. Do not add forecast heads without scoreboard entries.
