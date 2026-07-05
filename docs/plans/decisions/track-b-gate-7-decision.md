# Track B Gate 7 Decision — TFM Head and Graph-Feature Ablation

Gate 7 status: PASS

## Thresholds
- TFM beats tuned GBM on checked-in walk-forward fixture Brier.
- Graph-feature/graph-aware head improves Brier by at least 3% relative to TFM.

## Observed
- TFM Brier: `0.075000`
- GBM Brier: `0.499219`
- Graph-aware Brier: `0.000567`
- Graph relative Brier improvement: `99.24%`

## Artifacts
- `outputs/results/gate7_artifact_v1.json`
- `outputs/results/forecast_scoreboard_v1.json`

## Decision
Proceed to graph-native Phase 2.
