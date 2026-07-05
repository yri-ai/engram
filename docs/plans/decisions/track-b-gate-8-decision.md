# Track B Gate 8 Decision — Graph-Native Head and Ensemble

Gate 8 status: PASS

## Thresholds
- Graph-native head adds positive Brier delta relative to Phase 1 TFM reference.
- ULTRA and temporal-GNN deterministic adapters are both protocol-compatible and record-time-safe.

## Observed
- TFM reference Brier: `0.075000`
- ULTRA graph Brier: `0.000567`
- Temporal GNN Brier: `0.000567`
- Positive graph delta: `True`

## Artifacts
- `outputs/results/gate8_artifact_v1.json`
- graph export tests and protocol tests

## Decision
Proceed to LLM branch/supervisor Phase 3 with graph head retained on the scoreboard.
