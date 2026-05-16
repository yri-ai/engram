# Track B Gate 1 Decision

## Outcome
- **PASS**

## Evidence Reviewed
- `outputs/track_b/events.ndjson` — 470,000 labeled rows from Ginnie Mae payment history (10,000 loans × 48 months)
- `outputs/results/track_b_forecast_v1.json` — baseline backtest on 60,000 eval rows
- `outputs/results/track_b_forecast_v1_rerun.json` — identical rerun for drift check
- 34 unit tests passing (`uv run pytest tests/unit/test_track_b_*.py -v`)

## Metrics

| Metric | Threshold | Observed | Status |
|--------|-----------|----------|--------|
| sample_predictions >= 100 | >= 100 | 60,000 | PASS |
| top1_accuracy > 0.0 | > 0.0 | 0.9823 | PASS |
| rerun drift <= 0.01 | <= 0.01 | 0.0000 | PASS |

### Eval transition distribution

| From → To | Count | % of eval |
|-----------|-------|-----------|
| current → current | 58,030 | 96.7% |
| current → d30 | 713 | 1.2% |
| d30 → current | 684 | 1.1% |
| d60 → current | 111 | 0.2% |
| current → d60 | 111 | 0.2% |
| current → d90_plus | 92 | 0.2% |
| d90_plus → current | 76 | 0.1% |
| other transitions | 183 | 0.3% |

### Split counts

| Split | Rows |
|-------|------|
| Train | 400,000 |
| Eval | 60,000 |
| Holdout | 10,000 |

## Decision Rationale

All three Gate 1 thresholds are met. The pipeline is fully deterministic (zero drift on rerun). The baseline forecaster uses a real transition probability matrix, not hash-based placeholders.

The baseline accuracy of 98.2% reflects a highly stable portfolio where most loans stay current month-to-month. The 1.8% error rate corresponds to ~1,070 eval rows with actual bucket transitions — these are the cases where a more sophisticated model (Gates 2+) could improve predictions.

The data source (Ginnie Mae payment history) provides 48 months of history per loan, giving sufficient temporal depth for the transition-first prediction experiments in Gate 2.

## Direction Change
- Gate 1 baseline is established. Proceed to Gate 2 (H3 predictive primitive comparison).
- The ~1.8% transition rate means Gate 2 should focus on class-imbalanced metrics (Brier score, per-class recall) alongside top1_accuracy.
- No infrastructure changes needed — all Gate 2 experiments can run on the same event dataset.
