# Track B Gate 4 Decision

## Outcome
- **PASS**

## Evidence Reviewed
- `outputs/results/h1_schema_library_v1.json` — 77 motifs, full granularity sweep
- 7 unit tests passing (`uv run pytest tests/unit/test_h1_schema.py -v`)
- 470,000 events from Ginnie Mae payment history

## Metrics

| Metric | Threshold | Observed | Status |
|--------|-----------|----------|--------|
| average schema size <= 12 | <= 12 | **3.0** | PASS |
| transfer score >= 0.60 | >= 0.60 | **0.9998** | PASS |
| selected granularity within 0.01 of best | within 0.01 | 0.0000 diff | PASS |

### Granularity sweep

| Level | Accuracy | Avg Nodes | Motif Count |
|-------|----------|-----------|-------------|
| **event_only (selected)** | **0.9861** | **3.0** | **77** |
| event_plus_state | 0.9861 | 3.0 | 92 |
| event_state_gate | 0.9861 | 3.0 | 94 |

All three levels achieve identical accuracy. event_only is selected as the minimal sufficient level. Adding momentum direction or UPB gates produces more motifs (92, 94) but no accuracy gain — the extra complexity is pure noise for this domain.

### Transfer evaluation

| Metric | Value |
|--------|-------|
| In-family baseline accuracy | 0.9863 |
| Schema-guided accuracy (transfer) | 0.9861 |
| Transfer score | 0.9998 |
| Schema coverage | 1.0000 |

Schemas induced from 70% of loans predict outcomes on the remaining 30% with virtually no degradation. 100% of eval windows match a known motif.

### Top motifs discovered

| Motif | Pattern | Outcome | Support |
|-------|---------|---------|---------|
| M1 | current → current → current | current | 6,971 |
| M2 | current → current → current | **d30** | 839 |
| M3 | current → current → d30 | current | 821 |
| M4 | current → d30 → current | current | 757 |
| M5 | d30 → current → current | current | 757 |
| M6 | current → d30 → current | d30 | 349 |
| M7 | d30 → current → d30 | current | 339 |

M2 is the key motif for forecasting: a loan that has been current for 3 months can still transition to d30 (839 cases). M6 and M7 show the "revolving door" pattern — loans that bounce between current and d30.

## Decision Rationale

H1 (reusable precursor schemas) is **strongly supported**:

1. **Schemas are compact**: 3 nodes per motif, well under the 12-node limit
2. **Schemas are reusable**: 0.9998 transfer score — near-perfect generalization from train to eval loans
3. **Schemas are interpretable**: The top motifs correspond to recognizable delinquency patterns (stable, deterioration, recovery, revolving)
4. **Minimal granularity suffices**: event_only (just bucket sequences) captures all the signal; adding state/gate dimensions adds motifs but not accuracy

The simplicity of these schemas is the finding. Delinquency transition patterns in Ginnie Mae pools are captured by 77 short bucket-sequence motifs with no need for latent states or gate conditions.

## Direction Change
- H1 confirmed: schemas are real, compact, reusable, and minimal
- Proceed to Gate 5 (H4 symbolic pruning) and Gate 6 (H5 cross-structure transfer)
- Gate 5 can use the motif library as the symbolic constraint set
- Gate 6 will need a second structural family to test cross-family transfer
