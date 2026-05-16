# Track B Gate 6 Decision

## Outcome
- **HOLD** — insufficient data diversity to fully test H5; not killed

## Evidence Reviewed
- `outputs/results/h5_transfer_v1.json` — transfer artifact
- 3 unit tests passing (`uv run pytest tests/unit/test_h5_transfer.py -v`)
- 470,000 events split into stable (9,093 loans) and volatile (907 loans) families

## Metrics

| Metric | Threshold | Observed | Status |
|--------|-----------|----------|--------|
| transferable_motif_score >= 0.40 | >= 0.40 | 0.2632 | **FAIL** |
| transferable_motif_score >= 0.20 (kill) | >= 0.20 | 0.2632 | **Safe** |
| family_drift >= 0.40 (required for lift test) | >= 0.40 | 0.2500 | N/A |
| structural drivers non-empty | > 0.0 | 2.31 | **PASS** |

### Transfer comparison

| Model | Accuracy on volatile family |
|-------|---------------------------|
| Shared-core (all motifs) | 0.8779 |
| Family-specific (volatile motifs) | 0.8789 |
| Cross-family (stable motifs → volatile) | 0.8607 |

### Structural driver characterization (volatile family)

| Driver | Score | Interpretation |
|--------|-------|---------------|
| Threshold gates | 0.78 | 78% of transitions are ±1 bucket step |
| Counterparty behavior | 0.87 | High state-level variance in transition rates |
| Waterfall logic | 0.39 | 39% of transitions are step-by-step deterioration |
| Covenant/contract triggers | 0.22 | 22% involve non-adjacent bucket jumps |
| Optionality collapse | 0.05 | Direct-to-severe is rare |

### Kill condition check

| Kill Condition | Threshold | Observed | Triggered? |
|---|---|---|---|
| transferable_motif_score < 0.20 | < 0.20 | 0.2632 | **No** |
| no measurable benefit from family-specific | 0.0 | 0.0011 lift | **No** (marginal benefit exists) |

## Decision Rationale

This is a **HOLD**, not a PASS or KILL, because:

**Why not PASS:**
- Transferable motif score (0.26) is below the 0.40 threshold
- Family drift (0.25) is too low to trigger the lift test — the families aren't different enough to meaningfully test cross-structure transfer
- We're testing within a single data source (Ginnie Mae sub-populations), not across genuinely different structures (e.g., CMBS vs RMBS vs private credit)

**Why not KILL:**
- Transfer score is above the 0.20 kill threshold
- Cross-family accuracy (0.86) is reasonable — stable-family motifs capture 86% of volatile-family outcomes
- Family-specific accuracy (0.88) beats shared-core (0.88) and cross-family (0.86), showing family-specific motifs add marginal value
- Structural driver characterization produced meaningful results — threshold gates dominate, which is consistent with the delinquency domain

**The real test of H5 requires genuinely different structural families.** The Ginnie Mae stable/volatile split is a useful proxy but not the definitive experiment. The research thesis calls for "structured product unwind, private credit deterioration, sponsor stress, litigation escalation" — domains with fundamentally different transition grammars. This experiment shows the framework works; the hypothesis remains open.

## Direction Change
- H5 is **deferred**, not rejected — needs multi-source data to properly evaluate
- The structural driver characterization framework is validated and ready for use
- When CMBS (EDGAR data) or other product types are processed through the pipeline, re-run Gate 6
- The 0.87 counterparty behavior score suggests geographic segmentation could be a meaningful family axis for future experiments
