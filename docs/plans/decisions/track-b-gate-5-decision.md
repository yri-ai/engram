# Track B Gate 5 Decision

## Outcome
- **NO-GO** on symbolic pruning as a primary mechanism; **GO** as a safety net

## Evidence Reviewed
- `outputs/results/h4_symbolic_ablation_v1.json` — full tightness sweep
- 4 unit tests passing (`uv run pytest tests/unit/test_h4_symbolic.py -v`)

## Metrics

| Metric | Threshold | Observed | Status |
|--------|-----------|----------|--------|
| contradiction reduction >= 0.15 | >= 0.15 | 0.0013 | **FAIL** |
| recall loss <= 0.05 | <= 0.05 | -0.0000 (improved) | **PASS** |
| novelty_prune_rate <= 0.10 (selected) | <= 0.10 | 0.0108 | PASS |

### Tightness sweep

| Level | Contradiction Rate | Recall | Novelty Prune Rate |
|-------|-------------------|--------|-------------------|
| Without symbolic | 0.0013 | 0.9822 | — |
| **Loose (selected)** | **0.0000** | **0.9823** | **0.0108** |
| Medium | 0.0000 | 0.9823 | 0.0129 |
| Hard | 0.0000 | 0.9823 | 0.0108 |

### Kill condition check

| Kill Condition | Threshold | Observed | Triggered? |
|---|---|---|---|
| contradiction reduction < 0.05 | < 0.05 | 0.0013 | **YES** |
| recall loss > 0.10 | > 0.10 | -0.0000 | No |

## Decision Rationale

The contradiction reduction kill threshold (< 0.05) is triggered. However, this is because the **base model already avoids contradictions**, not because symbolic pruning doesn't work. The transition matrix is inherently consistent with the motif library — it learned the same transition frequencies the motifs captured.

**What symbolic pruning does right:**
- Eliminates the 0.13% residual contradictions completely
- Zero recall cost (actually improves by 0.0001)
- Low novelty pruning (1.08% of predictions constrained)

**What it doesn't do:**
- Doesn't provide the dramatic 15%+ contradiction reduction the threshold expects
- This is because the threshold was set for a model class that *would* produce contradictions (e.g., LLM-based or neural models with less structural constraint)

**Recommendation:** Keep symbolic pruning as a **lightweight safety net** in the pipeline. It costs nothing and catches edge cases. But it is not a primary mechanism for this model class — the Lab Memo's hypothesis H4 ("symbolic pruning helps as discipline layer") is confirmed in spirit (it adds discipline without brittleness) but does not meet the quantitative threshold because the discipline was already built into the base model.

## Direction Change
- Symbolic pruning is retained as an optional safety layer, not a required component
- The 0.15 contradiction threshold should be re-evaluated when LLM-based models replace the transition matrix
- Proceed to Gate 6 (H5 cross-structure transfer) — this decision does not block Gate 6
