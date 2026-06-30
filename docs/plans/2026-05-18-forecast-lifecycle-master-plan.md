# Forecast Lifecycle Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a measurable forecast lifecycle platform (question -> run -> resolve -> score) before deeper extraction work.

**Architecture:** Keep the current branch forecaster as the scoring engine and wrap it with lifecycle persistence and CLI workflows. Use existing graph primitives (`Entity`, `Fact`, `GraphStore`) for first implementation, then harden cutoff integrity and only then expand extraction depth.

**Tech Stack:** Python 3.11+, Pydantic, Typer CLI, existing GraphStore abstraction (MemoryStore/Neo4j), pytest, mypy, ruff.

## Milestones and Hand-off Plans

### Milestone Group A (Foundations): M1-M3

- M1: canonical forecast lifecycle models.
- M2: repository + persistence path.
- M3: CLI lifecycle commands.
- Detailed execution plan: `docs/plans/2026-05-18-forecast-lifecycle-m1-m3.md`.

### Milestone Group B (Measurement + Integrity): M4-M6

- M4: scoring + calibration report.
- M5: cutoff/leakage-proof retrieval.
- M6: extraction depth expansion.
- Detailed execution plan: `docs/plans/2026-05-18-forecast-lifecycle-m4-m6.md`.

## Completion Gates (global)

Every milestone must pass before moving to the next:

1. Tests for changed behavior are added first (red-green-refactor).
2. Unit tests for changed surface pass.
3. `uv run mypy src` passes.
4. `uv run ruff check` passes on changed files.
5. CLI behavior remains backward compatible for existing `forecast` command.

## Definition of Done (Stacked, Mandatory)

The program is not complete unless all three layers are fully complete in order.

### Layer 1: Operational MVP (required first)

Pass criteria:

1. At least 20 real forward-looking forecast questions are created before outcomes are known.
2. 100% of those questions have persisted lifecycle artifacts:
   - `ForecastQuestion`
   - one or more `ForecastRun` records
   - `ForecastResolution`
3. 100% of runs include valid `forecast_as_of` and evidence provenance references.
4. Leakage protection is verified by tests and spot audit:
   - future evidence does not change pre-cutoff forecast results,
   - retrieval path enforces cutoff.

Failure condition: if artifacts are missing, timestamps are invalid, or leakage controls fail, Layer 1 is not complete.

### Layer 2: Quality MVP (required second)

Pass criteria:

1. Forecast quality beats a naive baseline on Brier score over resolved forecasts.
2. Calibration is acceptable against defined ECE threshold.
3. Top-1 accuracy and sample count are reported for every evaluation window.
4. Metrics remain stable/improving across at least two consecutive evaluation windows.

Failure condition: if baseline is not beaten or calibration thresholds are not met, Layer 2 is not complete.

### Layer 3: Decision Impact MVP (required third)

Pass criteria:

1. Forecast outputs are explicitly referenced in real decision records.
2. At least one business impact metric is improved (for example: avoided diligence loss, improved decision hit-rate, reduced avoidable risk spend).
3. Impact evidence links decisions to forecasts and resolutions (not anecdotal).
4. Impact is measured over a defined period and compared to a pre-forecast baseline window.

Failure condition: if forecasts are not used in decisions or impact is unproven, Layer 3 is not complete.

## Final Completion Rule

Engram has achieved the forecasting-platform goal only when Layer 1, Layer 2, and Layer 3 are all complete.

- Layer 1 alone: not complete.
- Layers 1 + 2: not complete.
- Layers 1 + 2 + 3: complete.

## Risk Controls

- Keep lifecycle artifacts namespaced in metadata to avoid collision with existing facts.
- Require explicit `forecast_as_of` on persistent runs.
- Treat extraction depth changes as lower priority than measurable forecasting loop.

## Public Testing Corpus (Not Implementation Scope)

This program also requires a non-secret testing corpus so forecasting quality can be validated beyond the initial confidential deals.

This is a testing and evaluation track, not a core implementation milestone.

### Purpose

- reduce dependence on secret internal deals for early validation,
- create a shareable seed set for forecast lifecycle testing,
- pressure-test whether the system can forecast from real public evidence trails.

### Recommended public source mix

1. `SEC / EDGAR` filings for public REIT and real-estate-heavy public-company deals.
2. `CourtListener / RECAP` distressed-sale and bankruptcy sale timelines.
3. County recorder systems such as `NYC ACRIS` for close confirmation.

### What the testing corpus must contain

Each public deal should include:

1. canonical deal ID,
2. timestamped public evidence documents,
3. milestone history,
4. final resolved branch outcome,
5. enough chronology to reconstruct what was knowable at forecast time.

### Initial target

- 10-15 public REIT / EDGAR deals,
- 5-7 distressed/public-sale deals from `CourtListener`,
- recorder confirmation used as supporting evidence rather than primary source.

### Role in overall success

- This corpus is part of the testing/evaluation path for Layer 1 and Layer 2.
- It does not replace the confidential deal set.
- It strengthens evidence that the forecasting loop works on real non-secret cases.
