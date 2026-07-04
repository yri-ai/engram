# Forecast Lifecycle M4-M6 Implementation Plan

> **STATUS: HISTORICAL — DO NOT EXECUTE (2026-07-04).** M4–M5 shipped via the temporal forecasting kernel (`src/engram/services/forecast_scoring.py`, `src/engram/services/as_of_evidence.py`, `forecast-score-report` CLI). M6 (extraction depth) is NOT done and is now gated per the Task Ordering section of the master plan v2 — it must not be started from this document. Current work: `docs/plans/2026-05-18-forecast-lifecycle-master-plan.md` (v2) and `docs/plans/2026-07-04-forecast-lifecycle-m7-m8.md`. Kept for provenance only.

**Goal:** Add scoring/calibration, enforce leakage-proof retrieval, and then improve extraction depth with measurable impact.

**Architecture:** Build a scoring service over persisted run/resolution artifacts, add strict `as_of` retrieval filtering in forecast evidence loading, and only then expand structured extraction quality while preserving provenance and lifecycle compatibility.

**Tech Stack:** Python, Pydantic, existing forecast services, pytest fixtures, CLI Typer commands.

## Milestone Group Completion Target

This plan covers Layer 2 (Quality MVP) and enables Layer 3 (Decision Impact MVP) from the master Definition of Done.

To mark M4-M6 complete, the following must hold:

1. Layer 2 pass criteria are fully met (baseline-beating quality + calibration reporting),
2. Layer 3 instrumentation/traceability is present so decision impact can be proven,
3. extraction improvements are measurable against quality metrics (not anecdotal).

Important constraint:

- This plan cannot claim end-state completion alone; final completion requires demonstrated decision impact outcomes per Layer 3 in `docs/plans/2026-05-18-forecast-lifecycle-master-plan.md`.

## Public Testing Corpus Dependency

Evaluation in this milestone group should also use the public testing corpus defined in `docs/plans/2026-05-18-forecast-lifecycle-master-plan.md`.

Purpose:

- validate Layer 1 and Layer 2 behavior on non-secret deals,
- reduce overfitting to the confidential internal sample,
- test whether public evidence trails are sufficient for time-locked branch forecasting.

Constraint:

- this public corpus is part of testing and evaluation only, not part of the implementation scope for these milestones.

### Task 1: Add forecast scoring service

**Files:**
- Create: `src/engram/services/forecast_scoring.py`
- Test: `tests/unit/test_forecast_scoring.py`
- Create: `tests/fixtures/forecast_scores.json`

**Step 1: Write failing tests**
- Add tests for Brier score, top-1 accuracy, ECE binning, aggregate sample count.

**Step 2: Run tests and verify fail**
- Run: `uv run pytest tests/unit/test_forecast_scoring.py -q`
- Expected: FAIL because service is missing.

**Step 3: Implement minimal scoring service**
- Implement deterministic computations and typed report output.

**Step 4: Run tests and verify pass**
- Run: `uv run pytest tests/unit/test_forecast_scoring.py -q`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/services/forecast_scoring.py tests/unit/test_forecast_scoring.py tests/fixtures/forecast_scores.json && git commit -m "feat: add forecast scoring metrics service"`

### Task 2: Add score-forecasts CLI command

**Files:**
- Modify: `src/engram/cli/main.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Write failing test**
- Add test for `score-forecasts` command returning JSON report payload.

**Step 2: Run test and verify fail**
- Run: `uv run pytest tests/unit/test_cli.py -q -k score_forecasts`
- Expected: FAIL.

**Step 3: Implement minimal command**
- Wire repository and scoring service, emit aggregate + per-question report.

**Step 4: Run test and verify pass**
- Run: `uv run pytest tests/unit/test_cli.py -q -k score_forecasts`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/cli/main.py tests/unit/test_cli.py && git commit -m "feat: add score-forecasts CLI"`

### Task 3: Enforce cutoff/leakage integrity in evidence retrieval

**Files:**
- Modify: `src/engram/services/branch_forecasting.py`
- Create: `tests/unit/test_forecast_leakage_control.py`

**Step 1: Write failing tests**
- Add tests with future-dated evidence that would alter forecast if leaked.

**Step 2: Run tests and verify fail**
- Run: `uv run pytest tests/unit/test_forecast_leakage_control.py -q`
- Expected: FAIL.

**Step 3: Implement minimal retrieval filter**
- Add explicit `forecast_as_of` filter to graph evidence load path.

**Step 4: Run tests and verify pass**
- Run: `uv run pytest tests/unit/test_forecast_leakage_control.py -q`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/services/branch_forecasting.py tests/unit/test_forecast_leakage_control.py && git commit -m "feat: enforce as-of leakage control in forecasting"`

### Task 4: Add structured extraction module baseline

**Files:**
- Create: `src/engram/models/structured.py`
- Create: `src/engram/services/structured_extraction.py`
- Test: `tests/unit/test_structured_extraction.py`

**Step 1: Write failing tests**
- Add tests for DOCX/XLSX extraction contracts and provenance fields.

**Step 2: Run tests and verify fail**
- Run: `uv run pytest tests/unit/test_structured_extraction.py -q`
- Expected: FAIL.

**Step 3: Implement minimal extraction layer**
- Start with deterministic inventory + text extraction where available.

**Step 4: Run tests and verify pass**
- Run: `uv run pytest tests/unit/test_structured_extraction.py -q`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/models/structured.py src/engram/services/structured_extraction.py tests/unit/test_structured_extraction.py && git commit -m "feat: add structured extraction baseline"`

### Task 5: Full verification gate for M4-M6

**Files:**
- Test: `tests/unit/test_forecast_scoring.py`
- Test: `tests/unit/test_forecast_leakage_control.py`
- Test: `tests/unit/test_structured_extraction.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Run focused unit tests**
- Run: `uv run pytest tests/unit/test_forecast_scoring.py tests/unit/test_forecast_leakage_control.py tests/unit/test_structured_extraction.py tests/unit/test_cli.py -q`

**Step 2: Run typing checks**
- Run: `uv run mypy src`

**Step 3: Run lint checks**
- Run: `uv run ruff check src/engram/services/forecast_scoring.py src/engram/services/branch_forecasting.py src/engram/models/structured.py src/engram/services/structured_extraction.py src/engram/cli/main.py tests/unit/test_forecast_scoring.py tests/unit/test_forecast_leakage_control.py tests/unit/test_structured_extraction.py tests/unit/test_cli.py`

**Step 4: Commit verification changes (if any)**
- Run: `git add -A && git commit -m "test: verify forecast scoring, integrity, and extraction milestones"`
