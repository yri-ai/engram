# Forecast Lifecycle M1-M3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add lifecycle models, persistence, and CLI commands so forecasts are stored and resolvable rather than transient output.

**Architecture:** Add new lifecycle models in a dedicated module, then add a repository service that persists lifecycle artifacts via existing graph primitives, then expose the flow via new CLI commands while preserving current `forecast` behavior.

**Tech Stack:** Python, Pydantic models, Typer CLI, GraphStore abstraction, pytest.

## Milestone Group Completion Target

This plan covers Layer 1 (Operational MVP) from the master Definition of Done.

To mark M1-M3 complete, all Layer 1 pass criteria in `docs/plans/2026-05-18-forecast-lifecycle-master-plan.md` must be satisfied.

Local success for this plan:

1. lifecycle contracts are implemented and test-backed,
2. repository persistence works for question/run/resolution,
3. CLI supports create/run/resolve flow,
4. cutoff metadata and provenance are persisted for every run.

Non-goal for this plan:

- Beating forecast baselines and proving business impact are out of scope here (handled in later layers).

### Task 1: Add lifecycle model module

**Files:**
- Create: `src/engram/models/forecasting.py`
- Modify: `src/engram/models/__init__.py`
- Test: `tests/unit/test_forecasting_models.py`

**Step 1: Write failing tests**
- Add tests for required fields and JSON round-trip on `ForecastQuestion`, `ForecastRun`, `ForecastResolution`, `ForecastScore`.

**Step 2: Run tests and verify fail**
- Run: `uv run pytest tests/unit/test_forecasting_models.py -q`
- Expected: FAIL because module/classes do not exist.

**Step 3: Add minimal models**
- Implement models with explicit required fields including `forecast_as_of`, branch probabilities, and resolution metadata.

**Step 4: Run tests and verify pass**
- Run: `uv run pytest tests/unit/test_forecasting_models.py -q`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/models/forecasting.py src/engram/models/__init__.py tests/unit/test_forecasting_models.py && git commit -m "feat: add forecast lifecycle models"`

### Task 2: Add lifecycle repository service

**Files:**
- Create: `src/engram/services/forecast_repository.py`
- Modify: `src/engram/services/__init__.py`
- Test: `tests/unit/test_forecast_repository.py`
- Reference: `src/engram/storage/base.py`

**Step 1: Write failing tests**
- Add repository tests using MemoryStore for create/list question, save/list runs, save/get resolution.

**Step 2: Run tests and verify fail**
- Run: `uv run pytest tests/unit/test_forecast_repository.py -q`
- Expected: FAIL because repository does not exist.

**Step 3: Implement minimal repository**
- Implement typed repository methods and deterministic IDs.
- Persist via existing store primitives and namespaced metadata.

**Step 4: Run tests and verify pass**
- Run: `uv run pytest tests/unit/test_forecast_repository.py -q`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/services/forecast_repository.py src/engram/services/__init__.py tests/unit/test_forecast_repository.py && git commit -m "feat: add forecast lifecycle repository"`

### Task 3: Add CLI command to create forecast questions

**Files:**
- Modify: `src/engram/cli/main.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Write failing test**
- Add CLI test for `create-forecast-question` that validates required flags and JSON output.

**Step 2: Run test and verify fail**
- Run: `uv run pytest tests/unit/test_cli.py -q -k create_forecast_question`
- Expected: FAIL due missing command.

**Step 3: Implement minimal command**
- Add Typer command that validates inputs and persists question via repository.

**Step 4: Run test and verify pass**
- Run: `uv run pytest tests/unit/test_cli.py -q -k create_forecast_question`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/cli/main.py tests/unit/test_cli.py && git commit -m "feat: add create-forecast-question CLI"`

### Task 4: Add CLI command to run and persist forecasts

**Files:**
- Modify: `src/engram/cli/main.py`
- Modify: `src/engram/services/branch_forecasting.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Write failing test**
- Add test for `run-forecast` ensuring `forecast_as_of` is required and run output is persisted.

**Step 2: Run test and verify fail**
- Run: `uv run pytest tests/unit/test_cli.py -q -k run_forecast`
- Expected: FAIL.

**Step 3: Implement minimal command path**
- Reuse existing evidence loading + BranchForecaster.
- Persist immutable run payload through repository.

**Step 4: Run test and verify pass**
- Run: `uv run pytest tests/unit/test_cli.py -q -k run_forecast`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/cli/main.py src/engram/services/branch_forecasting.py tests/unit/test_cli.py && git commit -m "feat: persist run output from CLI forecast"`

### Task 5: Add CLI command to resolve forecasts

**Files:**
- Modify: `src/engram/cli/main.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Write failing test**
- Add test for `resolve-forecast` with required outcome branch and timestamp.

**Step 2: Run test and verify fail**
- Run: `uv run pytest tests/unit/test_cli.py -q -k resolve_forecast`
- Expected: FAIL.

**Step 3: Implement minimal command**
- Add command to persist resolution linked to question/run.

**Step 4: Run test and verify pass**
- Run: `uv run pytest tests/unit/test_cli.py -q -k resolve_forecast`
- Expected: PASS.

**Step 5: Commit**
- Run: `git add src/engram/cli/main.py tests/unit/test_cli.py && git commit -m "feat: add resolve-forecast CLI"`

### Task 6: Full verification gate for M1-M3

**Files:**
- Test: `tests/unit/test_forecasting_models.py`
- Test: `tests/unit/test_forecast_repository.py`
- Test: `tests/unit/test_cli.py`

**Step 1: Run focused unit tests**
- Run: `uv run pytest tests/unit/test_forecasting_models.py tests/unit/test_forecast_repository.py tests/unit/test_cli.py -q`

**Step 2: Run typing checks**
- Run: `uv run mypy src`

**Step 3: Run lint checks**
- Run: `uv run ruff check src/engram/models/forecasting.py src/engram/services/forecast_repository.py src/engram/cli/main.py tests/unit/test_forecasting_models.py tests/unit/test_forecast_repository.py tests/unit/test_cli.py`

**Step 4: Commit verification changes (if any)**
- Run: `git add -A && git commit -m "test: verify forecast lifecycle foundations"`
