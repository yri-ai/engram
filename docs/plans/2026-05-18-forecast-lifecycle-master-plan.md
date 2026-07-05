# Forecast Lifecycle Master Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a measurable forecast lifecycle platform (question -> run -> resolve -> score) before deeper extraction work.

**Revision note (2026-07-04):** v2 supersedes the 2026-05-18 original. M1–M6 foundations shipped via the temporal forecasting kernel (`docs/plans/2026-06-28-temporal-forecasting-kernel.md`). The May hand-off plans (`...-m1-m3.md`, `...-m4-m6.md`) are **HISTORICAL** — their greenfield tasks are implemented; do not execute them. Remaining work is defined here and in `docs/plans/2026-07-04-forecast-lifecycle-m7-m8.md`.

## Architecture (corrected to match code)

- **Forecasting engine:** `BranchForecaster.forecast(...)` (`src/engram/services/branch_forecasting.py:502`) and the auditable run-creation path `DeterministicForecastProtocol` (`src/engram/services/forecast_protocol.py`). These produce forecasts; they do not score.
- **Scoring layer:** `src/engram/services/forecast_scoring.py` — `ForecastScorer`, Brier/log/top-k metrics, `build_calibration_report` (ECE via calibration buckets).
- **Evidence integrity:** `AsOfEvidenceCompiler` (`src/engram/services/as_of_evidence.py`) enforces cutoff/leakage rules at compile time; CLI re-checks at run creation (`src/engram/cli/main.py` forecast-run-create guards).
- **Persistence — primary:** `JsonForecastRepository` (`src/engram/services/forecast_repository.py:26`), the append-only JSON ledger (`questions/ runs/ resolutions/ scores/`). This is the MVP path of record.
- **Persistence — secondary/compat:** graph-backed `ForecastRepository` (`src/engram/services/forecast_repository.py:145`). Status: compatibility layer for graph-native deployments. **Parity scope (explicit):** the graph backend supports questions, runs, and resolutions only; scores, dossiers, and decision records are JSON-ledger-only until graph support is added (tracked, not assumed). Parity contract applies to the supported subset: questions/runs/resolutions written to one backend must round-trip through the other via a normalized export format without semantic loss (field-normalized comparison, not byte equality — test-enforced in M7 Task 1). New features land JSON-first; each release documents the graph-backend gap list.

## CLI Surface (one canonical set)

**Canonical (kernel commands, `src/engram/cli/main.py:579–718`):** `forecast-question-create`, `forecast-dossier-compile`, `forecast-run-create`, `forecast-resolve-create`, `forecast-score-report`.

**Legacy (deprecated as of v2):** `create-forecast-question`, `run-forecast`, `resolve-forecast`, `score-forecasts` (`src/engram/cli/main.py:741–1021`). M7 Task 0 adds deprecation warnings in the next tagged release; **removal gate:** legacy commands are removed in the first tagged minor release *after* a release that shipped the warnings (two-release window). If no tagged release process exists when M7 Task 0 lands, removal is blocked on the release-cadence work in `2026-07-04-adoption-dx-execution.md` Phase H3 — deprecation warnings ship regardless. Docs and examples reference canonical commands only.

## Milestones

| Group | Scope | Status | Hand-off plan |
|---|---|---|---|
| M1–M3 | Lifecycle models, repository, CLI | **DONE** (kernel) | m1-m3 plan — historical |
| M4–M5 | Scoring/calibration, cutoff integrity | **DONE** (kernel) | m4-m6 plan — historical |
| M6 | Extraction depth expansion | **DEFERRED** — gated (see Task Ordering) | m4-m6 plan — historical |
| M7 | Persistence hardening + decision-impact instrumentation (Layer 3 path) | TODO | `2026-07-04-forecast-lifecycle-m7-m8.md` |
| M8 | Public testing corpus workstream | TODO | `2026-07-04-forecast-lifecycle-m7-m8.md` |

## Completion Gates (global)

1. Tests for changed behavior are added first (red-green against *current* code — assert missing behavior, not missing modules).
2. Unit tests for changed surface pass; `uv run mypy src` passes; `uv run ruff check` passes.
3. CLI behavior remains backward compatible for the canonical command set; legacy commands only ever gain deprecation warnings.

## Definition of Done (Stacked, Mandatory)

### Layer 1: Operational MVP

Pass criteria:

1. ≥20 real forward-looking forecast questions created before outcomes are known.
2. 100% have persisted `ForecastQuestion` + ≥1 `ForecastRun` + `ForecastResolution` in the ledger.
3. 100% of runs include valid `forecast_as_of` and evidence provenance references.
4. Leakage protection verified by tests and by audit: `forecast-audit-report` (M7 Task 4) passes on the full ledger — question `created_at` precedes resolution knowledge, run evidence recompiles identically at `forecast_as_of`, no cited evidence violates cutoff rules.

**Verification is operational, not honor-system:** the audit command is the gate artifact.

### Layer 2: Quality MVP — metric contracts

Definitions (all computed by `forecast_scoring.py` surfaces):

- **Baselines:** B0 = uniform over the question's closed branch set. B1 = base-rate baseline: branch-frequency distribution over previously resolved questions in the same question family; falls back to B0 when a family has <10 resolved questions. B1 is the bar; B0 is reported for context.
- **Evaluation window:** rolling 90 days by `resolved_at`, non-overlapping, boundaries at UTC midnight quarter starts.
- **Minimum samples:** a window is scoreable at ≥25 resolved forecasts; ECE criterion binds only at ≥50 cumulative resolved (below that, `build_calibration_report`'s low-sample warning stands and Layer 2 cannot close).
- **Corpus rule:** all resolved non-draft questions in the ledger; confidential and public-corpus (M8) questions reported separately and combined.

Pass criteria:

1. Mean multiclass Brier ≤ 0.95 × B1 Brier (≥5% relative improvement) in the latest scoreable window.
2. ECE ≤ 0.10 (10-bucket, as implemented in `build_calibration_report`).
3. Top-1 accuracy and sample count reported per window (already emitted by `forecast-score-report`).
4. Across two consecutive scoreable windows: Brier does not degrade by >2% relative, ECE stays ≤ 0.10.

### Layer 3: Decision Impact MVP — implementation path exists (M7)

Pass criteria (mechanics defined in M7):

1. Forecast outputs referenced in ≥10 resolved persisted `DecisionRecord` artifacts (M7 Task 2), each with a `primary_forecast_run_id`; impact metrics computed on primary runs only (pending decisions excluded from denominators). The pre-forecast comparison window must also contain ≥10 resolved `BaselineDecisionRecord` artifacts so the impact report has a real baseline denominator.
2. ≥1 impact metric improved vs. the pre-forecast baseline window: decision hit-rate (decisions consistent with resolved outcome) or recorded avoided-loss, computed by `forecast-impact-report` (M7 Task 3).
3. Impact evidence is ledger-linked (decision → run → resolution chains), not anecdotal.
4. Measurement period ≥ one full evaluation window; baseline = decision hit-rate in the window preceding first forecast-referenced decision.

## Final Completion Rule

Unchanged: Layers 1+2+3 all complete, in order. No layer closes without its named gate artifact (audit report, score report, impact report).

## Persistence Integrity (new in v2)

- **Schema versioning:** `schema_version` (default `1`, validated on read) is added by M7 Task 1 to exactly these models in `src/engram/models/forecasting.py`: `ForecastQuestion`, `EvidenceDossier`, `ForecastRun`, `ForecastResolution`, `ForecastScore`, `CalibrationSummary`, and new `DecisionRecord`/`BaselineDecisionRecord`/`BeliefUpdate`. Nested value objects (`OutcomeBranch`, `ResolutionCriteria`, `EvidenceItem`) are versioned by their parent artifact and do not carry their own field.
- **Migration rule:** the ledger is append-only; migrations are forward-only rewrites into a new ledger directory with a verification diff (`old count == new count`, IDs preserved, scores recomputed and compared on normalized stable fields, excluding generated timestamps such as `ForecastScore.scored_at` and `CalibrationSummary.generated_at`). No in-place mutation.
- **Rollback rule:** previous ledger directory is retained until the migrated ledger passes `forecast-audit-report`; rollback = repoint `--repo`.
- **JSON↔graph compatibility:** round-trip export/import parity test is the M7 Task 1 acceptance check; graph-backed writes carry the same `schema_version`.

## Task Ordering (revised)

M6 (extraction depth) is explicitly **gated behind**: (a) M8 corpus available with acceptance checks green, (b) Layer 2 metric contracts above in force with at least one scoreable window, (c) M7 audit + decision instrumentation shipped. Rationale: extraction depth changes the evidence distribution; without the corpus and metric contracts, its effect is unmeasurable.

## Cross-System Scope

MVP is **CLI + ledger only** — no API/UI consumers in scope for Layers 1–3. The FastAPI surface gains forecast endpoints only after Layer 2 closes, under `/v1` per the auth/versioning rules in `docs/plans/2026-07-04-adoption-dx-execution.md` (Phase E). Contract tests accompany that work, not this plan.

Relationship to the July prediction-upgrade plans: this plan owns the **product lifecycle loop** (question-level forecasts, ledger, audit, impact). `2026-07-04-prediction-upgrade-execution.md` owns **model R&D** (heads, harness, gates 6–10). Shared code direction: scoring/metrics consolidate into one module used by both (tracked as M7 Task 5); the R&D harness may register `DeterministicForecastProtocol` as a head, but neither plan blocks the other.

## Risk Controls

- Lifecycle artifacts stay namespaced (ledger dirs / metadata) to avoid collision with facts.
- Explicit `forecast_as_of` required on every persisted run (enforced in CLI guards; keep test coverage).
- Extraction depth remains lowest priority until its gate opens (above).

## Public Testing Corpus (M8 — now a real workstream)

Purpose, source mix (EDGAR REIT deals, CourtListener/RECAP distressed timelines, recorder confirmation), per-deal content requirements, and 20–25 public deal targets are defined in `2026-07-04-forecast-lifecycle-m7-m8.md` (12–15 EDGAR REIT deals + 8–10 CourtListener deals, audit-clean, ≥20 questions). The corpus is no longer "not implementation scope": it has schema, acquisition scripts, fixtures, loader, and acceptance checks as executable tasks in M8. Layer 1/2 validation may not cite corpus-dependent evidence until M8 acceptance checks pass.
