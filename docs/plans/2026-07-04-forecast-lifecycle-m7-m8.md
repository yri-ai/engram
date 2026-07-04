# Forecast Lifecycle M7–M8 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** M7 — persistence hardening, lifecycle audit, and decision-impact instrumentation (the Layer 3 path). M8 — public testing corpus as an executable workstream. All tasks are **deltas against current master** (kernel already shipped); red tests assert missing *behavior*, never missing modules.

**Tech Stack:** Python 3.11+ | Pydantic v2 | Typer | pytest | existing `JsonForecastRepository` / `ForecastScorer` / `AsOfEvidenceCompiler`

**Grounding:** `src/engram/models/forecasting.py` (contracts), `src/engram/services/forecast_repository.py` (`JsonForecastRepository:26`, graph `ForecastRepository:118`), `src/engram/services/forecast_scoring.py` (`ForecastScorer`, `build_calibration_report`), `src/engram/services/as_of_evidence.py`, CLI `forecast-*` commands (`cli/main.py:579–718`), fetcher pattern `scripts/data_collection/fetch_edgar_cmbs.py`.

---

## M7 — Persistence Hardening + Decision Impact

### Task 0: Deprecate legacy CLI commands
- Add deprecation warnings (stderr + docstring) to `create-forecast-question`, `run-forecast`, `resolve-forecast`, `score-forecasts` (`cli/main.py:741–1021`), pointing to the canonical `forecast-*` set. No behavior change.
- **Tests:** extend `tests/unit/test_cli.py` — legacy command emits warning, still functions; canonical commands emit none.

### Task 1: `schema_version` + backend parity
- Add `schema_version: int = 1` to every lifecycle model in `models/forecasting.py`; `JsonForecastRepository` validates on read (unknown version → explicit `SchemaVersionError`, not silent parse).
- Migration scaffold: `engram forecast-ledger-migrate --from DIR --to DIR` — forward-only rewrite, verification diff (counts, IDs, recomputed scores equal), refuses in-place.
- Parity: `export/import` round-trip between `JsonForecastRepository` and graph `ForecastRepository` — byte-equivalent artifacts modulo ordering.
- **Tests:** `test_forecast_repository.py` additions — version validation, migrate round-trip, JSON↔graph parity on a 3-question fixture ledger.

### Task 2: `DecisionRecord` model + repository support
- `models/forecasting.py`: `DecisionRecord{schema_version, decision_id, decided_at, decision_type, forecast_run_ids: list[str] (≥1), rationale, expected_outcome_branch, realized_outcome_branch: str|None, impact_value: float|None, impact_kind: "avoided_loss"|"hit"|"miss"|None}`.
- `JsonForecastRepository`: `decisions/*.json` (append-only by ID, same temp-file/rename semantics); referenced `forecast_run_ids` must exist at write time.
- **Tests:** round-trip; dangling run reference rejected; immutability (rewrite same ID → error).

### Task 3: Decision CLI + impact report
- `forecast-decision-create` (links runs at decision time), `forecast-decision-resolve` (records realized outcome + impact value), `forecast-impact-report --repo DIR --baseline-window --measure-window`:
  - decision hit-rate per window (decisions whose `expected_outcome_branch` == linked resolution branch),
  - aggregate `avoided_loss`,
  - baseline comparison per master plan Layer 3 contract (pre-forecast window vs measurement window),
  - refuses to report with <10 resolved decision records (mirrors low-sample behavior in `build_calibration_report`).
- **Tests:** integration test builds ledger fixture → decisions → report matches hand-computed hit-rates; sample-floor refusal.

### Task 4: Lifecycle audit command (Layer 1 gate artifact)
- `forecast-audit-report --repo DIR [--spot-check N]`:
  - every question: `created_at` present and < resolution `resolved_at`; `forecast_as_of` ≤ `created_at` tolerance rule documented,
  - every run: persisted, `forecast_as_of` set, all cited evidence IDs present in dossier,
  - leakage spot check: recompile dossier via `AsOfEvidenceCompiler` at `forecast_as_of` for N sampled runs; cited evidence must be a subset of recompiled evidence (any extra citation = leakage flag),
  - output: JSON artifact + pass/fail summary; nonzero exit on failure (CI-usable).
- **Tests:** clean fixture passes; three poisoned fixtures (missing resolution, post-cutoff citation, invalid timestamp) each fail with the right flag.

### Task 5: Metrics consolidation seam
- Extract pure scoring functions from `forecast_scoring.py` into shared `src/engram/forecasting/metrics.py` if the prediction-upgrade plan's Phase 0 has landed; otherwise re-export from `forecast_scoring` and leave a tracked TODO. Either way: one Brier implementation in the codebase.
- **Tests:** existing scoring tests pass unchanged against the consolidated import path.

**M7 exit:** audit report green on the live ledger; decision loop demonstrated end-to-end on fixtures; master plan Layer 1 gate now mechanically checkable.

---

## M8 — Public Testing Corpus

### Task 1: Corpus schema + fixtures
- `docs/corpus-schema.md` + Pydantic `PublicDeal{deal_id, source_kind, evidence_docs: [{doc_id, url, published_at, retrieved_at, text_ref}], milestones: [{at, kind, description}], resolved_branch, resolved_at}` in `src/engram/models/corpus.py`.
- Two hand-built fixture deals in `tests/fixtures/corpus/` (one EDGAR REIT, one CourtListener) — small, licensed-clean text excerpts.
- **Tests:** schema validation; chronology validator (every milestone/doc timestamp ordered, `resolved_at` after all pre-resolution evidence).

### Task 2: Acquisition scripts
- `scripts/data_collection/fetch_edgar_reit.py` and `fetch_courtlistener.py` (pattern: `fetch_edgar_cmbs.py`; manifests in `data/manifests/`; raw to `data/corpus/{edgar,courtlistener}/`). Rate-limit + retry per existing fetcher conventions; document API-key envs in `.env.example`.
- **Tests:** parser golden-files on checked-in sample responses (no network in tests).

### Task 3: Corpus → lifecycle loader
- `scripts/data_collection/build_corpus_questions.py`: for each `PublicDeal`, generate `ForecastQuestion` (branch set from milestone taxonomy, `forecast_as_of` set to a documented pre-resolution milestone), compile dossier from evidence docs with `published_at` as record time, write to a corpus ledger via `JsonForecastRepository`.
- **Leakage rule:** loader refuses any evidence doc with `published_at > forecast_as_of` — reuses compiler semantics, tested with a poisoned deal.
- **Tests:** fixture deals produce valid ledger; `forecast-audit-report` (M7 Task 4) passes on the generated ledger — this is the acceptance check.

### Task 4: Corpus acceptance gate
- Target: 10–15 EDGAR REIT deals + 5–7 CourtListener deals loaded, audit-clean, ≥20 questions total (satisfies Layer 1 count on public data alone).
- Artifact: `outputs/results/corpus_acceptance_v1.json` (deal counts, question counts, audit summary) + decision doc `docs/plans/decisions/corpus-acceptance-decision.md`.
- **This gate opening is a precondition for M6 extraction-depth work** (master plan Task Ordering).

---

## Sequencing

M7 Tasks 0–1 → 2 → 3 → 4 (5 anytime after 1). M8 Task 1 can start immediately; Task 3 depends on M7 Task 4 (audit) existing. Corpus acceptance (M8 Task 4) closes last.

## Standing Rules

Same as master plan v2 completion gates. Every red test asserts a missing behavior on an existing surface. Artifacts versioned in `outputs/results/`; gate decisions in `docs/plans/decisions/`.
