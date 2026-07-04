# Forecast Lifecycle M7–M8 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** M7 — persistence hardening, lifecycle audit, and decision-impact instrumentation (the Layer 3 path). M8 — public testing corpus as an executable workstream. All tasks are **deltas against current master** (kernel already shipped); red tests assert missing *behavior*, never missing modules.

**Tech Stack:** Python 3.11+ | Pydantic v2 | Typer | pytest | existing `JsonForecastRepository` / `ForecastScorer` / `AsOfEvidenceCompiler`

**Grounding:** `src/engram/models/forecasting.py` (contracts), `src/engram/services/forecast_repository.py` (`JsonForecastRepository:26`, graph `ForecastRepository:145`), `src/engram/services/forecast_scoring.py` (`ForecastScorer`, `build_calibration_report`), `src/engram/services/as_of_evidence.py`, CLI `forecast-*` commands (`src/engram/cli/main.py:579–718`), fetcher pattern `scripts/data_collection/fetch_edgar_cmbs.py`.

---

## M7 — Persistence Hardening + Decision Impact

### Task 0: Deprecate legacy CLI commands
- Add deprecation warnings (stderr + docstring) to `create-forecast-question`, `run-forecast`, `resolve-forecast`, `score-forecasts` (`src/engram/cli/main.py:741–1021`), pointing to the canonical `forecast-*` set. No behavior change.
- **Tests:** extend `tests/unit/test_cli.py` — legacy command emits warning, still functions; canonical commands emit none.

### Task 1: `schema_version`, dossier persistence, backend parity
- Add `schema_version: int = 1` to exactly: `ForecastQuestion`, `EvidenceDossier`, `ForecastRun`, `ForecastResolution`, `ForecastScore`, `CalibrationSummary` (+ `DecisionRecord`/`BeliefUpdate` when created). Nested value objects (`OutcomeBranch`, `ResolutionCriteria`, `EvidenceItem`) are versioned by their parent. `JsonForecastRepository` validates on read (unknown version → explicit `SchemaVersionError`, not silent parse).
- **Dossier persistence (prerequisite for Task 4 audit; = kernel plan NS-2):** `JsonForecastRepository` gains `dossiers/` dir + `save_dossier`/`load_dossier`/`list_dossiers` (exclusive create — immutable once a run cites one); `forecast-dossier-compile` writes through to `--repo`. Preserve current `--output` behavior for backward compatibility: after dossier persistence lands, `--output` becomes optional; when supplied, the command writes both the ledger dossier and the file output. `save_run` rejects a `dossier_id` not present in the ledger.
- **Resolution uniqueness:** `JsonForecastRepository.save_resolution` rejects a second resolution for the same `question_id` unless a future explicit correction model is added. M7 does not implement correction records; duplicate question resolutions are errors so scoring and impact reports have one outcome per question.
- Migration scaffold: `engram forecast-ledger-migrate --from DIR --to DIR` — forward-only rewrite, verification diff (counts, IDs, recomputed scores equal), refuses in-place. Additive `dossiers/` dir needs no version bump. Rollback: migration never mutates `--from`; if target verification fails, discard `--to` and keep using the original repo path.
- Parity — **scoped:** graph `ForecastRepository` supports questions/runs/resolutions only; scores/dossiers/decisions are JSON-ledger-only (gap documented per release, per master plan v2). Parity test: normalized export (`model_dump(mode="json")`, sorted keys) of the supported subset round-trips JSON↔graph without semantic loss — field-normalized comparison, not byte equality.
- **Tests:** `test_forecast_repository.py` additions — version validation, dossier round-trip/immutability, dangling `dossier_id` rejection, duplicate-resolution rejection, migrate round-trip, scoped JSON↔graph parity on a 3-question fixture ledger.

### Task 2: forecast-linked and baseline decision models + repository support
- Modify `src/engram/models/forecasting.py`: `DecisionRecord{schema_version, decision_id, decided_at, decision_type, primary_forecast_run_id: str, supporting_forecast_run_ids: list[str] = [], rationale, expected_outcome_branch, realized_outcome_branch: str|None, impact_value: float|None, impact_kind: "avoided_loss"|"hit"|"miss"|None}`.
- Add baseline-only model in `src/engram/models/forecasting.py`: `BaselineDecisionRecord{schema_version, decision_id, decided_at, decision_type, rationale, expected_outcome_branch, realized_outcome_branch: str|None, impact_value: float|None, impact_kind: "avoided_loss"|"hit"|"miss"|None}`. Baseline records deliberately have no forecast-run IDs; they represent the pre-forecast comparison window required by the master plan.
- **Hit-rate semantics (unambiguous):** forecast-linked impact metrics are computed on `DecisionRecord.primary_forecast_run_id` only — one primary run → one question → one repository-enforced resolution. `supporting_forecast_run_ids` are provenance context, never scored. Validation: primary must not appear in supporting list; `expected_outcome_branch` must be a branch of the primary run's question. Baseline hit-rate uses `BaselineDecisionRecord.expected_outcome_branch == BaselineDecisionRecord.realized_outcome_branch`.
- **Resolution states for reporting:** a forecast-linked decision is *resolved* when its primary run's question has a `ForecastResolution`; unresolved forecast-linked decisions are excluded from the hit-rate denominator and reported as `pending_forecast_linked_count`. Baseline records with `realized_outcome_branch is None` are excluded from the baseline denominator and reported as `pending_baseline_count`.
- `JsonForecastRepository`: `decisions/*.json` and `baseline_decisions/*.json` (exclusive create by ID). Forecast-linked records must reference existing run IDs at write time; baseline records must not reference forecast runs. Resolution is a guarded one-time update: `resolve_decision`/`resolve_baseline_decision` may fill `realized_outcome_branch`, `impact_value`, and `impact_kind` only when those fields are currently unset; a second resolve attempt fails.
- **Tests:** forecast-linked round-trip; baseline round-trip; dangling primary/supporting reference rejected; primary-in-supporting rejected; expected-branch-not-in-question rejected; exclusive create; guarded one-time resolve; pending-vs-resolved classification for both record types.

### Task 3: Decision CLI + impact report
- `forecast-decision-create` (links primary + supporting runs at decision time), `forecast-decision-resolve` (records realized outcome + impact value), `forecast-baseline-decision-create` (records pre-forecast baseline decisions without forecast links), `forecast-impact-report --repo DIR --baseline-window --measure-window`:
  - decision hit-rate per window over *resolved* records only (Task 2 semantics: forecast-linked records compare `expected_outcome_branch` to the resolution branch of the primary run's question; baseline records compare expected vs realized directly; pending records reported separately, never in denominators),
  - aggregate `avoided_loss`,
  - baseline comparison per master plan Layer 3 contract (pre-forecast window vs measurement window),
  - refuses to report when either comparison side has <10 resolved records (mirrors low-sample behavior in `build_calibration_report`).
- **Tests:** integration test builds ledger fixture → baseline decisions + forecast-linked decisions → report matches hand-computed hit-rates; sample-floor refusal on either side.

### Task 4: Lifecycle audit command (Layer 1 gate artifact)
- **Depends on Task 1 dossier persistence.**
- `forecast-audit-report --repo DIR [--spot-check N]`:
  - every question: `created_at` present and < resolution `resolved_at`; `forecast_as_of` ≤ `created_at` tolerance rule documented,
  - every run: persisted, `forecast_as_of` set, `dossier_id` resolves to a persisted dossier, all cited evidence IDs present in that dossier,
  - **runs predating dossier persistence** (no ledger dossier): classified `unauditable_evidence` — reported, never counted as pass; Layer 1's "100%" criteria are computed over auditable runs and the unauditable count must be zero for the gate to close (i.e., old runs must be re-run or explicitly retired),
  - leakage spot check: for N sampled runs *with a graph-backed evidence source*, recompile via `AsOfEvidenceCompiler` at `forecast_as_of`; cited evidence must be a subset of recompiled evidence (extra citation = leakage flag). For JSON-evidence-path runs, the check compares cited IDs against the persisted dossier's items and each item's `recorded_from ≤ forecast_as_of` (self-consistency check — weaker, labeled as such in the report),
  - output: JSON artifact + pass/fail summary; nonzero exit on failure (CI-usable).
- **Tests:** clean fixture passes; poisoned fixtures (missing resolution, post-cutoff citation, invalid timestamp, missing dossier) each fail with the right flag.

### Task 5: Metrics consolidation seam
- Create: `src/engram/forecasting/metrics.py` and extract pure scoring functions from `forecast_scoring.py` if the prediction-upgrade plan's Phase 0 has landed; otherwise re-export from `forecast_scoring` and leave a tracked TODO. Either way: one Brier implementation in the codebase.
- **Tests:** existing scoring tests pass unchanged against the consolidated import path.

**M7 exit:** audit report green on the live ledger; decision loop demonstrated end-to-end on fixtures; master plan Layer 1 gate now mechanically checkable.

---

## M8 — Public Testing Corpus

### Task 1: Corpus schema + fixtures
- Create: `docs/corpus-schema.md`; Create: `src/engram/models/corpus.py` with Pydantic `PublicDeal{deal_id, source_kind, evidence_docs: [{doc_id, url, published_at, retrieved_at, text_ref, summary}], milestones: [{at, kind, description}], resolved_branch, resolved_at, branch_taxonomy_id = "public_real_estate_milestone_v1"}` and `CorpusBranchTaxonomy`.
- Define the default machine-readable branch taxonomy in `src/engram/models/corpus.py` and document it in `docs/corpus-schema.md`: `public_real_estate_milestone_v1` has branch IDs `advance_or_close`, `reprice_or_restructure`, and `terminated_or_failed`. `resolved_branch` and annotation branch IDs must be members of this taxonomy; invalid branch IDs fail validation.
- Create: two hand-built fixture deals in `tests/fixtures/corpus/` (one EDGAR REIT, one CourtListener) — small, licensed-clean text excerpts.
- **Tests:** Create: `tests/unit/test_corpus_models.py` — schema validation; branch-taxonomy validation; chronology validator (every milestone/doc timestamp ordered, `resolved_at` after all pre-resolution evidence).

### Task 2: Acquisition scripts
- Create: `scripts/data_collection/fetch_edgar_reit.py` and Create: `scripts/data_collection/fetch_courtlistener.py` (pattern: existing `fetch_edgar_cmbs.py`; manifests in `data/manifests/`; raw to `data/corpus/{edgar,courtlistener}/`). Rate-limit + retry per existing fetcher conventions; document API-key envs in `.env.example`.
- **Tests:** parser golden-files on checked-in sample responses (no network in tests).

### Task 3: Corpus → lifecycle loader
- Create: `scripts/data_collection/build_corpus_questions.py`: for each `PublicDeal`, generate `ForecastQuestion` (branch set from `public_real_estate_milestone_v1`, `forecast_as_of` set to a documented pre-resolution milestone), compile a dossier, persist question + dossier to a corpus ledger via `JsonForecastRepository` (dossier persistence from M7 Task 1).
- **`PublicDeal` → `EvidenceItem` field mapping (complete):**

| EvidenceItem field | Source |
|---|---|
| `id` | `f"corpusdoc:{deal_id}:{doc_id}"` (deterministic) |
| `text` | `evidence_docs[i].summary` (curated excerpt; `text_ref` retained in metadata for full text) |
| `valid_from` | earliest milestone `at` the doc evidences, else `published_at` |
| `valid_to` | `None` (public docs assert states, not closed intervals, at MVP) |
| `recorded_from` | `published_at` (public knowledge time) |
| `recorded_to` | `None` |
| `source_id` | `doc_id` |
| `source_span` | `text_ref` locator when available |
| `supports_branch` / `opposes_branch` | from a per-deal curated annotation file (`tests/fixtures/corpus/{deal_id}.annotations.json`); loader runs without annotations but flags the dossier `unannotated` (protocol support counts are then zero — priors-only forecast) |
| `supersession_status` | `"current_as_of"` (corpus docs don't supersede at MVP; amendments modeled as new docs) |
| `supersedes_id` / `superseded_by_id` | `None` |
| `confidence` | `None` |
| `metadata` | `{deal_id, source_kind, url, retrieved_at, text_ref}` |

- **Leakage rule:** loader refuses any evidence doc with `published_at > forecast_as_of` — tested with a poisoned deal.
- **Tests:** fixture deals produce valid ledger (mapping asserted field-by-field on one doc); `forecast-audit-report` (M7 Task 4) passes on the generated ledger — this is the acceptance check.

### Task 4: Corpus acceptance gate
- Target: 12–15 EDGAR REIT deals + 8–10 CourtListener deals loaded, audit-clean, ≥20 public deals and ≥20 questions total (satisfies Layer 1 count on public data alone even if each deal creates one question).
- Artifact (generated, does not exist yet): `outputs/results/corpus_acceptance_v1.json` (deal counts, question counts, audit summary) + Create: `docs/plans/decisions/corpus-acceptance-decision.md`.
- **This gate opening is a precondition for M6 extraction-depth work** (master plan Task Ordering).

---

## Sequencing

M7 Tasks 0–1 → 2 → 3 → 4 (5 anytime after 1; Task 4 additionally requires Task 1's dossier persistence). M8 Task 1 can start immediately; M8 Task 3 depends on M7 Task 1 (dossier persistence) and its acceptance check depends on M7 Task 4 (audit). Corpus acceptance (M8 Task 4) closes last.

## Standing Rules

Same as master plan v2 completion gates. Every red test asserts a missing behavior on an existing surface. Artifacts versioned in `outputs/results/`; gate decisions in `docs/plans/decisions/`.
