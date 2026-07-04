# Temporal Forecasting Kernel

Status: MVP foundation implemented in `wf/temporal-forecasting-kernel-20260628`.

## Thesis

Engram's prediction layer is not a RAG app. The product thesis is a **bitemporal probabilistic forecasting kernel**:

```text
Bitemporal graph
  → as-of evidence state
  → forecast question / resolution contract
  → structured forecasting protocol
  → probabilistic forecast run
  → immutable forecast ledger
  → resolution + proper scoring
  → calibration reporting over time
```

The kernel answers:

> Given only what Engram knew at timestamp T, what probability should it assign to outcome Y by horizon H, and why?

Evidence lookup is plumbing. The first-class product object is the immutable forecast run.

## MVP components

### Forecast contracts

`src/engram/models/forecasting.py` defines:

- `ForecastQuestion`
- `OutcomeBranch`
- `ResolutionCriteria`
- `EvidenceItem`
- `EvidenceDossier`
- `ForecastRun`
- `ForecastResolution`
- `ForecastScore`
- `CalibrationSummary`

Every forecast question requires:

- `forecast_as_of`
- `horizon`
- resolution criteria
- a closed branch set

Forecast runs are frozen/immutable. Reruns must create new run IDs.

### As-of evidence compiler

`src/engram/services/as_of_evidence.py` defines:

- `GraphEvidenceAdapter`
- `MemoryGraphEvidenceAdapter`
- `AsOfEvidenceCompiler`

The compiler enforces deterministic leakage rules:

- exclude facts/relationships with `recorded_from > forecast_as_of`
- exclude records whose source was ingested after forecast time
- exclude resolution evidence
- exclude post-hoc derived evidence
- preserve supersession state instead of deleting old claims

For the first CLI flow, `forecast-dossier-compile` also supports a JSON evidence path without Neo4j so the lifecycle can be demonstrated with a simple local ledger.

### Forecast ledger

`src/engram/services/forecast_repository.py` provides `JsonForecastRepository`.

It stores:

```text
questions/*.json
runs/*.json
resolutions/*.json
scores/*.json
```

Rules:

- draft questions can be updated
- active questions require a new ID/version
- forecast runs are append-only by ID
- resolutions must match the question branch set
- writes use temp-file then rename semantics

### Deterministic protocol

`src/engram/services/forecast_protocol.py` provides `DeterministicForecastProtocol`.

It is intentionally simple and auditable:

- starts from uniform or branch priors
- applies support/opposition evidence counts
- records missing-evidence penalty in raw scores
- converts scores to probabilities with stable softmax
- validates cited evidence IDs
- stores config snapshots and raw score metadata

This is a baseline protocol, not a claim of forecasting quality.

### Scoring and calibration reporting

`src/engram/services/forecast_scoring.py` provides:

- binary Brier score
- multiclass Brier score
- clipped log score
- top-1/top-k accuracy
- probability assigned to resolved branch
- calibration bucket assignment
- low-sample warning reports

Calibration output is provisional until enough forecasts resolve.

## CLI flow

Create an active question:

```bash
uv run engram forecast-question-create \
  --repo /tmp/forecast-ledger \
  --question-id q-demo \
  --title "Will Alice renew?" \
  --forecast-as-of 2026-01-15T00:00:00+00:00 \
  --horizon 30d \
  --resolution-criteria "Renewal is recorded by the horizon." \
  --resolved-by 2026-02-15T00:00:00+00:00 \
  --branch yes:Yes \
  --branch no:No \
  --status active
```

Prepare evidence JSON:

```json
[
  {
    "id": "e-renewal-signal",
    "text": "Alice requested renewal paperwork.",
    "valid_from": "2026-01-15T00:00:00+00:00",
    "recorded_from": "2026-01-15T00:00:00+00:00",
    "source_id": "source-1",
    "supports_branch": ["yes"],
    "supersession_status": "current_as_of"
  }
]
```

Compile a dossier:

```bash
uv run engram forecast-dossier-compile \
  --repo /tmp/forecast-ledger \
  --question-id q-demo \
  --evidence-json /tmp/evidence.json \
  --output /tmp/dossier.json
```

Create a run:

```bash
uv run engram forecast-run-create \
  --repo /tmp/forecast-ledger \
  --question-id q-demo \
  --dossier /tmp/dossier.json \
  --run-id run-q-demo \
  --output /tmp/run.json
```

Resolve the question:

```bash
uv run engram forecast-resolve-create \
  --repo /tmp/forecast-ledger \
  --question-id q-demo \
  --resolved-branch yes \
  --resolved-at 2026-02-15T00:00:00+00:00 \
  --evidence-id resolution-source
```

Build a provisional scoring report:

```bash
uv run engram forecast-score-report \
  --repo /tmp/forecast-ledger \
  --bucket-count 5 \
  --low-sample-threshold 30 \
  --output /tmp/report.json
```

## What is deliberately not solved yet

- Neo4j-backed forecast repository
- production migrations/auth/secrets
- LLM forecaster protocol
- learned calibration
- temporal graph neural forecasting
- scenario trees
- continuous numeric forecasting
- claims that probabilities are calibrated

## Next steps (executable)

> Status doc note: this section was previously a wish list. It is now the execution
> contract for kernel follow-on work. Ordering is top-to-bottom; each task states
> touched files, tests, and done criteria. Persistence changes follow the
> schema-version/migration rules in the master plan v2 (`2026-05-18-forecast-lifecycle-master-plan.md`,
> "Persistence Integrity") — no task below may change artifact shape without a
> `schema_version` bump and forward-only ledger migration.

### NS-1: Neo4j adapter for `GraphEvidenceAdapter`

- **Files:** create `Neo4jGraphEvidenceAdapter` in `src/engram/services/as_of_evidence.py`
  (or `as_of_evidence_neo4j.py` if >150 lines); reuse read paths from
  `src/engram/storage/neo4j.py` (`query_knowledge_as_of` and fact queries).
- **Behavior contract:** must pass the same leakage tests as `MemoryGraphEvidenceAdapter`
  — parametrize `tests/unit/test_as_of_evidence.py` fixtures over both adapters
  (memory in unit lane, Neo4j in `tests/integration/` behind the existing
  integration marker).
- **Done:** parametrized suite green on both adapters; excluded_counts parity
  on the shared fixture graph; no future-recorded evidence in any dossier.

### NS-2: Persist evidence dossiers in the ledger

- **Files:** `JsonForecastRepository` gains `dossiers/` dir + `save_dossier`/
  `load_dossier`/`list_dossiers` (same temp-file semantics; exclusive create like
  runs — a dossier is immutable once a run cites it); `forecast-dossier-compile`
  CLI gains `--repo` write-through; `ForecastRun.dossier_id` becomes a checked
  reference at `save_run` (dangling → `ValueError`).
- **Migration:** additive (new directory) — no `schema_version` bump; existing
  ledgers remain valid. Rollback: ignore `dossiers/`.
- **Tests:** `tests/unit/test_forecast_repository.py` — round-trip, immutability,
  dangling `dossier_id` rejection; CLI test for write-through.
- **Done:** a run's evidence is reconstructable from the ledger alone (no live
  graph needed for audit), which unblocks the M7 audit command's spot checks.

### NS-3: Prompt/model protocol versions for LLM-assisted forecasting

- **Files:** `src/engram/services/forecast_protocol.py` — extract a
  `ForecastProtocol` Protocol (create_run signature); add `LLMForecastProtocol`
  with `protocol="llm.v1"`, prompt template in `config/prompts/forecast_llm.jinja2`,
  provider via existing `engram.llm.provider`; `protocol_config` records
  model name, prompt hash, temperature.
- **Guardrails carried over:** branch-set guard, audit-mode dossier refusal, and
  citation validation are shared via the base — test them against BOTH protocols
  (parametrize existing protocol tests).
- **Tests:** mocked-LLM determinism (existing `test_llm_provider.py` pattern);
  malformed-LLM-output → explicit error, never a silent uniform fallback.
- **Done:** `forecast-run-create --protocol llm.v1` produces a scoreable,
  provenance-complete run; H4 re-ablation hook noted for prediction plan Phase 3.

### NS-4: Real-estate diligence demo

- **Files:** `examples/diligence-demo/` — timestamped evidence JSON (fixture-safe,
  no confidential data; reuse M8 corpus fixtures when available), `run_demo.sh`
  driving question→dossier→run→resolve→score with the canonical `forecast-*` CLI.
- **Tests:** e2e smoke test in `tests/e2e/` (JSON evidence path — no Neo4j/Redis).
- **Done:** demo runs green in CI; README links it as the kernel quickstart.

### NS-5: Prequential belief updates

- **Files:** `src/engram/models/forecasting.py` — `BeliefUpdate` (frozen,
  `schema_version=1`: run_id refs prior/posterior, trigger evidence IDs,
  update_at); repository `updates/` dir (exclusive create); scoring:
  `forecast_scoring.py` gains prequential update-quality metrics (did updates
  move probability toward the resolved branch — per-update log-score delta).
- **Ordering:** requires NS-2 (dossier persistence) so update evidence is auditable.
- **Tests:** update chain integrity (posterior run must exist, timestamps
  monotonic); update-quality metric on a hand-computed fixture.
- **Done:** `forecast-score-report` includes update-quality section when updates exist.

### Deferred (explicitly out of this doc)

- **API endpoints:** out of scope until Layer 2 closes, per master plan v2
  "Cross-System Scope" (CLI + ledger only for MVP). When opened, endpoints land
  under `/v1` with the auth rules in `2026-07-04-adoption-dx-execution.md` Phase E.
- **Learned calibration / scenario trees / TKG forecasting:** owned by
  `2026-07-04-prediction-upgrade-execution.md` (Phases 2–3); the kernel consumes
  those as protocol implementations, not as kernel work.
