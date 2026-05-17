# Open-Source LLM Prediction Platform Phase

## Goal

Turn Engram from a temporal context graph plus branch-forecasting prototype into a usable open-source LLM prediction platform.

The platform must ingest messy, time-stamped evidence; store it in a temporal graph; retrieve only decision-relevant evidence under an explicit cutoff date; produce auditable probabilistic forecasts; and evaluate those forecasts after resolution.

This phase is not complete when the system merely prints a top branch. It is complete only when Engram can produce, store, retrieve, score, and explain forecasts end to end.

## Product Thesis

Engram should not compete as a generic "ask an LLM to predict anything" chatbot.

The defensible product is:

> An open-source temporal graph forecasting platform that turns private, messy, time-stamped evidence into auditable branch forecasts with provenance, uncertainty, and measurable calibration.

This positions Engram between:
- generic LLM forecasting bots, which often lack persistent temporal evidence and source-grounded audit trails,
- prediction market platforms, which handle community forecasts but not private document-grounded evidence graphs,
- document RAG systems, which retrieve context but generally do not produce scored, resolvable forecasts.

## How The LLM Knows It Achieved The Goal

The implementing LLM must not claim this phase is complete unless all of the following are true:

1. A user can ingest at least one structured sample deal folder into graph-backed forecast evidence.
2. A user can create a forecast question with:
   - objective,
   - structural family,
   - forecast horizon,
   - cutoff/as-of timestamp,
   - resolution criteria.
3. A graph-backed forecast can run using only evidence available at or before the cutoff timestamp.
4. The forecast output includes:
   - top branch,
   - probability distribution across branches,
   - selected evidence,
   - source provenance,
   - missing evidence gaps,
   - branch rationale.
5. Forecast outputs are persisted as forecast runs, not just printed.
6. A forecast can be resolved with an observed outcome.
7. At least one scoring command/report computes Brier score and calibration metrics over resolved forecasts.
8. Unit tests prove:
   - extraction contracts,
   - graph persistence,
   - cutoff filtering,
   - probabilistic branch output,
   - forecast persistence,
   - resolution scoring.
9. A manual smoke run on real sample data produces a stored graph-backed forecast and a stored forecast report.
10. Verification passes:
   - `uv run pytest tests/unit -q`
   - `uv run mypy src`
   - targeted `uv run ruff check` on changed files.

If any item is missing, the LLM must report the exact missing item as a remaining gap.

## Phase Requirements

### 1. Forecast Questions

Add a canonical forecast question model.

Required fields:
- `id`
- `tenant_id`
- `deal_id` or target entity id
- `objective`
- `structural_family`
- `created_at`
- `forecast_as_of`
- `horizon`
- `resolution_due_at`
- `resolution_criteria`
- `allowed_branch_names`
- `metadata`

Why this matters:
- ForecastBench-style evaluation depends on stable question definitions and resolution windows.
- Without a forecast question, branch forecasts are unscored one-off opinions.

### 2. Forecast Runs

Persist every forecast run.

Required fields:
- `id`
- `question_id`
- `model_or_engine`
- `created_at`
- `forecast_as_of`
- `branch_probabilities`
- `top_branch`
- `selected_evidence_ids`
- `evidence_gaps`
- `rationale`
- `config`
- `metadata`

Why this matters:
- A platform needs auditability and reproducibility.
- Scoring requires immutable historical forecasts.

### 3. Forecast Resolution

Add a resolution model.

Required fields:
- `question_id`
- `resolved_at`
- `outcome_branch`
- `outcome_probability_target` when applicable
- `resolution_notes`
- `resolved_by`
- `source`

Why this matters:
- Forecasts do not become useful until they resolve.
- Resolution is what enables Brier scoring, calibration, and model improvement.

### 4. Probabilistic Branch Forecasts

Upgrade branch output from normalized scores to explicit probabilities.

Requirements:
- `BranchForecast` must include `probabilities: dict[str, float]`.
- Probabilities must sum to 1.0 within tolerance.
- Top branch must equal max probability.
- Scoring must use probabilities, not raw heuristic scores.

Initial method:
- deterministic softmax over branch scores with configurable temperature.

Later method:
- calibrated probabilities from historical resolution data.

### 5. Cutoff / Leakage Control

Every graph-backed forecast must declare an `as_of` timestamp.

Requirements:
- Evidence retrieval must exclude evidence recorded after `forecast_as_of`.
- Tests must include future evidence that would change the forecast if leaked.
- The graph-backed forecast must prove it did not use that future evidence.

Why this matters:
- Forecasting benchmarks and real prediction systems fail if future information leaks into the forecast.

### 6. Evidence Extraction

Build structured extraction for real sample data.

Required formats:
- directory inventory
- XLS/XLSX workbook text and sheet names
- DOCX text
- PDF metadata/filename evidence first, optional text backend later

Required provenance:
- source path
- relative path
- document type
- page/sheet/paragraph/cell range when available
- extraction backend
- snippet

Near-term rule:
- no hard new dependency without explicit approval.

Recommended later option:
- evaluate Docling for PDF/table/OCR extraction because it is open-source, supports multiple formats, and preserves document structure.

### 7. Graph Persistence

Use current graph primitives first.

Initial graph shape:
- one deal entity per structured sample folder
- forecast evidence saved as bitemporal `Fact` records
- fact metadata stores forecast-specific provenance

Required behavior:
- deterministic fact IDs
- idempotent ingestion
- no duplicate facts on rerun
- `MemoryStore` tests for all graph operations
- optional Neo4j integration test

### 8. Graph-Native Evidence Retrieval

Forecasting must retrieve graph facts, not scan folders.

Requirements:
- load evidence by deal entity
- filter by tenant
- filter by `forecast_as_of`
- optionally filter by structural family event types
- convert facts to `EvidenceItem`
- preserve provenance metadata

Research alignment:
- temporal KG forecasting research emphasizes query-relevant subgraphs and historically relevant evidence, not large undifferentiated context windows.

### 9. Forecast Scoring

Add scoring utilities and CLI/report support.

Minimum metrics:
- Brier score for branch outcome
- top-1 accuracy
- expected calibration error over branch confidence bins
- sample count

Required commands:
- `engram resolve-forecast ...`
- `engram score-forecasts ...`

Output:
- JSON report with aggregate and per-question scores.

### 10. Platform Interfaces

Minimum open-source platform surface:
- CLI commands
- Python service API
- JSON schemas/models
- example data usage docs
- reproducible smoke script

Future surface:
- FastAPI endpoints for forecast questions, runs, resolutions, and reports.

## Architecture Additions

### New Models

Suggested files:
- `src/engram/models/forecasting.py`
- `src/engram/models/structured.py`

Models:
- `ForecastQuestion`
- `ForecastRun`
- `ForecastResolution`
- `ForecastScore`
- `StructuredDocument`
- `ExtractedEvidence`
- `StructuredIngestionResult`

### New Services

Suggested files:
- `src/engram/services/structured_extraction.py`
- `src/engram/services/structured_graph.py`
- `src/engram/services/forecast_repository.py`
- `src/engram/services/forecast_scoring.py`

Responsibilities:
- structured extraction
- graph ingestion
- graph evidence retrieval
- forecast persistence
- forecast resolution
- scoring reports

### Storage Additions

Short-term:
- Use `Fact` metadata for forecast evidence.
- Use JSON files for forecast question/run/resolution repositories if graph persistence is too heavy for first pass.

Medium-term:
- Add graph-backed forecast run persistence.
- Add Neo4j indexes for forecast facts by tenant, entity, fact key, and recorded time.

## Milestones

### Milestone 1: Graph-Backed Structured Evidence

Deliver:
- structured extraction contracts
- folder to extracted evidence
- graph ingestion as facts
- graph evidence retrieval

Acceptance:
- `MemoryStore` end-to-end test ingests a temp structured folder and forecasts from graph facts.

### Milestone 2: Forecast Questions And Runs

Deliver:
- forecast question model
- forecast run model
- run persistence
- CLI to create/run forecast questions

Acceptance:
- CLI can create a forecast question and persist a forecast run.

### Milestone 3: Probabilities And Scoring

Deliver:
- branch probabilities
- forecast resolution
- Brier score
- ECE
- score report

Acceptance:
- test resolves at least three stored forecasts and computes aggregate scoring.

### Milestone 4: Cutoff Safety

Deliver:
- cutoff filtering in graph evidence retrieval
- leakage regression tests

Acceptance:
- test proves future evidence is excluded and would otherwise alter branch outcome.

### Milestone 5: Real Sample Smoke

Deliver:
- smoke script for structured sample data
- docs for running on Sterling / Pura Vida / Legacy West

Acceptance:
- one command ingests, forecasts, stores, resolves mock outcomes, and scores sample forecasts.

## Non-Goals For This Phase

- Training a custom forecasting model.
- Building a web UI.
- Supporting arbitrary domains beyond the first structured real-estate acquisition family.
- Full PDF OCR/table extraction as a hard dependency.
- Claiming calibrated performance before resolved forecasts exist.

## Risks And Mitigations

- Risk: filename and inventory evidence are too weak.
  - Mitigation: add content extraction before claiming platform-quality forecasts.

- Risk: branch probabilities are false precision.
  - Mitigation: label initial probabilities as heuristic and score them transparently.

- Risk: graph schema becomes too generic.
  - Mitigation: use strict metadata schema now; promote to dedicated forecast nodes later if needed.

- Risk: sample data lacks real outcomes.
  - Mitigation: start with mock/human-labeled outcomes, but keep resolution criteria explicit.

- Risk: users mistake forecasts for advice.
  - Mitigation: forecast reports must show evidence, gaps, uncertainty, and resolution criteria.

## Verification Commands

```bash
uv run pytest tests/unit -q
uv run mypy src
uv run ruff check src/engram/models/forecasting.py src/engram/models/structured.py src/engram/services/structured_extraction.py src/engram/services/structured_graph.py src/engram/services/forecast_repository.py src/engram/services/forecast_scoring.py
```

## Done Definition

This phase is done when Engram can run this loop:

```text
structured deal folder
→ extracted evidence
→ temporal graph facts
→ graph evidence retrieval as of cutoff
→ probabilistic branch forecast
→ persisted forecast run
→ resolution
→ Brier/calibration score report
```

The LLM must verify that exact loop with tests and at least one real sample smoke run before calling the platform usable.
