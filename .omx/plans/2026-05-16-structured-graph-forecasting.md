# Structured Deal-Room Graph Forecasting Plan

## Goal

Create a graph-backed prediction system for structured real-estate deal-room data.

Given a deal folder such as `Sterling Town Center`, `Pura Vida`, or `Legacy West End`, the system must:
- extract forecast-relevant evidence from the folder's documents and workbooks,
- persist that evidence into Engram's temporal graph with source provenance,
- retrieve the graph-backed evidence for a named deal,
- rank plausible acquisition branches using `real_estate_acquisition`,
- explain which evidence supports each branch and which missing evidence would change the forecast.

The goal is achieved only when prediction no longer depends on direct folder scanning at forecast time. Folder scanning is allowed during ingestion, but the forecast itself must run from graph facts.

## How The LLM Knows It Achieved The Goal

The implementing LLM must not declare completion from code inspection alone. It knows the goal is achieved when all of the following are true:

1. `extract-structured --dry-run` on a structured sample folder emits typed evidence with source provenance.
2. `extract-structured` without `--dry-run` writes one deal entity and deterministic forecast facts to a `GraphStore`.
3. Re-running `extract-structured` on the same folder does not duplicate forecast facts.
4. `forecast-graph` loads evidence from graph facts, not from the folder path, and returns a `BranchForecast`.
5. The graph-backed forecast includes:
   - `top_branch`
   - branch scores
   - selected evidence IDs
   - missing precursor evidence
   - source paths or document locations for selected evidence
6. At least one unit or integration test proves the graph-backed path end to end using `MemoryStore`.
7. Manual smoke on real sample data produces a forecast from graph data for at least `Sterling Town Center`.
8. Verification commands pass:
   - `uv run pytest tests/unit -q`
   - `uv run mypy src`
   - targeted `uv run ruff check` on changed files

If any item above is missing, the LLM must report the exact missing item as a remaining gap rather than claiming the prediction system is working.

## Requirements Summary

Build an end-to-end path that extracts forecast evidence from structured real-estate deal-room sample data, persists that evidence into Engram's graph, and runs branch forecasts from graph evidence rather than directly from file inventory.

Current state:
- `EvidenceItem`, `BranchDefinition`, `ContextBudget`, `BranchScore`, and `BranchForecast` already define the forecast-facing contracts in `src/engram/models/branch_forecasting.py:10`.
- `real_estate_acquisition` branch definitions and filename-based evidence mapping already exist in `src/engram/services/branch_forecasting.py:54`.
- The current CLI forecast command reads JSON/NDJSON or directories directly through `evidence_from_path` in `src/engram/cli/main.py:368`.
- The graph already supports entities, relationships, temporal queries, and facts through `GraphStore` in `src/engram/storage/base.py:18`.
- Facts are the lowest-friction graph primitive for forecast evidence because they attach claims to one entity and already carry bitemporal fields plus metadata in `src/engram/models/fact.py:16`.
- Neo4j persistence for bitemporal entities/relationships is already implemented in `src/engram/storage/neo4j.py:204` and `src/engram/storage/neo4j.py:306`.

Target state:
- A structured deal folder becomes one graph-backed `Deal`/`Property` entity.
- Each source document becomes graph evidence with extracted snippets, event type, confidence/salience, source path, and timestamps.
- Forecasting can run from `GraphStore` facts for a deal entity.
- CLI supports:
  - `engram extract-structured <deal-folder> ...`
  - `engram forecast-graph <deal-name-or-id> ...`
- Tests prove extraction, graph persistence, graph retrieval, and branch forecasting work without requiring Neo4j for unit tests.

## Principles

- Preserve existing behavior: do not break the current file-backed `engram forecast` path.
- Use existing graph primitives first: prefer `Entity` + `Fact` before adding a new graph schema.
- Start with deterministic extractors and optional backends; no new dependency unless explicitly approved.
- Keep source provenance first-class: every forecast evidence item must point back to a source file and snippet.
- Treat prediction as branch ranking with evidence gaps, not a black-box label.

## Decision Drivers

- Structured sample data contains mixed PDFs, XLS/XLSX, DOCX, and deal-room folders outside this worktree.
- Existing graph persistence already has bitemporal fact support, so reusing facts is faster and lower risk than introducing new Neo4j node labels immediately.
- Forecasting quality depends more on reliable evidence extraction and provenance than on a more complex scorer right now.

## Viable Options

### Option A: Store forecast evidence as `Fact` records attached to a deal entity

Pros:
- Reuses existing `save_fact` / `get_facts` graph contract in `src/engram/storage/base.py:218`.
- Keeps bitemporal validity/record time through the existing `Fact` model in `src/engram/models/fact.py:39`.
- Works with `MemoryStore` unit tests immediately.
- Smallest schema and migration surface.

Cons:
- Forecast evidence is semantically richer than a generic fact, so metadata discipline matters.
- Querying across all deals by event type may need future indexes.

### Option B: Add explicit `ForecastEvidence` nodes and relationships

Pros:
- Cleaner domain schema for forecast evidence.
- Easier future graph traversal from document to evidence to branch.

Cons:
- Requires new GraphStore methods, Neo4j Cypher, MemoryStore storage, indexes, and more tests.
- More risk before we prove extraction quality.

### Decision

Choose Option A for the next implementation slice. Use `Entity` + `Fact` now, with a metadata schema that can be promoted to explicit nodes later.

## Proposed Graph Shape

Deal entity:
- `Entity.entity_type`: use existing `Concept` initially.
- `canonical_name`: folder/deal name, e.g. `sterling town center`.
- `metadata.kind`: `structured_deal`.
- `metadata.source_path`: absolute folder path.

Forecast evidence fact:
- `Fact.entity_id`: deal entity id.
- `Fact.fact_key`: forecast event type, e.g. `rent_roll`, `operating_statement`, `environmental_risk`.
- `Fact.fact_text`: extracted snippet or document-derived evidence statement.
- `Fact.confidence`: extraction confidence/salience.
- `Fact.valid_from` / `recorded_from`: ingestion timestamp or source timestamp when known.
- `Fact.metadata`:
  - `source_path`
  - `relative_path`
  - `document_type`
  - `extractor`
  - `page` or `sheet` when available
  - `snippet`
  - `tokens`
  - `forecast_event_type`

Conversion rule:
- `Fact -> EvidenceItem` maps `fact_key` to `event_type`, `fact_text` to `text`, `confidence` to `salience`, and metadata through unchanged.

## Implementation Steps

1. Add extraction contracts.
   - Create `src/engram/models/structured.py`.
   - Add `StructuredDocument`, `ExtractedEvidence`, and `StructuredIngestionResult`.
   - Make contracts narrow and serializable; do not include parser-specific objects.
   - Acceptance: Pydantic/dataclass contract tests cover PDF/XLSX/DOCX/file inventory records.

2. Add deterministic document inventory and text extraction service.
   - Create `src/engram/services/structured_extraction.py`.
   - Move/extend file inventory logic currently in `evidence_from_directory` at `src/engram/services/branch_forecasting.py:208`.
   - Implement stdlib extractors:
     - `.xlsx`: parse zipped workbook XML/shared strings enough to extract sheet text and sheet names.
     - `.docx`: parse zipped `word/document.xml` text.
     - `.pdf`: start with metadata/filename classification plus optional `pdftotext` backend if available; no hard dependency.
   - Keep unsupported files as inventory evidence with lower confidence.
   - Acceptance: tests use tiny generated DOCX/XLSX zip fixtures and fake PDF filenames; extraction returns stable event types and provenance.

3. Add graph ingestion service.
   - Create `src/engram/services/structured_graph.py`.
   - Responsibilities:
     - create/upsert one deal entity per folder
     - convert extracted evidence into `Fact` records
     - save facts through `GraphStore.save_fact`
     - make ingestion idempotent by deterministic fact IDs based on tenant, deal, relative path, event type, and snippet hash
   - Acceptance: `MemoryStore` test ingests a temp deal folder and verifies one deal entity plus forecast facts.

4. Add graph evidence retrieval.
   - Add helper service method `load_forecast_evidence_from_graph(store, tenant_id, deal_id, fact_keys=None) -> list[EvidenceItem]`.
   - Use existing `GraphStore.get_facts` from `src/engram/storage/base.py:223`.
   - Keep conversion in one place so file-backed and graph-backed forecasts share `BranchForecaster`.
   - Acceptance: test stores facts, loads `EvidenceItem`s, and forecasts `real_estate_acquisition`.

5. Add CLI commands.
   - Extend `src/engram/cli/main.py`.
   - Add:
     - `extract-structured PATH --deal-name ... --tenant-id ... --conversation-id ...`
     - `forecast-graph DEAL --objective ... --structural-family real_estate_acquisition`
   - `extract-structured` should support `--dry-run` to print extracted evidence without writing graph.
   - `forecast-graph` should initialize `Neo4jStore`, resolve the deal entity, load facts, and call `BranchForecaster`.
   - Acceptance: CLI tests monkeypatch store/services and verify both dry-run and graph forecast behavior.

6. Add a repeatable sample-data smoke script.
   - Create `scripts/data_collection/run_structured_graph_forecast.py`.
   - Inputs:
     - structured data root
     - deal folder name
     - objective
     - output path
   - Output JSON should include ingestion counts, top branch, branch scores, selected evidence, and evidence gaps.
   - Acceptance: script can run against `Sterling Town Center`, `Pura Vida`, and `Legacy West End` when data is available.

7. Add integration tests gated from normal unit tests.
   - Add tests under `tests/integration/test_structured_graph_forecast.py`.
   - Use `MemoryStore` for graph integration by default.
   - Add optional Neo4j test only under the existing integration marker.
   - Acceptance: `uv run pytest tests/unit -q` stays fast; `uv run pytest tests/integration -q -m integration` validates graph-backed path when services are available.

8. Calibrate the first real-estate acquisition schema.
   - Create a small expected-outcome fixture file in `tests/fixtures/structured_forecasts.json`.
   - Start with smoke expectations:
     - Sterling Town Center should have enough core docs to support `advance_diligence`.
     - Pura Vida and Legacy West End should produce mixed evidence with `reprice_or_restructure` currently plausible.
   - Acceptance: snapshot-style tests assert top branch and evidence gaps for deterministic fixture folders or mocked extracted evidence.

## Acceptance Criteria

- `extract-structured --dry-run` returns at least 5 typed evidence items for `Sterling Town Center`.
- Graph ingestion creates exactly one deal entity per ingested folder and deterministic facts for extracted evidence.
- Re-running graph ingestion against the same folder does not duplicate forecast facts.
- `forecast-graph` returns the same top branch as file-backed forecasting for the same extracted evidence.
- Every selected evidence item in a graph-backed forecast includes `source_path` and either a snippet, sheet, page, or file inventory statement.
- Unit tests cover JSON, directory inventory, DOCX text extraction, XLSX text extraction, fact conversion, MemoryStore graph ingestion, and graph-backed forecasting.
- Verification commands pass:
  - `uv run pytest tests/unit -q`
  - `uv run mypy src`
  - targeted `uv run ruff check` on changed files

## Risks And Mitigations

- Risk: PDF extraction without new dependencies is weak.
  - Mitigation: start with filename/document-type evidence for PDFs; add optional `pdftotext` backend; ask before adding a hard dependency such as `pypdf`.

- Risk: Existing `Fact` model may become overloaded.
  - Mitigation: keep forecast metadata schema strict and isolated; if graph queries become awkward, promote to `ForecastEvidence` nodes in a later migration.

- Risk: Filename-derived evidence can overstate confidence.
  - Mitigation: assign lower salience to inventory-only PDF evidence; assign higher salience only when extracted text/sheets contain matching terms.

- Risk: Real sample data lives outside the worktree.
  - Mitigation: commands accept absolute paths; tests use temp fixtures so CI does not depend on local data.

- Risk: Branch scores remain heuristic.
  - Mitigation: this phase focuses on data flow; add calibration/backtest after graph-backed forecasts are stable.

## Verification Plan

Unit:
- `tests/unit/test_structured_extraction.py`
- `tests/unit/test_structured_graph.py`
- Existing `tests/unit/test_branch_forecasting.py`
- Existing `tests/unit/test_cli.py`

Integration:
- MemoryStore end-to-end:
  - extract temp structured folder
  - save deal entity/facts
  - load graph evidence
  - forecast branch
- Optional Neo4j:
  - initialize schema
  - ingest one small fixture folder
  - query facts and forecast

Manual smoke:

```bash
uv run engram extract-structured "/Users/leonardlangsdorf/dev/projects/engram/data/structured/Sterling Town Center" \
  --deal-name "Sterling Town Center" \
  --tenant-id default \
  --conversation-id structured-sterling \
  --dry-run

uv run engram extract-structured "/Users/leonardlangsdorf/dev/projects/engram/data/structured/Sterling Town Center" \
  --deal-name "Sterling Town Center" \
  --tenant-id default \
  --conversation-id structured-sterling

uv run engram forecast-graph "Sterling Town Center" \
  --objective "acquisition diligence risk" \
  --structural-family real_estate_acquisition \
  --tenant-id default
```

## Execution Order

1. Contracts and deterministic extraction.
2. Graph ingestion using MemoryStore tests.
3. Graph retrieval to `EvidenceItem`.
4. CLI dry-run and graph-backed commands.
5. Real sample smoke tests.
6. Optional Neo4j integration verification.
7. Calibration fixtures.

## Done Definition

This phase is complete when the system can ingest a structured deal folder into the graph and produce a branch forecast from graph facts with provenance-backed evidence and reproducible tests.
