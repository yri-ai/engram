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

## Next steps

1. Add a Neo4j adapter for `GraphEvidenceAdapter`.
2. Persist evidence dossiers directly in the forecast ledger or graph.
3. Add API endpoints around the same lifecycle.
4. Add prompt/model protocol versions for LLM-assisted forecasting.
5. Build a real-estate diligence demo using timestamped private evidence.
6. Add prequential belief update objects and update-quality scoring.
