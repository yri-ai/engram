# Adoption & Developer Experience — Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Sequenced after:** `docs/plans/2026-07-04-prediction-upgrade-execution.md`
**Goal:** Close the last-mile gaps that decide whether a developer who finds Engram keeps it: MCP access, one-call context assembly, SDKs, zero-infra startup, security, cost honesty, benchmarks, and data lifecycle.

**Tech Stack:** Python 3.11+ | uv | FastAPI | Typer | Pydantic v2 | LiteLLM | pytest

---

## Codebase Ground Truth

| Asset | Location | Relevance |
|---|---|---|
| HTTP client (buried) | `src/engram/cli/main.py::EngramHTTPClient` (~lines 49–178) | Extract → public `engram.client`; basis for MCP server + SDK parity |
| REST API (11 routes) | `src/engram/api/routes.py` (`/messages`, `/entities*`, `/query/*`, `/search`, `/merge`) | MCP tools + `/context` endpoint mount here |
| Storage abstraction | `src/engram/storage/base.py::GraphStore` (complete ABC), `memory.py` (430 lines, working), `neo4j.py` | Lite mode already 80% possible |
| Dedup abstraction | `src/engram/services/dedup.py::DedupService` — Redis + in-memory impls both exist; `Settings.redis_enabled` flag exists | Lite mode needs zero new dedup code |
| Extraction pipeline | `src/engram/services/extraction.py::ExtractionPipeline` — 5 sequential LLM stages (`_extract_entities`, `_infer_relationships`, `_extract_facts`, `_extract_commitments`, `_generate_summary`) | Cost work: combine + parallelize stages |
| Prompts | `config/prompts/*.jinja2` (5 templates), loaded via `Settings.prompt_dir` | Combined-extraction prompt joins these |
| H2 context machinery | `src/engram/services/h2_context.py::profile_schema_guided`, budget logic | The `/context` endpoint productizes this |
| Settings | `src/engram/config.py::Settings` (pydantic-settings) | New flags land here |
| CLI | Typer app: `init`, `serve`, `ingest`, `query` | New commands: `mcp`, `serve --lite`, `export`, `redact` |
| Tests/CI | `tests/{unit,integration,e2e}`, ruff/mypy, GH workflows | Same bar for every task |

---

## Phase A — MCP Server (Tasks A1–A3, ~1 week) — highest leverage

### Task A1: Extract public Python client
- Move `EngramHTTPClient` from `cli/main.py` → `src/engram/client.py`; CLI imports from there (no behavior change). Add the missing surface: `ingest_message`, `search`, `point_in_time`, `evolution`, `get_facts`, `get_commitments` — mirroring `api/routes.py` 1:1.
- **Tests:** `tests/unit/test_client.py` against a mocked httpx transport; CLI tests stay green unchanged.

### Task A2: MCP server
- New `src/engram/mcp/server.py` using the official `mcp` Python SDK (add to core deps — it's light). Tools, each delegating to `engram.client`:
  - `engram_remember(text, conversation_id, group_id)` → POST /messages
  - `engram_search(query, mode)` → /search
  - `engram_context(query, token_budget)` → /context (Task B1; until then, formatted /search + relationships)
  - `engram_point_in_time(entity, as_of, mode)` → /query/point-in-time (expose all three bitemporal modes — this is the differentiator; no other memory MCP server has "what did we know on date X")
  - `engram_evolution(entity, type)` → /query/evolution
- CLI: `engram mcp` (stdio transport; `--http` for streamable HTTP). Config via env vars (existing `Settings`).
- **Tests:** `tests/integration/test_mcp_server.py` — tool schemas validate; round-trip against in-memory store.

### Task A3: MCP distribution
- README section with copy-paste config for Claude Desktop/Code, Cursor; submit to MCP registries. `docs/mcp.md` with the bitemporal query examples (the demo that makes people stay).
- **Exit criteria Phase A:** fresh machine → `uvx engram mcp` → Claude remembers and time-travels a conversation in <5 minutes.

## Phase B — Context Assembly Endpoint (Tasks B1–B2, ~1–2 weeks)

### Task B1: `/context` endpoint
- `POST /context {query, tenant_id, group_id, token_budget, as_of?}` in `api/routes.py` → new `src/engram/services/context_assembly.py`:
  1. entity resolution via existing `/search` path (embeddings + `find_similar_entities`)
  2. evidence selection via `profile_schema_guided` + H2 budget enforcement (reuse, don't rewrite)
  3. render to a single markdown block: active facts (with confidence + freshness), active relationships, open commitments, recent changes ("what changed since `as_of`" via existing snapshot deltas)
  4. hard token cap (tiktoken count), truncation order: lowest decayed-confidence first (reuse `temporal.py` decay)
- Response: `{context: str, token_count: int, sources: [message_ids], omitted_count: int}`.
- **Tests:** budget never exceeded (property test over random graphs); `as_of` variant returns period-correct facts (bitemporal fixture from existing `test_temporal.py` patterns); deterministic for fixed graph.

### Task B2: Wire into MCP + client + docs
- `engram_context` MCP tool switches to real endpoint; client + both SDKs get `get_context()`; README quickstart rewritten around it ("one call before your LLM call").

## Phase C — Zero-Infra Lite Mode (Tasks C1–C3, ~1–2 weeks)

### Task C1: `engram serve --lite`
- Wiring only — pieces exist: `--lite` selects `storage/memory.py` + in-memory `DedupService` (`redis_enabled=False` path) + no Neo4j check in `deps.py`. Add `Settings.storage_backend: "neo4j" | "memory" | "sqlite"`.
- Optional persistence for lite: snapshot the in-memory store to a JSON file on shutdown / interval (`--data-file engram.json`). Honest positioning: dev/prototyping mode, not production.
- **Tests:** e2e test boots lite server, ingests `examples/coaching-demo.json`, queries — no Docker services in the test env (this becomes the *fast* CI e2e lane).

### Task C2: SQLite-backed store (durable lite)
- `src/engram/storage/sqlite.py` implementing `GraphStore`: entities/relationships/facts tables with the 4 bitemporal columns indexed; vector search via `sqlite-vec` extension; graph traversals are ≤2 hops in current query patterns (verify against `base.py` ABC — every method is entity-anchored, no deep traversal), so relational is fine at this tier.
- **Tests:** run the *entire existing* storage test suite against sqlite backend (parametrize `test_memory_store.py` fixtures over backends — instant conformance suite).

### Task C3: Packaging
- Publish to PyPI (`hatchling` build already configured); `pip install engram-graph[lite]` works with zero services. Release workflow in `.github/workflows/` (tag → build → publish, trusted publishing).
- **Exit criteria Phase C:** `pip install → engram serve --lite → curl /context` in under 2 minutes, no Docker.

## Phase D — Cost & Latency (Tasks D1–D4, ~2 weeks)

### Task D1: Instrument first
- Add per-stage token/latency/cost capture in `ExtractionPipeline` (LiteLLM returns usage; sum into `ExtractionRun` — model already exists in `models/run.py`). Emit in `IngestResponse` + structured logs.
- **Artifact:** `docs/cost.md` with measured cost-per-message for gpt-4o-mini, Haiku, Ollama llama3 on the coaching demo. Publish real numbers; this is trust.

### Task D2: Combined single-call extraction mode
- New prompt `config/prompts/combined_extraction.jinja2` producing all five artifact types in one schema-validated JSON call; `Settings.extraction_mode: "staged" | "combined"`. Pipeline branch in `process_message` — downstream conflict-resolution/persistence code unchanged (it consumes parsed artifacts, not LLM calls).
- **Tests:** golden-set comparison staged vs combined on 30 fixture messages: entity/relationship/fact F1 within 5% of staged; cost reduction ≥60% measured via D1 instrumentation. Ship combined as default for lite mode if it passes.

### Task D3: Parallelize the staged mode
- `_extract_facts`, `_extract_commitments`, `_generate_summary` don't consume each other's output (verify against `process_message` flow) → `asyncio.gather` after entities+relationships. Target: wall-clock ≈ 2 sequential calls instead of 5.
- **Tests:** existing extraction tests pass; latency assertion on mocked-LLM timing fixture.

### Task D4: Async ingestion queue
- `POST /messages?mode=async` → 202 + run_id; background worker (FastAPI `BackgroundTasks` first; document the Redis-queue upgrade path). `GET /runs/{run_id}` for status (persist via existing `save_run`).
- **Tests:** integration test polls to completion; dedup holds under concurrent submits of the same message_id.

## Phase E — Security Baseline (Tasks E1–E2, ~1 week)

### Task E1: API-key auth + tenant binding
- `Settings.api_keys: dict[str, str]` (key → tenant_id) or `ENGRAM_API_KEY` single-tenant default. FastAPI dependency in `api/deps.py`: resolve tenant from key; **remove `tenant_id` as a free request parameter** on all 11 routes (breaking change → version API `/v1/`, keep old routes one release with deprecation warning). `--lite` defaults to auth-off with a startup warning.
- **Tests:** cross-tenant access attempt returns 403 with correct key isolation; every route covered (parametrized over the router table).

### Task E2: Secrets hygiene
- Startup refuses default Neo4j password unless `--lite`/dev flag; `.env.example` audit; SECURITY.md with reporting policy.

## Phase F — Benchmarks & Proof (Tasks F1–F2, ~2–3 weeks, parallel with D/E)

### Task F1: Memory benchmark harness
- `scripts/benchmarks/run_longmemeval.py` (+ LoCoMo): adapter mapping benchmark conversations → ingest, questions → `/context` + answer LLM. Reuse the prediction plan's scoreboard artifact format (`outputs/results/benchmark_*.json`).
- **Honesty rule:** publish whatever the numbers are, with configs pinned. A 70% with reproducible setup beats an uncited 94.8%.

### Task F2: README + site table
- Replace the Graphiti-cited numbers with Engram's own; comparison table gains a "measured by us, repo-reproducible" column: `uv run python scripts/benchmarks/run_longmemeval.py`.

## Phase G — Data Lifecycle (Tasks G1–G3, ~2 weeks)

### Task G1: Redaction (GDPR path that respects bitemporality)
- `DELETE /entities/{id}?mode=redact`: overwrite PII payloads (names, aliases, evidence quotes, embeddings) with tombstones while preserving graph structure + timestamps; cascade to facts/relationships `evidence` fields. Document the model in `docs/data-lifecycle.md`: *redaction rewrites content, never timelines*.
- **Tests:** post-redaction, no PII string appears anywhere in store dump; point-in-time queries still return tombstoned shells (structure intact).

### Task G2: Export/import
- `engram export --group-id X --out dump.ndjson` / `engram import` — full-fidelity NDJSON (entities, relationships with all 4 time columns, facts, commitments, snapshots). Doubles as the Neo4j↔sqlite↔memory migration path and the backup story.
- **Tests:** export→import→export round-trip is byte-identical (modulo ordering) across all three backends.

### Task G3: Schema versioning
- `schema_version` node in store + startup check + `engram migrate` scaffold (no-op migration registered). Cheap now, existential later.

## Phase H — Observability & Docs Polish (Tasks H1–H3, ~1–2 weeks)

### Task H1: OpenTelemetry hooks
- Optional `otel` extra; spans per pipeline stage + per storage call, trace_id in `IngestResponse`. Works with Langfuse/Jaeger out of the box; `docs/observability.md` with docker-compose Jaeger example.

### Task H2: Cookbook examples
- `examples/chatbot_with_memory/` (60-line script: chat loop + `/context` + async ingest), `examples/agent_mcp/` (Claude Code/Desktop config walkthrough), `examples/support_memory/` (the README's support use case, runnable). Each with its own README and `uv run` one-liner; CI smoke-runs them against lite mode.

### Task H3: Versioned docs + release cadence
- `CHANGELOG.md` (keep-a-changelog), GitHub release notes automation, `good first issue` seeding from this plan's smaller tasks (D1 docs, H2 examples, G2 import).

---

## Sequencing

```
Week:     1   2   3   4   5   6   7   8   9   10
Phase A   ██  ██
Phase B       ██  ██
Phase C           ██  ██
Phase D               ██  ██  ██
Phase E               ██
Phase F                   ██  ██  ██
Phase G                           ██  ██
Phase H                               ██  ██
```

Rationale: A→B→C is the adoption funnel in order of first-touch (find via MCP → get value via /context → install friction-free). D+E before any "production-ready" claim. F runs parallel once /context exists (benchmarks exercise it). G+H close the loop.

## Standing Rules

1. Every task: tests first, `uv run pytest tests/unit -q` + ruff + mypy green (existing bar).
2. No route without auth coverage after E1; no new endpoint skips the `/v1` prefix.
3. Every LLM-touching path reports cost via D1 instrumentation.
4. Breaking changes get one deprecation release + CHANGELOG entry.
5. Docs land in the same PR as the feature — an undocumented feature is unfinished.

## Immediate Start (when prediction plan Phase 0–1 are merged)

```bash
uv sync && uv run pytest tests/unit -q          # green baseline
# Task A1: extract client
git mv-equivalent: create src/engram/client.py, refactor cli/main.py imports
uv run pytest tests/unit/test_cli.py tests/unit/test_client.py -q
```
