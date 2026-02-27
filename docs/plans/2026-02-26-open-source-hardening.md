# Open-Source Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship Engram with a complete MIT license, a truthful README demo, and a functional CLI path so outside users can ingest/query without custom scripts.

**Architecture:** Keep Engram’s FastAPI + Neo4j core unchanged. Add an MIT `LICENSE` artifact, expand docs to show an end-to-end ingest/query walkthrough, and wire the CLI to call the running API via HTTPX so newcomers can exercise the system immediately.

**Tech Stack:** FastAPI, Typer, httpx, pytest, markdown docs

### Task 1: Add MIT License Artifact

**Files:**
- Create: `LICENSE`

**Step 1: Write MIT license text**

```
MIT License
…(full text)…
```

**Step 2: Save to `LICENSE`**

Command: `cat > LICENSE <<'EOF' …`
Expected: file exists with MIT copyright notice for current year + author.

**Step 3: Verify**

Command: `rg "MIT License" LICENSE`
Expected: matches line 1.

### Task 2: README Quickstart with Real Demo

**Files:**
- Modify: `README.md`
- Modify: `docs/onboarding.md`

**Step 1: Document CLI-based ingest/query walkthrough**

Add a “5-Minute Demo” section showing:
1. `uv sync && uv run engram init`
2. `uv run engram serve` (separate terminal)
3. `uv run engram ingest examples/coaching-demo.json --conversation-id coaching-demo --group-id client-kendra`
4. `uv run engram query "Kendra" --tenant-id default --as-of ...`
Include sample output JSON blocks.

**Step 2: Add comparison table**

Brief table comparing Engram vs Mem0, LightRAG, GraphRAG focusing on bitemporal tracking, exclusivity rules, decay.

**Step 3: Update onboarding doc**

Mirror CLI instructions plus pointers to new CLI flags.

**Step 4: Verify formatting**

Command: `markdownlint README.md docs/onboarding.md` (if available) or manual preview.

### Task 3: Wire CLI to Real API

**Files:**
- Modify: `src/engram/cli/main.py`
- Modify: `pyproject.toml` (ensure httpx dependency already present)
- Modify: `tests/unit/test_cli.py` (new file)

**Step 1: Add helper client**

Implement internal `_EngramAPI` that loads settings/env (default `http://localhost:8000`). Use `httpx.AsyncClient` to:
- `POST /messages` per message when ingesting
- `GET /entities` to resolve canonical name
- `GET /entities/{id}/relationships` for query display

**Step 2: Update `ingest` command**

Read JSON, loop messages, call API for each message, show per-message result counts in table.

**Step 3: Update `query` command**

Add options: `--api-url`, `--group-id`, `--knowledge/--truth toggle`. Query `/entities?canonical_name=...`? (if not available, call new helper that resolves via store by hitting `/search` once implemented; for now reuse `/entities` filter + local match). Then fetch relationships and print.

**Step 4: Add unit tests**

Create `tests/unit/test_cli.py` using Typer’s `CliRunner` and `respx` (install if needed) to mock HTTP endpoints and assert CLI output.
- Test ingest handles success + API errors.
- Test query prints table rows for relationships.

**Step 5: Run tests**

Command: `uv run pytest tests/unit/test_cli.py -v`

**Step 6: Manual smoke**

Start FastAPI (`uv run engram serve --reload`) in split terminal, run CLI ingest/query commands to ensure they hit live API without errors.

### Task 4: Surface Demo Script in Repo Root

**Files:**
- Add: `Makefile` (if not present) or `scripts/demo.sh`

**Step 1: Write script**

Script should:
- Ensure `.env` exists
- Start docker-compose (or expect running services)
- Run `uv run engram ingest ...`
- Run `uv run engram query ...`
- Print “Demo complete!”

**Step 2: Document script in README quickstart.

**Step 3: Verify script executable**

Command: `bash scripts/demo.sh`
Expected: outputs sample results (stub ok).

---

Plan complete and saved to `docs/plans/2026-02-26-open-source-hardening.md`. Two execution options:

1. Subagent-Driven (this session) — dispatch fresh subagent per task with reviews between tasks.
2. Parallel Session (separate) — open new session using superpowers:executing-plans for batched execution.

Which approach?`
