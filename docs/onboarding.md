# Engram Onboarding Guide

How to set up Engram, ingest documents, and query what the system learned over time.

---

## 0. Fast Demo

Already have the stack running (either via `docker compose up` or `uv run engram serve`)? Run the scripted walkthrough:

```bash
./scripts/demo.sh          # assumes API at http://localhost:8000
# or override API_URL, GROUP_ID, CONVERSATION_ID
API_URL=http://localhost:8000 ./scripts/demo.sh examples/coaching-demo.json
```

The script checks `/health`, ingests `examples/coaching-demo.json`, and runs a query so you can see Engram's responses immediately.

## 1. Setup

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key (or Anthropic/Ollama — see [LLM Configuration](#llm-configuration))

### Start Services

```bash
git clone https://github.com/yri-ai/engram.git
cd engram

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

docker-compose up
```

This starts three services:

| Service | Port | Purpose |
|---------|------|---------|
| **Neo4j** | 7474 (browser), 7687 (bolt) | Graph database — stores entities and relationships |
| **Redis** | 6379 | Message deduplication (idempotency) |
| **API** | 8000 | FastAPI server — all ingestion and querying happens here |

Verify everything is running:

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "0.1.0"}
```

### Neo4j Browser

Open [http://localhost:7474](http://localhost:7474) and log in with `neo4j` / `password` to visually explore the graph.

### API Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### Local Development (without Docker)

```bash
# Requires: Python 3.11+, uv, a running Neo4j instance, optionally Redis

uv sync
cp .env.example .env
# Edit .env — point NEO4J_URI to your Neo4j, add OPENAI_API_KEY

# Initialize Neo4j schema
uv run engram init

# Start the API server
uv run engram serve
```

# CLI Walkthrough

Once the API is running you can ingest/query entirely through the CLI:

```bash
uv sync
uv run engram serve                # terminal A

uv run engram ingest examples/coaching-demo.json \
  --conversation-id coaching-demo \
  --group-id client-kendra        # terminal B

uv run engram query "Kendra" \
  --conversation-id coaching-demo \
  --mode world_state
```

The CLI prints an ingestion summary table and renders the active or point-in-time relationships returned from the FastAPI service.

### LLM Configuration

Engram uses [LiteLLM](https://docs.litellm.ai/) for multi-provider support. Set in `.env`:

```bash
# OpenAI (default)
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic
# LLM_MODEL=claude-3-haiku-20240307
# ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local, fully offline)
# LLM_MODEL=ollama/llama3
# OLLAMA_BASE_URL=http://localhost:11434
```

---

## 2. Core Concepts

### Entities

Things extracted from text. Each entity has a **type** and a **canonical name**.

| Type | Examples |
|------|----------|
| `PERSON` | Kendra, Coach Sarah |
| `PREFERENCE` | Nike running shoes, Adidas |
| `GOAL` | Boston Marathon sub-4 |
| `EVENT` | Boston Marathon |
| `CONCEPT` | arch support, long distance running |
| `TOPIC` | training plan |

### Relationships (Bitemporal)

Edges between entities. Every relationship carries four timestamps:

| Field | Meaning |
|-------|---------|
| `valid_from` | When the relationship became true in the real world |
| `valid_to` | When it stopped being true (`null` = still active) |
| `recorded_from` | When Engram first learned about it |
| `recorded_to` | When Engram stopped believing it (`null` = still believed) |

This bitemporal model lets you ask two different questions:
- **World state**: "What was actually true on date X?" (uses `valid_from`/`valid_to`)
- **Knowledge state**: "What did we believe on date X?" (uses `recorded_from`/`recorded_to`)

### Relationship Types

| Type | Meaning | Exclusivity |
|------|---------|-------------|
| `prefers` | Active preference | Exclusive — new `prefers` terminates old ones for same source. Also terminates `avoids`. |
| `avoids` | Active dislike | Exclusive — same rules as `prefers`, reversed. |
| `knows` | Social connection | Non-exclusive — accumulates |
| `has_goal` | Objective/intention | Non-exclusive — accumulates |
| `discussed` | Mentioned in conversation | Non-exclusive — accumulates |
| `mentioned_with` | Co-occurrence | Non-exclusive — accumulates |
| `relates_to` | Fallback for unknown types | Non-exclusive — accumulates |

### Scoping: tenant_id, conversation_id, group_id

Engram uses three levels of scoping:

| Scope | Purpose | Default |
|-------|---------|---------|
| `tenant_id` | Top-level isolation. Separate tenants never see each other's data. | `"default"` |
| `conversation_id` | Identifies a single session or document. | `"default"` |
| `group_id` | Links multiple conversations into one knowledge scope. Entities and exclusivity rules are shared within a group. | Falls back to `conversation_id` if not set |

**`group_id` is the key concept.** Without it, each conversation is isolated — Kendra in conversation A is a different entity than Kendra in conversation B. Set `group_id` to the same value across conversations and they share a unified knowledge graph.

Use cases:
- **Per-person memory**: `group_id = "client-kendra"` — all sessions with Kendra share one graph
- **Per-project corpus**: `group_id = "project-atlas"` — all documents about Project Atlas share one graph
- **Per-team knowledge**: `group_id = "team-eng"` — all engineering conversations share one graph

---

## 3. The Ingest API

All ingestion goes through a single endpoint:

```
POST /messages
```

### Request Body

```json
{
  "text": "Kendra mentioned she switched to Adidas for better arch support.",
  "speaker": "Kendra",
  "timestamp": "2024-03-20T14:30:00Z",
  "conversation_id": "coaching-session-1",
  "tenant_id": "default",
  "group_id": "client-kendra",
  "message_id": "msg-unique-123",
  "metadata": {}
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `text` | **Yes** | The message text to extract knowledge from |
| `speaker` | **Yes** | Who said it |
| `timestamp` | No | When it was said. Defaults to now. **Use the real timestamp** — this controls `valid_from` on extracted relationships. |
| `conversation_id` | No | Session/document identifier. Default: `"default"` |
| `tenant_id` | No | Tenant isolation. Default: `"default"` |
| `group_id` | No | Cross-conversation linking scope. Default: falls back to `conversation_id` |
| `message_id` | No | Idempotency key. Auto-generated UUID if not provided. **Set this yourself** for replay safety. |
| `metadata` | No | Arbitrary JSON. Stored but not used by the pipeline. |

### Response

```json
{
  "message_id": "msg-unique-123",
  "entities_extracted": 2,
  "relationships_inferred": 1,
  "conflicts_resolved": 1,
  "processing_time_ms": 342.5
}
```

### What Happens Inside

1. **Dedup check** — If `message_id` was already processed, returns immediately with zeros. Safe to retry.
2. **Entity extraction** (LLM) — Identifies people, preferences, goals, etc. Upserts entities to the graph.
3. **Relationship inference** (LLM) — Determines how entities relate. Applies confidence snapping (1.0, 0.8, 0.6, 0.4).
4. **Conflict resolution** (rule-based) — Enforces exclusivity policies. If Kendra now `prefers` Adidas, terminates the old Nike `prefers` relationship by setting `valid_to`.
5. **Dedup rollback** — If steps 2-4 fail, the dedup mark is rolled back so the message can be retried.

### Idempotency

Set `message_id` to a deterministic value (e.g., hash of file path + modification time). Re-sending the same `message_id` is a no-op — the response returns zeros.

```bash
# First call: processes the message
curl -X POST http://localhost:8000/messages -H "Content-Type: application/json" \
  -d '{"text": "I prefer Nike", "speaker": "Kendra", "message_id": "abc-123"}'
# {"message_id": "abc-123", "entities_extracted": 2, ...}

# Second call: deduped, no work done
curl -X POST http://localhost:8000/messages -H "Content-Type: application/json" \
  -d '{"text": "I prefer Nike", "speaker": "Kendra", "message_id": "abc-123"}'
# {"message_id": "abc-123", "entities_extracted": 0, ...}
```

---

## 4. Querying the Graph

### List Entities

```bash
curl "http://localhost:8000/entities?tenant_id=default&entity_type=PERSON&limit=50"
```

Query parameters: `tenant_id`, `conversation_id`, `entity_type` (`PERSON`, `PREFERENCE`, `GOAL`, `EVENT`, `CONCEPT`, `TOPIC`), `limit`, `offset`.

### Get Entity Details

```bash
curl "http://localhost:8000/entities/{entity_id}"
```

Returns: `id`, `tenant_id`, `conversation_id`, `entity_type`, `canonical_name`, `aliases`, `created_at`, `last_mentioned`, `source_messages`, `metadata`.

Entity IDs are deterministic: `{tenant}:{group_id}:{type}:{canonical_name}`. Example: `default:client-kendra:PERSON:kendra`.

### Get Active Relationships

```bash
curl "http://localhost:8000/entities/{entity_id}/relationships?rel_type=prefers"
```

Returns only **currently active** relationships (where `valid_to` is null).

### Point-in-Time Query

"What was true about entity X on date Y?"

```bash
# World state: what was actually true on Feb 15, 2024?
curl "http://localhost:8000/query/point-in-time?entity=kendra&as_of=2024-02-15T00:00:00Z&tenant_id=default&mode=world_state"

# Knowledge state: what did we believe on Feb 15, 2024?
curl "http://localhost:8000/query/point-in-time?entity=kendra&as_of=2024-02-15T00:00:00Z&tenant_id=default&mode=knowledge"

# Both timelines at once
curl "http://localhost:8000/query/point-in-time?entity=kendra&as_of=2024-02-15T00:00:00Z&tenant_id=default&mode=bitemporal"
```

| Mode | Returns |
|------|---------|
| `world_state` | Relationships where `valid_from <= as_of` and (`valid_to` is null or `valid_to > as_of`) |
| `knowledge` | Relationships where `recorded_from <= as_of` and (`recorded_to` is null or `recorded_to > as_of`) |
| `bitemporal` | Both `world_state` and `knowledge` arrays in one response |

Optional filter: `&rel_type=prefers`

### Evolution Query

"How did entity X's relationships change over time?"

```bash
# All relationship types
curl "http://localhost:8000/query/evolution?entity=kendra&tenant_id=default"

# Filter by type
curl "http://localhost:8000/query/evolution?entity=kendra&tenant_id=default&rel_type=prefers"

# Filter by target
curl "http://localhost:8000/query/evolution?entity=kendra&tenant_id=default&target=nike-running-shoes"
```

Returns all versions of relationships (including terminated ones), sorted chronologically. Each entry includes `version`, `supersedes`, `valid_from`, `valid_to`.

### Search Entities

```bash
curl "http://localhost:8000/search?q=kendra&tenant_id=default&limit=20"
```

Substring search on canonical names and aliases.

---

## 5. Ingesting a Corpus of Markdown Files

The primary use case: point Engram at a directory of markdown files and watch it build a knowledge graph that evolves as files change.

### The Pattern

Each markdown file becomes one `POST /messages` call:

| Ingestion Field | Mapped From |
|-----------------|-------------|
| `text` | File contents (or chunked if very large) |
| `speaker` | Author, or the filename if author is unknown |
| `timestamp` | File modification time (`mtime`) |
| `conversation_id` | Filename or document slug (e.g., `"meeting-2024-01-15"`) |
| `group_id` | Corpus name — **same value for all files** (e.g., `"project-atlas"`) |
| `message_id` | Deterministic hash of file path + `mtime` (for idempotency) |

### Example: Ingestion Script

```python
#!/usr/bin/env python3
"""Ingest a directory of markdown files into Engram."""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

ENGRAM_URL = "http://localhost:8000"
CORPUS_DIR = Path("./my-notes")
GROUP_ID = "my-knowledge-base"
TENANT_ID = "default"


def ingest_file(client: httpx.Client, file_path: Path) -> dict:
    """Ingest a single markdown file."""
    text = file_path.read_text()
    mtime = os.path.getmtime(file_path)
    timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)

    # Deterministic message_id: same file+mtime = same ID = idempotent
    content_key = f"{file_path}:{mtime}"
    message_id = hashlib.sha256(content_key.encode()).hexdigest()[:16]

    response = client.post(
        f"{ENGRAM_URL}/messages",
        json={
            "text": text,
            "speaker": file_path.stem,  # filename as speaker
            "timestamp": timestamp.isoformat(),
            "conversation_id": file_path.stem,
            "group_id": GROUP_ID,
            "tenant_id": TENANT_ID,
            "message_id": message_id,
        },
    )
    response.raise_for_status()
    return response.json()


def main():
    md_files = sorted(CORPUS_DIR.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files in {CORPUS_DIR}")

    with httpx.Client(timeout=60.0) as client:
        for f in md_files:
            result = ingest_file(client, f)
            print(
                f"  {f.name}: "
                f"{result['entities_extracted']} entities, "
                f"{result['relationships_inferred']} relationships, "
                f"{result['conflicts_resolved']} conflicts"
            )

    print("Done. Query the graph at http://localhost:8000/docs")


if __name__ == "__main__":
    main()
```

### Re-Ingestion on File Changes

When a file changes:
1. Its `mtime` changes → the `message_id` hash changes → it's treated as a new message.
2. The extraction pipeline runs again, producing new entities and relationships.
3. Exclusivity policies handle contradictions automatically — if a preference changed, the old one is terminated.

Files that haven't changed produce the same `message_id` hash and are skipped (dedup).

Run the script on a cron or file watcher:

```bash
# Re-ingest every hour (only changed files produce new work)
0 * * * * cd /path/to/project && python ingest_corpus.py
```

### Chunking Large Files

The LLM has a context window limit. For files over ~8,000 words, split into chunks:

```python
def chunk_text(text: str, max_words: int = 4000) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks, current_chunk, current_words = [], [], 0

    for para in paragraphs:
        word_count = len(para.split())
        if current_words + word_count > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_words = [], 0
        current_chunk.append(para)
        current_words += word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks
```

Each chunk becomes a separate `POST /messages` call with a unique `message_id` (include chunk index in the hash).

---

## 6. Walkthrough: The Coaching Demo

The included `examples/coaching-demo.json` demonstrates the full lifecycle. Six messages, spread across four months, show Kendra's shoe preference evolving from Nike to Adidas.

### Ingest the Demo

```bash
# Ingest each message via the API
for i in $(seq 0 5); do
  TEXT=$(python3 -c "
import json
with open('examples/coaching-demo.json') as f:
    msgs = json.load(f)['messages']
print(json.dumps({
    'text': msgs[$i]['text'],
    'speaker': msgs[$i]['speaker'],
    'timestamp': msgs[$i]['timestamp'],
    'conversation_id': 'coaching-session-1',
    'tenant_id': 'default',
    'group_id': 'client-kendra',
    'message_id': 'demo-msg-$((i+1))'
}))
  ")
  curl -s -X POST http://localhost:8000/messages \
    -H "Content-Type: application/json" \
    -d "$TEXT" | python3 -m json.tool
done
```

Or use the Python script approach from Section 5.

### What Gets Built

After ingesting all 6 messages:

**Entities:**
- `kendra` (PERSON)
- `boston-marathon` (EVENT)
- `nike-running-shoes` (PREFERENCE)
- `coach-sarah` (PERSON)
- `adidas` (PREFERENCE)

**Relationships (active):**
- Kendra `has_goal` Boston Marathon
- Kendra `prefers` Adidas *(Nike was terminated when Adidas was ingested)*
- Kendra `knows` Coach Sarah
- Kendra `discussed` Boston Marathon

**Relationships (terminated):**
- Kendra `prefers` Nike Running Shoes *(valid_to = 2024-03-20)*

### Query: Current State

```bash
curl -s "http://localhost:8000/search?q=kendra" | python3 -m json.tool
```

### Query: What Did Kendra Prefer in February?

```bash
curl -s "http://localhost:8000/query/point-in-time?entity=kendra&as_of=2024-02-15T00:00:00Z&rel_type=prefers&mode=world_state" | python3 -m json.tool
```

Returns: Nike (valid from Jan 15, terminated Mar 20).

### Query: What Does Kendra Prefer Now?

```bash
curl -s "http://localhost:8000/query/point-in-time?entity=kendra&as_of=2024-12-01T00:00:00Z&rel_type=prefers&mode=world_state" | python3 -m json.tool
```

Returns: Adidas (valid from Mar 20, still active).

### Query: Full Preference Evolution

```bash
curl -s "http://localhost:8000/query/evolution?entity=kendra&rel_type=prefers" | python3 -m json.tool
```

Returns both versions:

1. Nike — version 1, `valid_from: 2024-01-15`, `valid_to: 2024-03-20`, superseded
2. Adidas — version 2, `valid_from: 2024-03-20`, `valid_to: null`, active

---

## 7. Watching the Graph Learn

The temporal model means you can observe *how* the graph evolves as you ingest content over time.

### Scenario: Ingesting Meeting Notes Weekly

```
Week 1: meeting-jan-15.md → "Team is using React for the frontend"
Week 2: meeting-jan-22.md → "Sarah joined as tech lead"
Week 3: meeting-jan-29.md → "We're switching from React to Svelte"
Week 4: meeting-feb-05.md → "Sarah completed the Svelte migration"
```

After ingesting all four (with the same `group_id`):

```bash
# What tech stack were we using in week 2?
curl "http://localhost:8000/query/point-in-time?entity=team&as_of=2024-01-22T00:00:00Z&rel_type=prefers&mode=world_state"
# → React

# What about now?
curl "http://localhost:8000/query/point-in-time?entity=team&as_of=2024-02-05T00:00:00Z&rel_type=prefers&mode=world_state"
# → Svelte

# Show the full evolution
curl "http://localhost:8000/query/evolution?entity=team&rel_type=prefers"
# → React (terminated week 3) → Svelte (active)
```

### The Key Insight

The `timestamp` you set on each message controls `valid_from`. By using real document dates (not ingestion time), you build a historically accurate timeline even if you batch-ingest months of documents in one go.

---

## 8. Configuration Reference

All settings are environment variables. Set in `.env` or pass directly.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (unless using Anthropic/Ollama) |

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | *(none)* | Redis password |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_ENABLED` | `true` | Set `false` to use in-memory dedup (no Redis needed) |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | LiteLLM model identifier |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (lower = more deterministic) |

### Decay Rates

Confidence decay controls how fast relationships lose confidence when not reinforced.

| Variable | Default | Description |
|----------|---------|-------------|
| `DECAY_PRESET` | `balanced` | Preset: `balanced`, `fast` (2x), `slow` (0.5x) |
| `DECAY_RATE_PREFERS` | `0.05` | Preferences decay fast (~1.5-day half-life) |
| `DECAY_RATE_KNOWS` | `0.005` | Social connections decay slowly (~7-day half-life) |
| `DECAY_RATE_HAS_GOAL` | `0.02` | Goals decay moderately |

---

## 9. CLI Reference

Working commands:

| Command | Description |
|---------|-------------|
| `engram init` | Initialize Neo4j schema and indexes |
| `engram serve` | Start the FastAPI server |
| `engram health` | Check Neo4j, Redis, and LLM connectivity |

Pending implementation (stubs — use the HTTP API instead):

| Command | Status |
|---------|--------|
| `engram ingest <file>` | Stub — parses JSON but does not call the pipeline |
| `engram query <entity>` | Stub — displays empty table |
| `engram export` | Stub — writes empty JSON |

---

## 10. API Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/messages` | Ingest a message through the extraction pipeline |
| `GET` | `/entities` | List entities (filter: `tenant_id`, `conversation_id`, `entity_type`, `limit`, `offset`) |
| `GET` | `/entities/{id}` | Get entity by ID |
| `GET` | `/entities/{id}/relationships` | Get active relationships (filter: `rel_type`) |
| `GET` | `/query/point-in-time` | Temporal query (params: `entity`, `as_of`, `tenant_id`, `rel_type`, `mode`) |
| `GET` | `/query/evolution` | Evolution timeline (params: `entity`, `tenant_id`, `target`, `rel_type`) |
| `GET` | `/search` | Search entities by name (params: `q`, `tenant_id`, `limit`) |
| `POST` | `/entities/{id}/merge` | Merge duplicate entity into primary (body: `{"duplicate_id": "..."}`) |
| `GET` | `/health` | Health check |
