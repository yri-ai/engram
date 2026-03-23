# Engram

**Temporal context graph for AI — not memory, context.**

Memory is recall. Context is understanding *what changed, when, and why it matters now*. Engram extracts structured knowledge from conversations — entities, relationships, facts, commitments — tracks how they evolve bitemporally, and gives AI systems the full picture, not just a snapshot.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.16+-green.svg)](https://neo4j.com/)

---

## The Problem

"AI memory" systems store what was said. They don't understand what it *means* now:

- **Mem0, GraphRAG, LightRAG**: Chunk documents and retrieve by similarity. Can't answer "How did X change?" or "What did we know on date Y?"
- **Zep/Graphiti**: Built the right thing (bitemporal graphs) but went closed-source. The OSS space is empty.
- **Vector databases**: Good at "find similar things." Useless at "what evolved, what contradicts, what supersedes."

## What Engram Does

Engram builds a living context graph from conversations:

- **Extract** → Entities, relationships, standalone facts, commitments, and narrative summaries — all from natural conversation
- **Connect** → Bitemporal knowledge graph where every edge tracks *when it was true* and *when we learned it*
- **Reason** → Point-in-time queries, evolution tracking, confidence decay, fact supersession

### Context, Not Memory

Memory systems store snapshots. Engram tracks *evolution*:

```
Message 1: "Kendra loves Nike running shoes"
  → Entity: Kendra (PERSON)
  → Relationship: (Kendra)-[:prefers]->(Nike)
  → Fact: "Kendra's preferred brand is Nike" (key: brand)

Message 2: "Kendra switched to Hoka — better arch support"
  → Relationship: terminates (Kendra)-[:prefers]->(Nike), creates ->(Hoka)
  → Fact: supersedes old brand fact with "Kendra's preferred brand is Hoka"
  → Summary: shift from Nike loyalty to Hoka based on arch support
  → Snapshot: delta shows 1 superseded fact, 1 new relationship

Query "What did Kendra prefer in Week 2?" → Nike
Query "What does Kendra prefer now?" → Hoka
Query "What changed?" → Brand preference shifted, evidence: arch support
```

A memory system returns "Hoka." Engram returns *why it changed, when, and what it replaced*.

---

## Quickstart

Get Engram running in < 5 minutes:

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (or Anthropic/Ollama for local models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yri-ai/engram.git
cd engram

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start Engram
docker-compose up

# In another terminal, try the demo
docker-compose exec api engram ingest examples/coaching-demo.json
docker-compose exec api engram query "Kendra's current preferences"
```

**That's it!** You now have a running temporal knowledge graph.

### 5-Minute CLI Demo

Prefer running everything locally? The CLI now speaks to the FastAPI server so you can ingest and query without writing curl commands.

```bash
# 1. Install deps
uv sync

# 2. Terminal A - start the API (uses your .env settings)
uv run engram serve

# 3. Terminal B - ingest the demo conversation
uv run engram ingest examples/coaching-demo.json \
  --conversation-id coaching-demo \
  --group-id client-kendra

# 4. Query what Engram learned
uv run engram query "Kendra" \
  --conversation-id coaching-demo \
  --mode world_state
```

Sample output:

```
Ingestion Summary
+--------------+----------+----------------+----------+--------------+
| Message ID   | Entities | Relationships  | Conflicts| Latency (ms) |
+--------------+----------+----------------+----------+--------------+
| cli-...      | 3        | 2              | 1        | 342.5        |
+--------------+----------+----------------+----------+--------------+
```

Want it automated? `scripts/demo.sh` checks `http://localhost:8000/health`, ingests `examples/coaching-demo.json`, and runs the query above. Start `docker compose up` (or `uv run engram serve`) first, then run `./scripts/demo.sh`.

### Web UI

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API explorer, or [http://localhost:7474](http://localhost:7474) for the Neo4j Browser (credentials: `neo4j`/`password`).

---

## What Gets Extracted

Engram's 7-stage pipeline extracts five artifact types from every message:

### Entities
Conversation-native types: **Person**, **Preference**, **Goal**, **Concept**, **Event**, **Topic**. Each with vector embeddings for deduplication and deterministic IDs for idempotent processing.

### Relationships (Bitemporal)
Edges between entities with full temporal versioning:
- `valid_from` / `valid_to` — when the relationship was true in the world
- `recorded_from` / `recorded_to` — when we learned about it
- Exclusivity policies (e.g., `prefers` and `avoids` are mutually exclusive)
- Confidence decay over time, reinforcement on re-mention

### Facts
Standalone knowledge claims about entities — things that aren't relationships between two entities. "Alice is 32", "Bob works at Acme", "The deadline is March 30th." Facts have supersession chains: when new information contradicts old, the old fact is marked `SUPERSEDED` and linked to its replacement.

### Commitments
Future-oriented actions extracted from conversation: "I'll do X by Friday." Tracked with status (active, completed, cancelled, missed) and target dates.

### Conversation Summaries
Narrative arc per message: opening state, key shift, closing state. Captures *what happened* at a high level, inspired by temporal-relationships' SessionArc pattern.

---

## Temporal Reasoning

**Confidence decay**: Knowledge fades if not reinforced.

```
Preferences decay fast (0.05/day): 1.0 → 0.22 after 30 days
Social connections decay slowly (0.005/day): 1.0 → 0.86 after 30 days
```

**Point-in-time queries**: "What was true on date X?" — valid-time, record-time, or bitemporal.

**Evolution tracking**: Full version history of every relationship and fact, with supersession chains.

---

## How Engram Compares

| Capability | Engram | Mem0 | LightRAG | GraphRAG |
|------------|--------|------|----------|----------|
| Bitemporal timelines (valid + record time) | ✅ 4-column model | ❌ snapshot | ❌ snapshot | ⚠️ record only |
| Standalone facts with supersession | ✅ fact chains | ❌ | ❌ | ❌ |
| Conversation-native schema | ✅ entities, relationships, facts, commitments | ⚠️ metadata | ❌ chunk-based | ❌ document graph |
| Confidence decay + reinforcement | ✅ per-type configurable | ❌ | ❌ | ❌ |
| Extraction context awareness | ✅ prior facts + relationships fed to LLM | ❌ | ❌ | ❌ |
| Point-in-time + evolution queries | ✅ API endpoints | ❌ | ❌ | ⚠️ limited |
| Idempotent ingestion | ✅ Redis dedup | ⚠️ manual | ⚠️ manual | ⚠️ manual |
| OSS license | ✅ MIT | ⚠️ custom | ✅ Apache-2.0 | ✅ MIT |

---

## API Examples

### Ingest a Conversation Message

```bash
curl -X POST http://localhost:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Kendra mentioned she loves Nike running shoes, they are the best for marathons",
    "speaker": "Kendra",
    "timestamp": "2024-01-15T10:00:00Z"
  }'
```

**Result**: Extracts entities (Kendra, Nike, marathons) and relationships (prefers, mentioned_with).

### Query Current Relationships

```bash
curl "http://localhost:8000/entities/Kendra/relationships?type=prefers"
```

**Response**:
```json
{
  "entity": "Kendra Martinez",
  "relationships": [
    {
      "target": "Adidas",
      "type": "prefers",
      "confidence": 0.85,
      "valid_from": "2024-03-20T14:30:00Z",
      "valid_to": null,
      "evidence": "Switched to Adidas, better arch support"
    }
  ]
}
```

### Point-in-Time Query

```bash
curl "http://localhost:8000/query/point-in-time?entity=Kendra&as_of=2024-02-01T00:00:00Z&type=prefers"
```

**Response**:
```json
{
  "entity": "Kendra Martinez",
  "as_of": "2024-02-01T00:00:00Z",
  "relationships": [
    {
      "target": "Nike",
      "type": "prefers",
      "confidence": 0.95,
      "valid_from": "2024-01-15T10:00:00Z",
      "valid_to": "2024-03-20T14:30:00Z"
    }
  ]
}
```

### Relationship Evolution

```bash
curl "http://localhost:8000/query/evolution?entity=Kendra&type=prefers"
```

**Response**:
```json
{
  "entity": "Kendra Martinez",
  "relationship_type": "prefers",
  "timeline": [
    {
      "version": 1,
      "target": "Nike",
      "period": "2024-01-15 to 2024-03-20",
      "confidence": 0.95,
      "evidence": "I love Nike running shoes, they're the best",
      "status": "superseded"
    },
    {
      "version": 2,
      "target": "Adidas",
      "period": "2024-03-20 to present",
      "confidence": 0.85,
      "evidence": "Switched to Adidas, better arch support",
      "status": "active"
    }
  ]
}
```

---

## Use Cases

### 1. Coaching Platforms

Track client progress, goals, and preferences over time:

- **Goal evolution**: "Client wanted to lose 10 lbs (Jan), achieved it (Mar), now wants to run marathon"
- **Preference changes**: "Switched from morning to evening workouts"
- **Relationship dynamics**: "Confidence in fitness routine increased over 3 months"

### 2. Personal Knowledge Graphs

Build your own "second brain" from conversations:

- Import chat logs (Slack, Discord, WhatsApp)
- Track how your interests evolve
- Query: "What was I interested in 6 months ago?"

### 3. Customer Support Memory

Remember user issues and context across sessions:

- "User reported login bug 2 weeks ago (marked resolved), now reports again (regression?)"
- "User preferred email contact (changed to phone last week)"
- Track feature requests over time

### 4. Sales/CRM

Track lead interests and interactions:

- "Lead interested in Enterprise plan (Q1), switched to Pro (Q2)"
- "Mentioned pricing concerns 3 times in last month (deal at risk?)"
- Relationship strength decay (no contact in 30 days = cold lead)

---

## Architecture

Engram's architecture is described in detail in [ARCHITECTURE.md](ARCHITECTURE.md). Key highlights:

### Tech Stack

- **Database**: Neo4j Community Edition (native graph, vector indexes)
- **Backend**: Python 3.11+, FastAPI, Pydantic
- **LLM**: LiteLLM (multi-provider: OpenAI, Anthropic, Ollama)
- **Embeddings**: text-embedding-3-small (or nomic-embed-text for local)

### Extraction Pipeline (7 stages)

1. **Entity Extraction** → LLM identifies entities with rich prior context (existing facts + relationships)
2. **Relationship Inference** → LLM extracts relationships with confidence scores and structured evidence
3. **Conflict Resolution** → Rule-based exclusivity enforcement and temporal versioning
4. **Fact Extraction** → LLM extracts standalone facts with supersession detection
5. **Commitment Extraction** → LLM identifies future-oriented actions and intentions
6. **Conversation Summary** → LLM generates narrative arc (opening → shift → closing)
7. **Snapshot** → Captures conversation state and what changed (delta tracking)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full schema, API design, and implementation details.

---

## Roadmap

### v0.1.0 — MVP (Current)

- ✅ Core graph schema (entities, bitemporal relationships)
- ✅ 7-stage extraction pipeline (entities, relationships, facts, commitments, summaries)
- ✅ Context-aware extraction (prior facts + relationships fed to LLM)
- ✅ Fact model with supersession chains
- ✅ Conversation snapshots with delta tracking
- ✅ Conflict resolution & exclusivity policies
- ✅ Confidence decay + reinforcement
- ✅ Point-in-time and evolution queries
- ✅ REST API (10 endpoints including facts)
- ✅ CLI tools (`engram serve`, `ingest`, `query`)
- ✅ Docker Compose deployment (Neo4j + Redis + API)

### v0.2 — Multi-Tenancy (Week 8)

- Multi-user support (isolated graphs per user)
- API key authentication
- Relationship clustering (group related entities)
- Advanced search (filters, aggregations)

### v0.3 — Analytics & Integrations (Week 12)

- Advanced analytics (PageRank, centrality, trending topics)
- Vertical schema templates (coaching, sales, support)
- Slack integration (ingest channel conversations)
- LangChain retriever plugin

### v1.0 — Production Ready (Month 6)

- Hosted beta (commercial SaaS layer)
- SSO (SAML, OAuth2)
- Audit trails
- Custom extraction pipeline builder (no-code)
- PostgreSQL + AGE adapter (community contribution)

### v2.0 — Advanced Features (Year 1)

- Entity merging UI (manual review + active learning)
- Learned decay rates (from user feedback)
- Graph diffing (compare states between dates)
- Time-travel visualization (animated evolution)
- CRM integrations (Salesforce, HubSpot)

---

## License

**MIT License** — Free forever, self-hostable, no feature gimping.

We believe the core knowledge graph engine should be open source. Our commercial layer (Engram Cloud) will offer:
- Managed hosting (no DevOps required)
- SSO and enterprise auth
- Vertical-specific schemas
- Advanced analytics
- Premium support

See [ARCHITECTURE.md § Core vs. Commercial Split](ARCHITECTURE.md#4-core-vs-commercial-split) for details.

---

## Contributing

Engram is in active development. Contributions are welcome!

**How to contribute**:

1. **Issues**: Found a bug? Have a feature request? [Open an issue](https://github.com/yri-ai/engram/issues)
2. **Discussions**: Questions or ideas? [Start a discussion](https://github.com/yri-ai/engram/discussions)
3. **Pull Requests**: Want to contribute code? Fork, branch, and submit a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and the full PR checklist.

**Priority areas for contributions**:
- PostgreSQL + AGE adapter
- JavaScript/TypeScript SDK
- Additional LLM providers (Cohere, local models)
- Vertical schema templates (healthcare, education, sales)
- Documentation improvements

---

## Community

- **Discord**: [Join the Engram community](https://discord.gg/engram) (coming soon)
- **Twitter**: [@EngramAI](https://twitter.com/EngramAI) (updates, demos)
- **Blog**: [engram.dev/blog](https://engram.dev/blog) (architecture deep dives, use cases)

---

## Research

Engram's architecture is grounded in research and production patterns. See [RESEARCH.md](RESEARCH.md) for:

- Competitive analysis (Mem0, GraphRAG, LightRAG, Graphiti)
- Temporal graph modeling patterns (bitemporal, versioning)
- Database technology comparison (Neo4j, PostgreSQL AGE, ArangoDB)
- Production examples (Zep, OpenMemory)

Key findings:
- **Graphiti** (Zep) achieved 94.8% accuracy on Deep Memory Retrieval benchmark using temporal graphs
- **Aion** (Neo4j extension) provides 10x speedup for temporal queries
- **OSS gap**: No active open-source temporal knowledge graph for conversations

---

## FAQ

### How is this different from vector databases?

Vector databases (Pinecone, Qdrant) excel at similarity search but don't understand relationships or time. Engram uses vectors for entity resolution but stores knowledge as a graph, enabling:
- "Who knows whom?"
- "What did Kendra prefer before she switched?"
- "Show me relationships that weakened over time"

### How is this different from Mem0?

Mem0 is vector-first with optional graph overlay. Relationships are not versioned — you can't track *when* a preference changed. Engram versions relationships by default.

### How is this different from GraphRAG?

GraphRAG is batch-oriented for document summarization. Adding new information requires re-indexing. Engram is real-time and conversation-native — ingest messages as they happen, no re-indexing.

### Why not use Zep/Graphiti?

Zep's Graphiti had the right architecture (bitemporal graphs) but went closed-source in 2025. Engram brings that approach back to open source, with MIT license forever.

### Can I use this with LangChain/LlamaIndex?

**v0.3+** will have official plugins. For now, use Engram's REST API as a retrieval backend in your LangChain chains.

### What LLM providers are supported?

Via LiteLLM:
- OpenAI (GPT-4o-mini, GPT-4o)
- Anthropic (Claude Haiku, Sonnet)
- Ollama (local models: Llama 3, Mistral)
- Cohere, Google, Azure OpenAI

### Does it work offline?

Yes, use Ollama for local LLM inference and nomic-embed-text for embeddings. Fully air-gapped deployment possible.

### How does it scale?

**Conversation-scale** (10K-100K entities): Single Neo4j instance, Docker Compose deployment.  
**Commercial-scale** (1M+ entities): Neo4j AuraDB (managed), Kubernetes, Aion extension for 10x temporal query speedup.

### What about privacy?

**Self-hosted**: Your data never leaves your infrastructure.  
**Commercial (future)**: SOC2, HIPAA-compliant deployments available. EU data residency options.

---

## Citation

If you use Engram in research or production, please cite:

```bibtex
@software{engram2026,
  author = {Leonard Langsdorf},
  title = {Engram: Temporal Context Graph for AI},
  year = {2026},
  url = {https://github.com/yri-ai/engram},
  license = {MIT}
}
```

---

## Acknowledgments

Engram is inspired by:
- **Graphiti** (Zep) — pioneered temporal knowledge graphs for AI memory
- **OpenMemory** — demonstrated temporal fact tracking patterns
- **Aion** (Neo4j extension) — proved 10x temporal query speedups possible

Built with: Neo4j, FastAPI, LiteLLM, Pydantic, and the amazing OSS community.

---

## Status

**Current**: v0.1.0 MVP — actively developing

---

**Engram** — context, not memory. What changed, when, and why it matters now.
