# Engram

**The only OSS temporal knowledge graph built for conversations, not documents.**

Engram creates structured, persistent memory for AI systems by extracting entities and relationships from conversations, tracking how they evolve over time, and enabling temporal reasoning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.16+-green.svg)](https://neo4j.com/)

---

## The Problem

Existing AI memory systems fall short:

- **Mem0, GraphRAG, LightRAG**: Focus on document chunks, not relationships. They can't track *how* knowledge evolves over time.
- **Zep/Graphiti**: Had temporal knowledge graphs but went closed-source. The OSS space is now empty.
- **Vector databases**: Great for similarity search, but can't answer "What did we know about X on date Y?" or "How did this relationship change?"

## The Solution

Engram is conversation-native:

- **Extract** → Automatically identifies entities (people, preferences, goals) and relationships from conversations
- **Connect** → Builds a knowledge graph with versioned, temporal relationships
- **Reason** → Supports point-in-time queries, relationship evolution tracking, and confidence decay

### What Makes It Different

**Bitemporal relationship versioning** — the secret sauce:

```
Example: Tracking evolving preferences

Week 1: "Kendra loves Nike running shoes"
→ Creates: (Kendra)-[:prefers {valid_from: W1, valid_to: null}]->(Nike)

Week 3: "Kendra switched to Adidas, Nike quality dropped"
→ Terminates old: (Kendra)-[:prefers {valid_to: W3}]->(Nike)
→ Creates new: (Kendra)-[:prefers {valid_from: W3}]->(Adidas)

Query "What did Kendra prefer in Week 2?"
→ Returns: Nike (because it was valid from W1 to W3)

Query "What does Kendra prefer now?"
→ Returns: Adidas (current preference)
```

**This pattern is proven** — built in production for a coaching platform where it tracks coach-client relationships, extracts insights from conversations, and surfaces relevant context over time.

---

## Quickstart

Get Engram running in < 5 minutes:

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (or Anthropic/Ollama for local models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/engram.git
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

## Core Concepts

### 1. Entities

Conversation-native entity types:

- **Person**: Individuals in conversations
- **Preference**: Likes, dislikes, favorites (high decay rate)
- **Goal**: Objectives, intentions
- **Concept**: Abstract ideas, topics
- **Event**: Meetings, deadlines, milestones
- **Topic**: Conversation subjects

### 2. Relationships (Versioned)

Every relationship has:

- `valid_from` — when the relationship became true
- `valid_to` — when it stopped being true (NULL = still active)
- `created_at` — when we learned about it
- `confidence` — 0.0-1.0 (decays over time if not reinforced)

This **bitemporal model** lets you query:
- **Current state**: "What relationships are active now?"
- **Historical state**: "What did we know on date X?"
- **Evolution**: "How did this relationship change over time?"

### 3. Temporal Reasoning

**Decay function**: Relationships lose confidence over time if not reinforced.

```python
# Preferences decay fast (0.05/day)
After 30 days: confidence drops from 1.0 → 0.22
After 60 days: confidence floors at 0.1

# Social relationships decay slowly (0.005/day)
After 30 days: confidence drops from 1.0 → 0.86
```

**Reinforcement**: When a relationship is re-mentioned, confidence increases.

---

## How Engram Compares

| Capability | Engram | Mem0 | LightRAG | GraphRAG |
|------------|--------|------|----------|----------|
| Bitemporal validity/knowledge timelines | ✅ Built-in (`valid_*` + `recorded_*`) | ❌ snapshot only | ❌ snapshot only | ⚠️ record time only |
| Conversation-native entity/relationship schema | ✅ People, preferences, decay-aware exclusivity | ⚠️ metadata only | ❌ chunk-based | ❌ document graph |
| Confidence decay + reinforcement | ✅ configurable per type | ❌ | ❌ | ❌ |
| Idempotent ingestion with dedup + retries | ✅ Redis/InMemory dedup service | ⚠️ manual | ⚠️ manual | ⚠️ manual |
| Point-in-time + evolution queries | ✅ `/query/point-in-time`, `/query/evolution` | ❌ | ❌ | ⚠️ limited |
| OSS license | ✅ MIT | ⚠️ custom | ✅ Apache-2.0 | ✅ MIT |

⚠️ = partial/DIY implementations according to the latest public docs.

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

### Extraction Pipeline

1. **Entity Extraction** → LLM identifies entities and canonical forms
2. **Relationship Inference** → LLM extracts relationships with confidence scores
3. **Conflict Resolution** → Temporal versioning handles contradictions

### Temporal Model

**Bitemporal property graph**:
- `valid_from` / `valid_to` → when fact was true (valid-time)
- `created_at` → when fact was recorded (transaction-time)

This enables:
- Point-in-time queries ("as of date X")
- Relationship evolution tracking
- Backdated corrections

See [ARCHITECTURE.md](ARCHITECTURE.md) for full schema, API design, and implementation details.

---

## Roadmap

### v0.1.0 — MVP (Current)

**Goal**: Working end-to-end demo, runs in < 5 minutes

- ✅ Core graph schema (entities, versioned relationships)
- ✅ Entity extraction (LLM-based, multi-provider)
- ✅ Relationship inference with temporal markers
- ✅ Conflict resolution & versioning
- ✅ Decay function (exponential, configurable rates)
- ✅ Point-in-time queries
- ✅ Relationship evolution tracking
- ✅ REST API (8 endpoints)
- ✅ CLI tools (`engram init`, `ingest`, `query`, `export`)
- ✅ Basic web UI (graph visualization, entity browser)
- ✅ Docker Compose deployment

**Timeline**: 4-6 weeks

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

1. **Issues**: Found a bug? Have a feature request? [Open an issue](https://github.com/yourusername/engram/issues)
2. **Discussions**: Questions or ideas? [Start a discussion](https://github.com/yourusername/engram/discussions)
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
  author = {Your Name},
  title = {Engram: Temporal Knowledge Graphs for AI Memory},
  year = {2026},
  url = {https://github.com/yourusername/engram},
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

**Current**: Design & Architecture Phase  
**Next**: Implementation (v0.1.0)  
**Timeline**: MVP in 4-6 weeks

⭐ **Star this repo** to follow development and support the project!

---

**Engram** — because AI systems should remember what they learned, when they learned it, and how it changed.
