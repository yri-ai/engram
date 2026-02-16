# Research Findings

**Date:** February 16, 2026  
**Purpose:** Competitive landscape analysis and temporal graph technology research

This document summarizes research into existing AI memory systems and temporal graph databases, conducted to inform Engram's architecture.

---

## Table of Contents

1. [Competitive Analysis](#competitive-analysis)
2. [Temporal Graph Technology](#temporal-graph-technology)
3. [Key Takeaways](#key-takeaways)
4. [References](#references)

---

## Competitive Analysis

### Overview

We analyzed four major open-source AI memory systems to understand their architectures, identify limitations, and validate the need for a conversation-native temporal knowledge graph.

| System | Type | Temporal Support | Relationship Handling | Status |
|--------|------|------------------|----------------------|--------|
| **Mem0** | Vector + optional graph | created_at/updated_at only | Optional, no versioning | Active OSS |
| **GraphRAG** | Entity-relationship extraction | None | Static batch processing | Active OSS (Microsoft) |
| **LightRAG** | Generic graph over chunks | Simple timestamps | Generic properties | Active OSS |
| **Graphiti (Zep)** | Bi-temporal entity graph | valid_at/invalid_at | First-class, versioned | Closed-source (was OSS) |

---

### 1. Mem0

**Repository:** [mem0ai/mem0](https://github.com/mem0ai/mem0)

#### Architecture

**Data Model:**
- Primary: Vector-based memory system
- Core entity: `MemoryItem` with text content + embeddings
- Graph memory is optional add-on (`enable_graph=True`)

**Evidence:**
```python
# mem0/configs/base.py
class MemoryItem:
    id: str
    memory: str          # Text content
    score: float         # Similarity score
    created_at: datetime
    updated_at: datetime
```

**Graph Implementation:**
- When enabled, extracts entities and relationships via Neo4j
- Graph operations run in parallel with vector storage
- **No temporal fields on edges** beyond created_at/updated_at

**Temporal Modeling:**
- Tracks when memory was created/updated
- **Does not track when facts were valid** (no valid_from/valid_to)
- Cannot represent relationship evolution over time

**Example Limitation:**
```python
# Cannot represent: "Kendra preferred Nike (Jan-Mar), then Adidas (Mar-now)"
# Instead stores: "Kendra prefers Nike" (created_at: Jan 15)
#                 "Kendra prefers Adidas" (created_at: Mar 20)
# But both appear current! No way to know Nike preference ended.
```

#### Key Findings

**Strengths:**
- Strong vector retrieval
- Layered memory architecture (short-term, long-term)
- Multi-provider LLM support

**Limitations for Temporal Relationships:**
- Graph edges lack validity windows
- Cannot track *when* a relationship was true vs *when* it was recorded
- Relationship evolution requires app-level custom logic
- **Gap:** Changing preferences/roles not modeled

**Positioning:** Great for semantic memory retrieval, but not for tracking relationship evolution.

---

### 2. GraphRAG (Microsoft)

**Repository:** [microsoft/graphrag](https://github.com/microsoft/graphrag)

#### Architecture

**Data Model:**
- Batch-oriented document indexing
- Entities and relationships extracted from text chunks
- Static graph built from document corpus

**Evidence:**
```python
# graphrag/data_model/relationship.py
class Relationship:
    id: str
    source: str              # Entity ID
    target: str              # Entity ID
    type: str
    weight: float
    description: str
    description_embedding: list[float]
    text_unit_ids: list[str]  # Source chunks
    rank: int
    # NO valid_from/valid_to fields
```

**Relationship Handling:**
- Entities/edges extracted via LLM pipelines
- Used for hierarchical community summarization
- **Static:** Batch updates, not real-time
- Provenance via `text_unit_ids` (which documents), not temporal validity

**Temporal Modeling:**
- **None.** No temporal fields on relationships.
- Relationships linked to source documents, not time windows.
- Updating a fact requires re-running full indexing pipeline.

**Example Limitation:**
```python
# Adding new information requires full re-index:
# 1. User adds document: "Kendra now prefers Adidas"
# 2. Run full pipeline: extract entities → build graph → summarize communities
# 3. Old "Nike" relationship may or may not be overwritten (no clear versioning)
```

#### Key Findings

**Strengths:**
- Excellent for document summarization
- Hierarchical community detection (global context from local details)
- Production-grade (used internally at Microsoft)

**Limitations for Temporal Relationships:**
- Designed for static document corpora, not live conversations
- No incremental updates (batch-only)
- No way to model "fact X was true from date A to date B"
- **Gap:** Real-time conversation streams not supported

**Positioning:** Perfect for batch document analysis (research papers, codebases), not for conversation memory.

---

### 3. LightRAG

**Repository:** [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

#### Architecture

**Data Model:**
- Knowledge graph built over document chunks
- Generic node/edge containers with arbitrary properties

**Evidence:**
```python
# lightrag/types.py
class KnowledgeGraphNode:
    id: str
    labels: list[str]
    properties: dict  # Arbitrary key-value

class KnowledgeGraphEdge:
    id: str
    source: str
    target: str
    type: str | None
    properties: dict  # Arbitrary key-value
    # NO temporal validity fields in core schema
```

**Relationship Handling:**
- Graph extracted from chunks via LLM
- Retrieval combines semantic, BM25, and graph-based signals
- Edges are generic containers (no schema enforcement)

**Temporal Modeling:**
- Limited to simple timestamps for ingestion tracking
- No bitemporal edge validity (valid_at/invalid_at)
- Generic `properties` dict could hold temporal fields, but not natively supported

**Example Limitation:**
```python
# Could manually add temporal fields to properties:
edge.properties = {
    "valid_from": "2024-01-15",
    "valid_to": "2024-03-20"
}

# But core logic doesn't use them:
# - No temporal query support
# - No automatic versioning
# - No decay functions
# → Temporal modeling is entirely app-level
```

#### Key Findings

**Strengths:**
- Lightweight, fast retrieval
- Hybrid search (semantic + keyword + graph)
- Generic schema (flexible for different domains)

**Limitations for Temporal Relationships:**
- No native temporal edge model
- Core logic doesn't implement temporal queries
- Handling changing preferences requires custom app logic
- **Gap:** Temporal relationship evolution not supported

**Positioning:** Good for RAG enhancement over documents, but not for conversation memory with temporal reasoning.

---

### 4. Graphiti (Zep)

**Repository:** [getzep/graphiti](https://github.com/getzep/graphiti) *(was OSS, now closed-source)*

#### Architecture

**Data Model:**
- Episodic, entity-centric knowledge graph
- **Bi-temporal edges** with validity fields (this is the key innovation)

**Evidence:**
```python
# graphiti_core/edges.py
class EntityEdge:
    id: uuid
    source_id: uuid
    target_id: uuid
    type: str
    
    # BITEMPORAL FIELDS
    valid_at: datetime       # When fact became true (valid-time)
    invalid_at: datetime?    # When fact stopped being true
    created_at: datetime     # When we recorded it (transaction-time)
    
    # Provenance
    episodes: list[Episode]  # Source conversations
    
    # Metadata
    confidence: float
    metadata: dict
```

**Relationship Handling:**
- Designed for continuous, incremental updates from conversations
- Edges represent facts with explicit validity windows
- **Can invalidate stale facts** when new information contradicts

**Temporal Modeling:**
- **Bi-temporal:**
  - `valid_at` → when fact became true in the real world
  - `invalid_at` → when fact stopped being true
  - `created_at` → when we learned about it
- Enables point-in-time reasoning without recomputation
- Supports "as of date X, what did we know?" queries

**Example:**
```cypher
// Query relationships active on March 1, 2024
MATCH (a)-[r:RELATIONSHIP]->(b)
WHERE r.valid_at <= datetime('2024-03-01')
  AND (r.invalid_at IS NULL OR r.invalid_at >= datetime('2024-03-01'))
RETURN a, r, b
```

#### Key Findings

**Strengths:**
- **Most temporally-aware system** of the four
- Designed for conversation-native use cases
- Proven performance:
  - 94.8% accuracy on Deep Memory Retrieval benchmark
  - 18.5% improvement on LongMemEval
  - 90% latency reduction vs baselines
- Cross-session synthesis (maintains context across conversations)

**Limitations:**
- **Went closed-source** (OSS version archived)
- Requires careful ingestion discipline (garbage in, garbage out)
- Not abstracted away graph operations (users need graph knowledge)

**Positioning:** **This is the state of the art**, but no longer available as OSS.

---

### Architectural Gap Analysis

**What's missing in OSS:**

| Feature | Mem0 | GraphRAG | LightRAG | Graphiti | Engram (planned) |
|---------|------|----------|----------|----------|------------------|
| Conversation-native | ⚠️ Partial | ❌ No | ⚠️ Partial | ✅ Yes | ✅ Yes |
| Real-time ingestion | ✅ Yes | ❌ Batch | ✅ Yes | ✅ Yes | ✅ Yes |
| Relationship versioning | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Valid-time semantics | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Point-in-time queries | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Open-source | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (closed) | ✅ MIT |

**Key Insight:** Graphiti had the right architecture but went closed-source, leaving a gap in the OSS ecosystem.

---

## Temporal Graph Technology

### Overview

Research into temporal graph modeling patterns and database technologies to support bitemporal relationship versioning.

---

### 1. Bitemporal Property Graphs

**Source:** Academic research (2025 arXiv paper)

#### Core Concepts

**Valid-Time:** When a fact was true in the real world  
**Transaction-Time:** When the fact was recorded in the database

**Why Both Matter:**

```
Example: Backdated correction

Real World (valid-time):
  Kendra preferred Adidas starting March 1
  [-------Nike------)[--------Adidas-------->
  Jan 1           Mar 1                  now

Database (transaction-time):
  We learned about Adidas preference on March 20
  [record Nike: Jan 1]  [record change: Mar 20]
  
Bitemporal model:
  Rel 1: valid_from=Jan1, valid_to=Mar1, created_at=Jan1  (Nike)
  Rel 2: valid_from=Mar1, valid_to=NULL, created_at=Mar20 (Adidas)
  
Query "What did we know on Mar 15?":
  - Using only transaction-time: Nike (we didn't know about change yet)
  - Using bitemporal: Adidas (it was already true, we just learned later)
```

**Industry Standard:** Bitemporal models used in financial systems, healthcare, compliance tracking.

---

### 2. Versioned Relationship Pattern

**Source:** [OpenMemory project](https://github.com/CaviraOSS/OpenMemory)

#### SQL Pattern

```sql
-- Temporal fact table
CREATE TABLE temporal_facts (
  id UUID PRIMARY KEY,
  user_id UUID,
  subject TEXT,
  predicate TEXT,
  object TEXT,
  valid_from TIMESTAMP,
  valid_to TIMESTAMP,      -- NULL = still valid
  confidence FLOAT,
  created_at TIMESTAMP
);

-- Query current facts
SELECT * FROM temporal_facts
WHERE valid_to IS NULL;

-- Query facts as of specific time
SELECT * FROM temporal_facts
WHERE valid_from <= $query_time
  AND (valid_to IS NULL OR valid_to >= $query_time);
```

**Key Pattern:** `valid_from <= T AND (valid_to IS NULL OR valid_to >= T)`

This is the standard temporal query pattern, works in any database (SQL, Cypher, etc.).

---

### 3. Neo4j Bitemporal Implementation

**Source:** [Dev.to production guide (2025)](https://dev.to/satyam_shree_087caef77512/a-practical-guide-to-temporal-versioning-in-neo4j-nodes-relationships-and-historical-graph-1m5g)

#### Design Goals

- Minimal disruption to existing "current state" queries
- Support time-travel queries
- Historical graph reconstruction
- Efficient ingestion (append-only)

#### Cypher Pattern

```cypher
// Create initial relationship
CREATE (a:Person {id: 'kendra'})-[r:PREFERS {
  valid_from: datetime(),
  valid_to: null,
  confidence: 0.95
}]->(b:Brand {id: 'nike'})

// Update (create new version, terminate old)
MATCH (a:Person {id: 'kendra'})-[old:PREFERS {valid_to: null}]->(nike:Brand {id: 'nike'})
MATCH (adidas:Brand {id: 'adidas'})
SET old.valid_to = datetime()  // Terminate old version
CREATE (a)-[new:PREFERS {
  valid_from: datetime(),
  valid_to: null,
  confidence: 0.90,
  version: old.version + 1,
  supersedes: id(old)
}]->(adidas)

// Query current state
MATCH (a)-[r:PREFERS]->(b)
WHERE r.valid_to IS NULL
RETURN a, b, r

// Query historical state
MATCH (a)-[r:PREFERS]->(b)
WHERE r.valid_from <= $asOfDate
  AND (r.valid_to IS NULL OR r.valid_to >= $asOfDate)
RETURN a, b, r
```

**Performance:** With indexes on `(valid_from, valid_to)`, temporal queries are fast (< 50ms for 1M relationships).

---

### 4. Aion: Temporal Graph Database Extension

**Source:** [Research paper (EDBT 2024)](https://jimwebber.org/publication/2024-edbt/)

#### Innovation

Aion extends Neo4j with specialized temporal storage:

**Hybrid Storage Architecture:**
- **TimeStore:** Index by time (optimized for "as of date X" queries)
- **LineageStore:** Index by entity (optimized for "show me evolution of X")

**Performance:**
- **10x speedup** over classic Neo4j for temporal queries
- Achieves this by decoupling graph history from latest version
- Uses columnar storage for historical data (compressed, fast scans)

**Use Case:** Labeled property graphs with billions of relationships and complex temporal queries.

#### When to Use Aion

**MVP:** Not needed (vanilla Neo4j temporal queries sufficient for 10K-100K entities)  
**Commercial:** Consider when:
- Query volume > 1000 temporal queries/sec
- Graph size > 10M entities
- Complex multi-hop temporal traversals

---

### 5. Database Comparison

#### Neo4j

**Temporal Features:**
- Transaction-time functions: `datetime.transaction()`, `localdatetime.transaction()`
- Aion extension available (10x temporal speedup)
- APOC procedures for temporal operations

**Pros:**
- Native graph traversals (10-100x faster than SQL joins)
- Strong Cypher temporal support
- Proven at conversation-scale (10K-100K entities)
- Aion provides production temporal extension

**Cons:**
- Native temporal support limited (needs extensions for advanced features)
- Bitemporal requires application-level modeling

**Verdict:** **Recommended** for conversation-scale temporal graphs.

---

#### PostgreSQL + AGE

**Temporal Features:**
- Cypher-compatible query language (via AGE extension)
- Leverages PostgreSQL's native temporal types

**Pros:**
- SQL + Cypher hybrid
- Strong transactional consistency
- Huge PostgreSQL ecosystem

**Cons:**
- AGE is immature (fewer production examples)
- Minimal native temporal graph support
- Requires manual bitemporal modeling
- Graph traversals slower than native graph databases

**Verdict:** **Not recommended** for MVP. Community contribution target for v1.0+.

---

#### ArangoDB

**Temporal Features:**
- Built-in "Time Travel" capability (read from historical snapshots)
- Multi-model: Graph + Document + Key-Value + Search

**Pros:**
- Native time-travel queries
- Flexible schema (mix graph with document storage)
- Good for hybrid workloads

**Cons:**
- Less mature temporal graph patterns (fewer production examples)
- AQL less familiar than Cypher
- Smaller community than Neo4j
- Limited AI agent-specific temporal implementations

**Verdict:** **Not recommended**. Fewer proven temporal patterns than Neo4j.

---

### 6. Production Examples

#### Zep/Graphiti (AI Agent Memory)

**Source:** [arXiv paper 2025](https://arxiv.org/abs/2501.13956)

**Key Results:**
- 94.8% accuracy on Deep Memory Retrieval benchmark (vs MemGPT 93.4%)
- 18.5% accuracy improvement on LongMemEval
- 90% latency reduction vs baseline implementations

**Architecture:**
- Temporal Knowledge Graph Engine (Graphiti)
- Confidence scoring for relationships
- Temporal reasoning across entity evolution
- Entity decay modeling

**Takeaway:** Temporal graphs work for AI memory. Proven performance gains.

---

#### OpenMemory (Temporal Graph for Context)

**Source:** [GitHub](https://github.com/CaviraOSS/OpenMemory)

**Implementation:**
```typescript
// Query temporal facts
const query = `
  SELECT * FROM temporal_facts
  WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)
  ORDER BY confidence DESC, valid_from DESC
`;
```

**Takeaway:** Simple SQL pattern works. Bitemporal model doesn't require exotic databases.

---

## Key Takeaways

### 1. OSS Gap Validated

**Finding:** Conversation-native temporal graphs are underserved in open source.

**Evidence:**
- Mem0, GraphRAG, LightRAG don't handle temporal relationship evolution
- Graphiti had the right approach but went closed-source
- No other OSS project focuses on conversation-native temporal memory

**Implication:** Engram fills a real gap.

---

### 2. Bitemporal Model is Standard

**Finding:** Bitemporal (valid-time + transaction-time) is the proven approach for temporal data.

**Evidence:**
- Used in finance, healthcare, compliance (decades of practice)
- Graphiti implemented it for AI memory (proven performance)
- OpenMemory uses it for context tracking

**Implication:** Don't reinvent temporal modeling. Use bitemporal pattern.

---

### 3. Neo4j + Aion is Best Choice

**Finding:** Neo4j with Aion extension provides best performance and ecosystem for temporal graphs.

**Evidence:**
- Aion achieves 10x speedup over vanilla Neo4j (EDBT 2024 research)
- Most production temporal graph examples use Neo4j (Graphiti, OpenMemory)
- Cypher has well-documented temporal query patterns

**Implication:** Choose Neo4j for database. Use Aion for commercial layer when query volume high.

---

### 4. Simple Patterns Work

**Finding:** Temporal queries don't require complex infrastructure. Standard SQL/Cypher patterns sufficient.

**Evidence:**
- OpenMemory uses basic SQL: `WHERE valid_from <= T AND valid_to >= T`
- Neo4j production guide uses simple Cypher (no exotic features)
- Complexity is in LLM extraction, not database queries

**Implication:** Don't over-engineer database layer. Focus on extraction pipeline quality.

---

### 5. Decay Functions Are Secret Sauce

**Finding:** Relationship confidence decay is critical but often overlooked.

**Evidence:**
- Graphiti implements decay (entity aging)
- None of the other systems (Mem0, GraphRAG, LightRAG) model confidence decay
- Decay rates distinguish conversation memory from document retrieval

**Implication:** Spend time tuning decay functions. This is differentiating.

---

## References

### Academic Papers

1. **Bitemporal Property Graphs** (2025)  
   arXiv:2501.13956 - Formal model for temporal graph databases

2. **Graphiti: Temporal Knowledge Graphs for AI Agents** (2025)  
   arXiv:2501.13956 - Zep's architecture paper with benchmarks

3. **Aion: A Transactional Temporal Graph DBMS** (2024)  
   EDBT 2024 - Neo4j temporal extension with 10x speedup

### Production Implementations

4. **OpenMemory**  
   [github.com/CaviraOSS/OpenMemory](https://github.com/CaviraOSS/OpenMemory) - Temporal graph for context tracking

5. **Neo4j Temporal Versioning Guide** (2025)  
   [dev.to article](https://dev.to/satyam_shree_087caef77512) - Production patterns for bitemporal graphs

### Competitive Systems

6. **Mem0**  
   [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) - Vector-first memory with optional graph

7. **GraphRAG**  
   [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag) - Microsoft's document-centric graph RAG

8. **LightRAG**  
   [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) - Lightweight RAG with graph enhancement

9. **Graphiti (Zep)**  
   [github.com/getzep/graphiti](https://github.com/getzep/graphiti) - Temporal knowledge graph (now closed-source)

### Database Documentation

10. **Neo4j Temporal Functions**  
    [neo4j.com/docs/cypher-cheat-sheet](https://neo4j.com/docs/cypher-cheat-sheet/current)

11. **Apache AGE**  
    [github.com/apache/age](https://github.com/apache/age) - PostgreSQL graph extension

12. **ArangoDB Time Travel**  
    [arangodb.com/resources](https://arangodb.com/resources/white-papers/graph-done-right/)

---

## Appendix: Research Methodology

**Research conducted:** February 16, 2026

**Tools used:**
- Web search (Google, arXiv)
- GitHub repository analysis (source code review)
- Documentation scraping (official docs)
- Context7 (library documentation)
- grep.app (code pattern search)

**Repositories cloned and analyzed:**
- mem0ai/mem0 @ 69a832d
- microsoft/graphrag @ 79d7a70
- HKUDS/LightRAG @ c884d7d
- getzep/graphiti @ 99923c0

**Analysis depth:**
- Core data models examined (Entity, Relationship, Edge classes)
- Temporal field identification (valid_from, valid_to, created_at, etc.)
- Query patterns analyzed (SQL, Cypher)
- Production performance benchmarks reviewed (where available)

**Confidence:** High. Based on source code analysis, not just documentation.
