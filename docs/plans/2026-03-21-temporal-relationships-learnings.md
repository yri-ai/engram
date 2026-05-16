# Temporal-Relationships Learnings Implementation Plan

> **Note for contributors:** Implement this plan task-by-task, ensuring each step is completed and validated before moving on.

**Goal:** Port the 4 most impactful patterns from temporal-relationships into engram: standalone Fact model, richer LLM extraction context, conversation state snapshots with delta tracking, and conversation summary generation.

**Architecture:** Each improvement is a vertical slice — model, storage, service, prompt, test. Facts fill the modeling gap where knowledge doesn't fit entity→entity edges. Richer context improves extraction quality by giving the LLM prior knowledge during inference. Snapshots enable "what changed?" queries and make the system debuggable. Summaries provide high-level narrative context.

**Tech Stack:** Python 3.11+, Pydantic, FastAPI, Neo4j, pytest, Jinja2

---

## Analysis: What temporal-relationships does better

After reviewing the temporal-relationships Go codebase, these are the key patterns engram should adopt:

| Pattern | temporal-relationships | engram today | Gap |
|---------|----------------------|--------------|-----|
| **Standalone facts** | `Fact` artifact with confidence, supersession chains, temporal status | Everything forced into entity→entity relationships | "Leo is 32" has no natural home |
| **Prior extraction context** | LLM gets prior facts, relationships, patterns, arcs when extracting | LLM gets only recent entity names (20 items) | Extraction misses continuity — can't recognize supersession |
| **State snapshots** | `BuildForRun` generates snapshot + delta after each extraction | No snapshot mechanism; raw graph only | Can't answer "what changed in this message?" |
| **Conversation summaries** | `SessionArc` with opening→shift→closing narrative | Nothing | No high-level conversation view |

---

## Task 1: Fact Model

**Files:**
- Create: `src/engram/models/fact.py`
- Modify: `src/engram/storage/base.py`
- Modify: `src/engram/storage/memory.py`
- Modify: `src/engram/storage/neo4j.py`
- Test: `tests/unit/test_models.py`
- Test: `tests/unit/test_memory_store.py`

**Step 1: Write the Fact model test**

```python
# Add to tests/unit/test_models.py

from engram.models.fact import Fact, FactStatus

def test_fact_build_id():
    fid = Fact.build_id("t1", "msg1", 0)
    assert fid == "t1:fact:msg1:0"

def test_fact_defaults():
    f = Fact(
        id="t1:fact:msg1:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        entity_id="t1:c1:PERSON:alice",
        fact_key="age",
        fact_text="Alice is 32 years old",
        confidence=0.9,
    )
    assert f.status == FactStatus.ACTIVE
    assert f.supersedes_fact_id is None
    assert f.valid_to is None

def test_fact_supersession():
    f = Fact(
        id="t1:fact:msg2:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg2",
        entity_id="t1:c1:PERSON:alice",
        fact_key="age",
        fact_text="Alice is 33 now",
        confidence=1.0,
        supersedes_fact_id="t1:fact:msg1:0",
    )
    assert f.supersedes_fact_id == "t1:fact:msg1:0"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_models.py::test_fact_build_id -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engram.models.fact'`

**Step 3: Write the Fact model**

Create `src/engram/models/fact.py`:

```python
"""Standalone knowledge claims with temporal tracking and supersession."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class Fact(BaseModel):
    """A standalone knowledge claim about an entity.

    Unlike relationships (entity→entity edges), facts capture knowledge
    that belongs to a single entity: "Alice is 32", "Bob works at Acme",
    "The project deadline is March 30th".

    Inspired by temporal-relationships' Fact artifact type.
    """

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str | None = None

    # What entity this fact is about
    entity_id: str

    # The fact itself
    fact_key: str  # Semantic key for grouping (e.g., "age", "employer", "location")
    fact_text: str  # Human-readable fact statement
    confidence: float = Field(ge=0.0, le=1.0)

    # Status & supersession
    status: FactStatus = FactStatus.ACTIVE
    supersedes_fact_id: str | None = None  # Chain: new fact supersedes old

    # Bitemporal
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_to: datetime | None = None
    recorded_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recorded_to: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, message_id: str, index: int) -> str:
        """Build deterministic fact ID."""
        return f"{tenant_id}:fact:{message_id}:{index}"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_models.py -k fact -v`
Expected: PASS

**Step 5: Add Fact storage methods to GraphStore base**

Add these abstract methods to `src/engram/storage/base.py`:

```python
# In the imports section, add:
from engram.models.fact import Fact

# After the commitment methods, add:

# --- Fact Operations ---

@abstractmethod
async def save_fact(self, fact: Fact) -> Fact:
    """Save a fact (knowledge claim about an entity)."""
    ...

@abstractmethod
async def get_facts(
    self,
    tenant_id: str,
    entity_id: str,
    fact_key: str | None = None,
    active_only: bool = True,
) -> list[Fact]:
    """Get facts for an entity, optionally filtered by key."""
    ...

@abstractmethod
async def supersede_fact(
    self,
    old_fact_id: str,
    new_fact: Fact,
) -> Fact:
    """Mark old fact as superseded and save the new one."""
    ...
```

**Step 6: Write MemoryStore Fact tests**

Add to `tests/unit/test_memory_store.py`:

```python
from engram.models.fact import Fact, FactStatus

async def test_save_and_get_facts(memory_store):
    f = Fact(
        id="t1:fact:msg1:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        entity_id="t1:c1:PERSON:alice",
        fact_key="age",
        fact_text="Alice is 32",
        confidence=0.9,
    )
    await memory_store.save_fact(f)
    facts = await memory_store.get_facts("t1", "t1:c1:PERSON:alice")
    assert len(facts) == 1
    assert facts[0].fact_text == "Alice is 32"

async def test_get_facts_by_key(memory_store):
    f1 = Fact(id="t1:fact:m1:0", tenant_id="t1", conversation_id="c1",
              message_id="m1", entity_id="e1", fact_key="age",
              fact_text="Alice is 32", confidence=0.9)
    f2 = Fact(id="t1:fact:m1:1", tenant_id="t1", conversation_id="c1",
              message_id="m1", entity_id="e1", fact_key="location",
              fact_text="Alice lives in NYC", confidence=0.8)
    await memory_store.save_fact(f1)
    await memory_store.save_fact(f2)
    age_facts = await memory_store.get_facts("t1", "e1", fact_key="age")
    assert len(age_facts) == 1
    assert age_facts[0].fact_key == "age"

async def test_supersede_fact(memory_store):
    old = Fact(id="t1:fact:m1:0", tenant_id="t1", conversation_id="c1",
               message_id="m1", entity_id="e1", fact_key="age",
               fact_text="Alice is 32", confidence=0.9)
    await memory_store.save_fact(old)

    new = Fact(id="t1:fact:m2:0", tenant_id="t1", conversation_id="c1",
               message_id="m2", entity_id="e1", fact_key="age",
               fact_text="Alice is 33 now", confidence=1.0,
               supersedes_fact_id="t1:fact:m1:0")
    result = await memory_store.supersede_fact("t1:fact:m1:0", new)

    assert result.status == FactStatus.ACTIVE
    # Old fact should be superseded
    all_facts = await memory_store.get_facts("t1", "e1", active_only=False)
    old_fact = next(f for f in all_facts if f.id == "t1:fact:m1:0")
    assert old_fact.status == FactStatus.SUPERSEDED
```

**Step 7: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_memory_store.py -k fact -v`
Expected: FAIL (methods not implemented)

**Step 8: Implement MemoryStore Fact methods**

Add to `src/engram/storage/memory.py`:

```python
# Add _facts dict to __init__:
self._facts: dict[str, Fact] = {}

# Implement methods:
async def save_fact(self, fact: Fact) -> Fact:
    self._facts[fact.id] = fact
    return fact

async def get_facts(
    self,
    tenant_id: str,
    entity_id: str,
    fact_key: str | None = None,
    active_only: bool = True,
) -> list[Fact]:
    results = []
    for f in self._facts.values():
        if f.tenant_id != tenant_id or f.entity_id != entity_id:
            continue
        if fact_key and f.fact_key != fact_key:
            continue
        if active_only and f.status != FactStatus.ACTIVE:
            continue
        results.append(f)
    return results

async def supersede_fact(self, old_fact_id: str, new_fact: Fact) -> Fact:
    if old_fact_id in self._facts:
        old = self._facts[old_fact_id]
        old.status = FactStatus.SUPERSEDED
        old.valid_to = new_fact.valid_from
    new_fact.supersedes_fact_id = old_fact_id
    self._facts[new_fact.id] = new_fact
    return new_fact
```

**Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_memory_store.py -k fact -v`
Expected: PASS

**Step 10: Implement Neo4jStore Fact methods**

Add to `src/engram/storage/neo4j.py` — use `(:Entity)-[:HAS_FACT]->(:Fact)` pattern:

```python
async def save_fact(self, fact: Fact) -> Fact:
    query = """
    MERGE (f:Fact {id: $id})
    SET f += $props
    WITH f
    MATCH (e:Entity {id: $entity_id})
    MERGE (e)-[:HAS_FACT]->(f)
    RETURN f
    """
    props = {
        "id": fact.id,
        "tenant_id": fact.tenant_id,
        "conversation_id": fact.conversation_id,
        "message_id": fact.message_id,
        "extraction_run_id": fact.extraction_run_id,
        "entity_id": fact.entity_id,
        "fact_key": fact.fact_key,
        "fact_text": fact.fact_text,
        "confidence": fact.confidence,
        "status": fact.status.value,
        "supersedes_fact_id": fact.supersedes_fact_id,
        "valid_from": fact.valid_from.isoformat(),
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "recorded_from": fact.recorded_from.isoformat(),
        "recorded_to": fact.recorded_to.isoformat() if fact.recorded_to else None,
    }
    async with self._driver.session() as session:
        await session.run(query, id=fact.id, props=props, entity_id=fact.entity_id)
    return fact

async def get_facts(
    self,
    tenant_id: str,
    entity_id: str,
    fact_key: str | None = None,
    active_only: bool = True,
) -> list[Fact]:
    conditions = ["f.tenant_id = $tenant_id", "f.entity_id = $entity_id"]
    params: dict = {"tenant_id": tenant_id, "entity_id": entity_id}
    if fact_key:
        conditions.append("f.fact_key = $fact_key")
        params["fact_key"] = fact_key
    if active_only:
        conditions.append("f.status = 'active'")
    where = " AND ".join(conditions)
    query = f"MATCH (f:Fact) WHERE {where} RETURN f ORDER BY f.recorded_from DESC"
    async with self._driver.session() as session:
        result = await session.run(query, **params)
        records = await result.data()
    return [self._record_to_fact(r["f"]) for r in records]

async def supersede_fact(self, old_fact_id: str, new_fact: Fact) -> Fact:
    query = """
    MATCH (old:Fact {id: $old_id})
    SET old.status = 'superseded', old.valid_to = $valid_to
    """
    async with self._driver.session() as session:
        await session.run(query, old_id=old_fact_id,
                         valid_to=new_fact.valid_from.isoformat())
    new_fact.supersedes_fact_id = old_fact_id
    return await self.save_fact(new_fact)
```

**Step 11: Commit**

```bash
git add src/engram/models/fact.py src/engram/storage/base.py src/engram/storage/memory.py src/engram/storage/neo4j.py tests/unit/test_models.py tests/unit/test_memory_store.py
git commit -m "feat: add Fact model for standalone knowledge claims with supersession"
```

---

## Task 2: Fact Extraction Pipeline Stage

**Files:**
- Create: `config/prompts/fact_extraction.jinja2`
- Modify: `src/engram/services/extraction.py`
- Test: `tests/unit/test_extraction.py`

**Step 1: Write the fact extraction test**

Add to `tests/unit/test_extraction.py`:

```python
async def test_fact_extraction(pipeline, mock_llm, memory_store):
    """Test that facts are extracted from messages."""
    # Mock LLM returns for entity extraction, relationship inference, AND fact extraction
    mock_llm.complete_json.side_effect = [
        {"entities": [{"name": "Alice", "canonical": "alice", "type": "PERSON", "confidence": 1.0}]},
        {"relationships": []},
        {"facts": [
            {"entity_mention": "Alice", "fact_key": "age", "fact_text": "Alice is 32 years old", "confidence": 0.9},
        ]},
        {"commitments": []},
    ]

    request = IngestRequest(
        text="Alice mentioned she is 32 years old",
        speaker="Alice",
        timestamp=datetime.now(UTC),
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)
    assert response.entities_extracted == 1

    # Verify fact was saved
    facts = await memory_store.get_facts("t1", "t1:c1:PERSON:alice")
    assert len(facts) == 1
    assert facts[0].fact_key == "age"
    assert facts[0].fact_text == "Alice is 32 years old"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_extraction.py::test_fact_extraction -v`
Expected: FAIL

**Step 3: Create the fact extraction prompt template**

Create `config/prompts/fact_extraction.jinja2`:

```
You are extracting standalone facts about entities from a conversation message.

Message: "{{ message_text }}"
Timestamp: {{ timestamp }}
Speaker: {{ speaker }}

Extracted Entities:
{{ entities | tojson(indent=2) }}

{% if existing_facts %}
Known facts about these entities:
{{ existing_facts | tojson(indent=2) }}
{% endif %}

Extract FACTS — knowledge claims about a single entity that are NOT relationships
between two entities. Examples:
- "Alice is 32 years old" -> fact about Alice (key: "age")
- "Bob works at Acme Corp" -> fact about Bob (key: "employer")
- "The deadline is March 30th" -> fact about the deadline event (key: "date")

Rules:
1. Only extract facts explicitly stated or strongly implied.
2. Each fact needs a semantic key for grouping (e.g., "age", "location", "employer").
3. If a fact contradicts a known fact, it supersedes the old one.
4. Do NOT extract relationships between entities — those are handled separately.

Return JSON:
{
  "facts": [
    {
      "entity_mention": "exact entity name from the message",
      "fact_key": "semantic category (age, location, employer, etc.)",
      "fact_text": "human-readable fact statement",
      "confidence": 0.0-1.0,
      "supersedes_key": "fact_key of the old fact this replaces, if any"
    }
  ]
}

If no standalone facts are found, return an empty list.
```

**Step 4: Add fact extraction stage to ExtractionPipeline**

Add `_extract_facts` method to `src/engram/services/extraction.py` (after `_extract_commitments`), and call it from `process_message` between relationship inference and commitment extraction:

```python
async def _extract_facts(
    self,
    request: IngestRequest,
    entities: list[Entity],
    message_id: str,
    run_id: str,
) -> list[Fact]:
    """Extract standalone facts about entities via LLM."""
    entity_dicts = [{"name": e.canonical_name, "type": e.entity_type.value} for e in entities]

    # Gather existing facts for context
    existing_facts: list[dict[str, Any]] = []
    for entity in entities:
        facts = await self._store.get_facts(request.tenant_id, entity.id)
        for f in facts:
            existing_facts.append({
                "entity": entity.canonical_name,
                "key": f.fact_key,
                "text": f.fact_text,
                "confidence": f.confidence,
            })

    prompt = self._config_service.render_prompt(
        "fact_extraction.jinja2",
        message_text=request.text,
        speaker=request.speaker,
        timestamp=request.timestamp.isoformat(),
        entities=entity_dicts,
        existing_facts=existing_facts if existing_facts else None,
    )

    result = await self._llm.complete_json(prompt)
    raw_items: list[dict[str, Any]] = result.get("facts", [])

    facts: list[Fact] = []
    for i, item in enumerate(raw_items):
        entity = self._find_entity_by_mention(item.get("entity_mention", ""), entities)
        if not entity:
            continue

        fact = Fact(
            id=Fact.build_id(request.tenant_id, message_id, i),
            tenant_id=request.tenant_id,
            conversation_id=request.conversation_id,
            message_id=message_id,
            extraction_run_id=run_id,
            entity_id=entity.id,
            fact_key=item["fact_key"],
            fact_text=item["fact_text"],
            confidence=snap_confidence(item.get("confidence", 0.8)),
            valid_from=request.timestamp,
        )

        # Handle supersession: if this fact replaces an existing one
        supersedes_key = item.get("supersedes_key")
        if supersedes_key:
            existing = await self._store.get_facts(
                request.tenant_id, entity.id, fact_key=supersedes_key
            )
            if existing:
                await self._store.supersede_fact(existing[0].id, fact)
                facts.append(fact)
                continue

        await self._store.save_fact(fact)
        facts.append(fact)

    return facts
```

In `process_message`, add the call after relationship inference (Step 3):

```python
# Step 3.5: Fact Extraction
await self._extract_facts(request, entities, message_id, run.id)
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_extraction.py::test_fact_extraction -v`
Expected: PASS

**Step 6: Commit**

```bash
git add config/prompts/fact_extraction.jinja2 src/engram/services/extraction.py tests/unit/test_extraction.py
git commit -m "feat: add fact extraction pipeline stage for standalone knowledge claims"
```

---

## Task 3: Richer LLM Context (Prior Facts + Relationships in Prompts)

**Files:**
- Modify: `src/engram/services/extraction.py` (the `_get_context_entities` and `_extract_entities` methods)
- Modify: `config/prompts/entity_extraction.jinja2`
- Modify: `config/prompts/relationship_inference.jinja2`
- Test: `tests/unit/test_extraction.py`

This is the single highest-impact improvement from temporal-relationships. Currently, engram passes only entity names as context. temporal-relationships passes prior facts, relationships, and patterns — enabling the LLM to recognize when information supersedes prior knowledge.

**Step 1: Write a test for enriched context**

Add to `tests/unit/test_extraction.py`:

```python
async def test_entity_extraction_receives_prior_knowledge_context(pipeline, mock_llm, memory_store):
    """LLM should receive prior facts and relationships as context, not just entity names."""
    # Pre-populate store with existing entities, relationships, and facts
    from engram.models.entity import Entity, EntityType
    from engram.models.relationship import Relationship, RelationshipType
    from engram.models.fact import Fact

    entity = Entity(
        id="t1:c1:PERSON:alice", tenant_id="t1", conversation_id="c1",
        entity_type=EntityType.PERSON, canonical_name="alice",
        last_mentioned=datetime.now(UTC),
    )
    await memory_store.upsert_entity(entity)

    fact = Fact(
        id="t1:fact:old:0", tenant_id="t1", conversation_id="c1",
        message_id="old", entity_id="t1:c1:PERSON:alice",
        fact_key="age", fact_text="Alice is 32", confidence=0.9,
    )
    await memory_store.save_fact(fact)

    # Mock LLM — we just want to verify the prompt includes prior knowledge
    mock_llm.complete_json.side_effect = [
        {"entities": []},  # entity extraction
    ]

    request = IngestRequest(
        text="Alice said something",
        speaker="Bob",
        timestamp=datetime.now(UTC),
        conversation_id="c1",
        tenant_id="t1",
    )
    await pipeline.process_message(request)

    # Verify the prompt included prior facts context
    call_args = mock_llm.complete_json.call_args_list[0]
    prompt_text = call_args[0][0]
    assert "alice" in prompt_text.lower()
    assert "32" in prompt_text  # Prior fact should appear in context
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_extraction.py::test_entity_extraction_receives_prior_knowledge_context -v`
Expected: FAIL (prompt doesn't include facts)

**Step 3: Enrich context-gathering in extraction.py**

Replace `_get_context_entities` with a richer `_get_extraction_context`:

```python
async def _get_extraction_context(self, request: IngestRequest) -> dict[str, Any]:
    """Build rich context for LLM extraction: recent entities + their facts + active relationships.

    Inspired by temporal-relationships' approach of giving the LLM full prior
    knowledge during extraction, not just entity names.
    """
    since = request.timestamp - timedelta(days=_CONTEXT_LOOKBACK_DAYS)
    recent_entities = await self._store.get_recent_entities(
        tenant_id=request.tenant_id,
        conversation_id=request.conversation_id,
        since=since,
        limit=20,
    )

    entity_context: list[dict[str, Any]] = []
    for e in recent_entities:
        entry: dict[str, Any] = {"name": e.canonical_name, "type": e.entity_type.value}

        # Attach known facts
        facts = await self._store.get_facts(request.tenant_id, e.id)
        if facts:
            entry["known_facts"] = [
                {"key": f.fact_key, "text": f.fact_text, "confidence": f.confidence}
                for f in facts[:5]  # Limit to avoid token bloat
            ]

        entity_context.append(entry)

    # Active relationships between recent entities
    rel_context: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for e in recent_entities:
        rels = await self._store.get_active_relationships(e.id)
        for r in rels:
            key = (r.source_id, r.target_id, r.rel_type)
            if key not in seen:
                seen.add(key)
                rel_context.append({
                    "source": r.source_id.split(":")[-1],
                    "target": r.target_id.split(":")[-1],
                    "type": r.rel_type.value if isinstance(r.rel_type, RelationshipType) else str(r.rel_type),
                    "evidence": r.evidence[:100] if r.evidence else "",
                })

    return {
        "entities": entity_context,
        "relationships": rel_context[:20],  # Limit to avoid token bloat
    }
```

Update `_extract_entities` to use the richer context and pass it to the prompt.

**Step 4: Update entity_extraction.jinja2 prompt**

```
You are extracting entities from a conversation message for a knowledge graph.

Message: "{{ message_text }}"
Timestamp: {{ timestamp }}
Speaker: {{ speaker }}

Prior knowledge about this conversation:

Entities mentioned recently:
{{ context.entities | tojson(indent=2) if context.entities else "(none)" }}

{% if context.relationships %}
Active relationships:
{{ context.relationships | tojson(indent=2) }}
{% endif %}

Use this prior knowledge to:
- Resolve pronouns and ambiguous references to known entities
- Recognize when new information supersedes what was known before
- Avoid creating duplicate entities for known individuals/concepts

Extract entities following these rules:
... (rest of existing prompt unchanged)
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_extraction.py::test_entity_extraction_receives_prior_knowledge_context -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/engram/services/extraction.py config/prompts/entity_extraction.jinja2 tests/unit/test_extraction.py
git commit -m "feat: enrich LLM extraction context with prior facts and relationships"
```

---

## Task 4: Conversation State Snapshots & Delta Tracking

**Files:**
- Create: `src/engram/models/snapshot.py`
- Create: `src/engram/services/snapshot.py`
- Test: `tests/unit/test_snapshot.py`
- Modify: `src/engram/services/extraction.py` (call snapshot after extraction)

Inspired by temporal-relationships' `BuildForRun` and delta tracking. After each message extraction, capture the conversation's current state and what changed.

**Step 1: Write snapshot model test**

Create `tests/unit/test_snapshot.py`:

```python
from datetime import UTC, datetime
from engram.models.snapshot import ConversationSnapshot, SnapshotDelta, ChangeType

def test_snapshot_creation():
    snap = ConversationSnapshot(
        id="snap-1",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        extraction_run_id="run1",
        entity_count=3,
        relationship_count=2,
        fact_count=1,
        entities=["alice", "bob", "running"],
        created_at=datetime.now(UTC),
    )
    assert snap.entity_count == 3

def test_delta_creation():
    delta = SnapshotDelta(
        change_type=ChangeType.ADDED,
        artifact_type="entity",
        artifact_id="t1:c1:PERSON:alice",
        summary="New entity: alice (PERSON)",
    )
    assert delta.change_type == ChangeType.ADDED
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_snapshot.py -v`
Expected: FAIL

**Step 3: Write the snapshot model**

Create `src/engram/models/snapshot.py`:

```python
"""Conversation state snapshots and delta tracking.

Inspired by temporal-relationships' BuildForRun/delta tracking pattern.
After each extraction, capture the conversation state and what changed.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChangeType(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    SUPERSEDED = "superseded"


class SnapshotDelta(BaseModel):
    """A single change that occurred during an extraction."""

    change_type: ChangeType
    artifact_type: str  # "entity", "relationship", "fact", "commitment"
    artifact_id: str
    summary: str  # Human-readable description


class ConversationSnapshot(BaseModel):
    """Point-in-time snapshot of a conversation's knowledge state."""

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str

    # Counts
    entity_count: int = 0
    relationship_count: int = 0
    fact_count: int = 0
    commitment_count: int = 0

    # Entity names (lightweight summary, not full objects)
    entities: list[str] = Field(default_factory=list)

    # What changed in this extraction
    deltas: list[SnapshotDelta] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_snapshot.py -v`
Expected: PASS

**Step 5: Write snapshot service test**

Add to `tests/unit/test_snapshot.py`:

```python
import pytest
from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.models.fact import Fact
from engram.services.snapshot import SnapshotService
from engram.storage.memory import MemoryStore

@pytest.fixture
def memory_store():
    store = MemoryStore()
    return store

@pytest.fixture
def snapshot_service(memory_store):
    return SnapshotService(memory_store)

@pytest.mark.asyncio
async def test_build_snapshot(snapshot_service, memory_store):
    # Add an entity
    e = Entity(id="t1:c1:PERSON:alice", tenant_id="t1", conversation_id="c1",
               entity_type=EntityType.PERSON, canonical_name="alice")
    await memory_store.upsert_entity(e)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1", conversation_id="c1",
        message_id="msg1", run_id="run1",
    )
    assert snap.entity_count == 1
    assert "alice" in snap.entities

@pytest.mark.asyncio
async def test_build_snapshot_with_deltas(snapshot_service, memory_store):
    e = Entity(id="t1:c1:PERSON:alice", tenant_id="t1", conversation_id="c1",
               entity_type=EntityType.PERSON, canonical_name="alice")
    await memory_store.upsert_entity(e)

    new_entities = [e]
    new_relationships = []
    new_facts = []

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1", conversation_id="c1",
        message_id="msg1", run_id="run1",
        new_entities=new_entities,
        new_relationships=new_relationships,
        new_facts=new_facts,
    )
    assert len(snap.deltas) == 1
    assert snap.deltas[0].change_type == ChangeType.ADDED
    assert snap.deltas[0].artifact_type == "entity"
```

**Step 6: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_snapshot.py -k build -v`
Expected: FAIL

**Step 7: Write snapshot service**

Create `src/engram/services/snapshot.py`:

```python
"""Snapshot service — captures conversation state after each extraction."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from engram.models.snapshot import ChangeType, ConversationSnapshot, SnapshotDelta

if TYPE_CHECKING:
    from engram.models.entity import Entity
    from engram.models.fact import Fact
    from engram.models.relationship import Relationship
    from engram.storage.base import GraphStore


class SnapshotService:
    """Builds conversation state snapshots after extraction runs."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    async def build_snapshot(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        run_id: str,
        new_entities: list[Entity] | None = None,
        new_relationships: list[Relationship] | None = None,
        new_facts: list[Fact] | None = None,
    ) -> ConversationSnapshot:
        """Build a snapshot of current conversation state with deltas."""
        # Get current state
        entities = await self._store.list_entities(
            tenant_id=tenant_id, conversation_id=conversation_id, limit=500,
        )

        # Build deltas from what was just extracted
        deltas: list[SnapshotDelta] = []
        if new_entities:
            for e in new_entities:
                deltas.append(SnapshotDelta(
                    change_type=ChangeType.ADDED,
                    artifact_type="entity",
                    artifact_id=e.id,
                    summary=f"New entity: {e.canonical_name} ({e.entity_type.value})",
                ))
        if new_relationships:
            for r in new_relationships:
                deltas.append(SnapshotDelta(
                    change_type=ChangeType.ADDED,
                    artifact_type="relationship",
                    artifact_id=f"{r.source_id}->{r.target_id}",
                    summary=f"{r.source_id.split(':')[-1]} {r.rel_type.value} {r.target_id.split(':')[-1]}",
                ))
        if new_facts:
            for f in new_facts:
                change = ChangeType.SUPERSEDED if f.supersedes_fact_id else ChangeType.ADDED
                deltas.append(SnapshotDelta(
                    change_type=change,
                    artifact_type="fact",
                    artifact_id=f.id,
                    summary=f"{f.fact_key}: {f.fact_text}",
                ))

        return ConversationSnapshot(
            id=f"snap-{uuid.uuid4()}",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_id=message_id,
            extraction_run_id=run_id,
            entity_count=len(entities),
            relationship_count=0,  # Could count via store query
            fact_count=len(new_facts) if new_facts else 0,
            entities=[e.canonical_name for e in entities],
            deltas=deltas,
        )
```

**Step 8: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_snapshot.py -v`
Expected: PASS

**Step 9: Integrate snapshot into extraction pipeline**

In `src/engram/services/extraction.py`, after all extraction stages complete (before the `run.status = RunStatus.COMPLETED` line), add:

```python
# Build snapshot of what changed
from engram.services.snapshot import SnapshotService
snapshot_service = SnapshotService(self._store)
snapshot = await snapshot_service.build_snapshot(
    tenant_id=request.tenant_id,
    conversation_id=request.conversation_id,
    message_id=message_id,
    run_id=run.id,
    new_entities=entities,
    new_relationships=relationships,
    new_facts=facts,
)
logger.info(
    "Extraction snapshot: %d entities, %d deltas for message %s",
    snapshot.entity_count, len(snapshot.deltas), message_id,
)
```

Note: For now, the snapshot is logged but not persisted. A future task could add snapshot storage + an API endpoint.

**Step 10: Commit**

```bash
git add src/engram/models/snapshot.py src/engram/services/snapshot.py tests/unit/test_snapshot.py src/engram/services/extraction.py
git commit -m "feat: add conversation state snapshots with delta tracking"
```

---

## Task 5: Conversation Summary (SessionArc)

**Files:**
- Create: `config/prompts/conversation_summary.jinja2`
- Create: `src/engram/models/summary.py`
- Modify: `src/engram/services/extraction.py`
- Test: `tests/unit/test_extraction.py`

Inspired by temporal-relationships' `SessionArc` — captures the narrative trajectory of a conversation segment.

**Step 1: Write summary model test**

Add to `tests/unit/test_models.py`:

```python
from engram.models.summary import ConversationSummary

def test_conversation_summary():
    s = ConversationSummary(
        id="t1:summary:msg1",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        opening_state="Discussion about fitness goals",
        key_shift="Alice revealed she switched from Nike to Hoka",
        closing_state="Commitment to try trail running",
        breakthrough=True,
    )
    assert s.breakthrough is True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_models.py::test_conversation_summary -v`
Expected: FAIL

**Step 3: Write the summary model**

Create `src/engram/models/summary.py`:

```python
"""Conversation summary model — narrative arc per message/session.

Inspired by temporal-relationships' SessionArc artifact type.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """High-level narrative summary of a conversation segment."""

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str | None = None

    # Narrative arc (opening -> shift -> closing)
    opening_state: str  # What was the context/state going in
    key_shift: str | None = None  # What changed (if anything)
    closing_state: str  # Where things ended up

    breakthrough: bool = False  # Was there a significant revelation?

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, message_id: str) -> str:
        return f"{tenant_id}:summary:{message_id}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_models.py::test_conversation_summary -v`
Expected: PASS

**Step 5: Create conversation summary prompt**

Create `config/prompts/conversation_summary.jinja2`:

```
You are generating a brief narrative summary of a conversation message.

Message: "{{ message_text }}"
Timestamp: {{ timestamp }}
Speaker: {{ speaker }}

Extracted Entities:
{{ entities | tojson(indent=2) }}

{% if facts %}
Facts extracted:
{{ facts | tojson(indent=2) }}
{% endif %}

{% if relationships %}
Relationships inferred:
{{ relationships | tojson(indent=2) }}
{% endif %}

Generate a 3-part narrative arc:
1. opening_state: What was the context/situation before this message?
2. key_shift: What new information or change occurred? (null if no shift)
3. closing_state: What is the state after this message?
4. breakthrough: Was there a significant revelation or change? (true/false)

Return JSON:
{
  "opening_state": "brief description of prior context",
  "key_shift": "what changed (or null)",
  "closing_state": "brief description of current state",
  "breakthrough": false
}
```

**Step 6: Add summary generation to extraction pipeline**

Add `_generate_summary` method to `ExtractionPipeline`:

```python
async def _generate_summary(
    self,
    request: IngestRequest,
    entities: list[Entity],
    relationships: list[Relationship],
    facts: list[Fact],
    message_id: str,
    run_id: str,
) -> ConversationSummary | None:
    """Generate a narrative summary of the conversation message."""
    entity_dicts = [{"name": e.canonical_name, "type": e.entity_type.value} for e in entities]
    fact_dicts = [{"key": f.fact_key, "text": f.fact_text} for f in facts] if facts else None
    rel_dicts = [
        {"source": r.source_id.split(":")[-1], "type": r.rel_type.value, "target": r.target_id.split(":")[-1]}
        for r in relationships
    ] if relationships else None

    prompt = self._config_service.render_prompt(
        "conversation_summary.jinja2",
        message_text=request.text,
        speaker=request.speaker,
        timestamp=request.timestamp.isoformat(),
        entities=entity_dicts,
        facts=fact_dicts,
        relationships=rel_dicts,
    )

    result = await self._llm.complete_json(prompt)
    return ConversationSummary(
        id=ConversationSummary.build_id(request.tenant_id, message_id),
        tenant_id=request.tenant_id,
        conversation_id=request.conversation_id,
        message_id=message_id,
        extraction_run_id=run_id,
        opening_state=result.get("opening_state", ""),
        key_shift=result.get("key_shift"),
        closing_state=result.get("closing_state", ""),
        breakthrough=result.get("breakthrough", False),
    )
```

Call from `process_message` after fact extraction:

```python
# Step 5: Conversation summary
summary = await self._generate_summary(request, entities, relationships, facts, message_id, run.id)
if summary:
    logger.info("Summary: %s -> %s", summary.opening_state, summary.closing_state)
```

**Step 7: Run all tests**

Run: `python -m pytest tests/unit/ -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/engram/models/summary.py config/prompts/conversation_summary.jinja2 src/engram/services/extraction.py tests/unit/test_models.py
git commit -m "feat: add conversation summary generation (SessionArc pattern)"
```

---

## Task 6: API Endpoints for New Models

**Files:**
- Modify: `src/engram/api/routes.py`
- Test: `tests/integration/test_api.py`

**Step 1: Write API test for facts endpoint**

Add to `tests/integration/test_api.py`:

```python
async def test_get_entity_facts(client, seeded_store):
    """GET /entities/{id}/facts returns facts for an entity."""
    response = await client.get(f"/entities/{entity_id}/facts?tenant_id=t1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_api.py::test_get_entity_facts -v`
Expected: FAIL (404)

**Step 3: Add API endpoints**

Add to `src/engram/api/routes.py`:

```python
@router.get("/entities/{entity_id}/facts")
async def get_entity_facts(
    entity_id: str,
    tenant_id: str,
    fact_key: str | None = None,
    store: GraphStore = Depends(get_store),
):
    """Get facts about an entity."""
    facts = await store.get_facts(tenant_id, entity_id, fact_key=fact_key)
    return [f.model_dump() for f in facts]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_api.py::test_get_entity_facts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engram/api/routes.py tests/integration/test_api.py
git commit -m "feat: add API endpoint for entity facts"
```

---

## Execution Notes

**Dependencies between tasks:**
- Task 1 (Fact model) must complete before Task 2 (Fact extraction) and Task 3 (Richer context)
- Task 4 (Snapshots) depends on Task 2 (needs `facts` list from extraction)
- Task 5 (Summary) depends on Task 2 (needs `facts` for prompt context)
- Task 6 (API) depends on Task 1

**Suggested order:** 1 → 2 → 3 → 4 → 5 → 6

**LLM call budget per message after all tasks:** 5 calls (entity extraction, relationship inference, fact extraction, commitment extraction, summary generation). This is acceptable — temporal-relationships uses 4+ iterations per extraction. Consider making summary generation optional via config if latency is a concern.
