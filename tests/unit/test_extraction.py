"""Tests for the 6-stage extraction pipeline.

Covers:
- Basic extraction (entity + relationship creation)
- Idempotency (duplicate message returns 0)
- Correction detection (retraction via conflict resolution)
- Open Question #2: Unknown relationship types fall back to relates_to
- Open Question #3: group_id for cross-conversation entity linking
- Open Question #4: Confidence snapping to discrete levels
- Fact extraction (standalone knowledge claims about entities)
- Edge cases: empty extraction, unresolvable mentions
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from engram.models.commitment import CommitmentStatus
from engram.models.message import IngestRequest
from engram.models.relationship import RelationshipType
from engram.models.run import RunStatus
from engram.services.dedup import InMemoryDedup
from engram.services.extraction import ExtractionPipeline, snap_confidence
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore


@pytest.fixture
async def pipeline():
    """Pipeline with mocked LLM, in-memory store, and real dedup/resolver."""
    store = MemoryStore()
    await store.initialize()
    dedup = InMemoryDedup()
    resolver = ConflictResolver(store)
    llm = AsyncMock()
    return ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=llm)


# ── snap_confidence (Open Question #4) ──────────────────────────────────────


class TestSnapConfidence:
    """Confidence calibration: snap raw LLM values to discrete semantic levels."""

    def test_exact_levels_unchanged(self):
        assert snap_confidence(1.0) == 1.0
        assert snap_confidence(0.8) == 0.8
        assert snap_confidence(0.6) == 0.6
        assert snap_confidence(0.4) == 0.4

    def test_snaps_to_nearest(self):
        assert snap_confidence(0.95) == 1.0  # close to 1.0
        assert snap_confidence(0.85) == 0.8  # close to 0.8
        assert snap_confidence(0.73) == 0.8  # |0.73-0.8|=0.07 < |0.73-0.6|=0.13
        assert snap_confidence(0.5) == 0.6  # |0.5-0.6|=0.1 ties with 0.4; min picks 0.6 (first)
        assert snap_confidence(0.3) == 0.4  # close to 0.4
        assert snap_confidence(0.15) == 0.4  # closest to 0.4

    def test_zero_snaps_to_lowest(self):
        assert snap_confidence(0.0) == 0.4

    def test_custom_levels(self):
        assert snap_confidence(0.9, levels=(1.0, 0.5)) == 1.0
        assert snap_confidence(0.3, levels=(1.0, 0.5)) == 0.5


# ── Full Pipeline ────────────────────────────────────────────────────────────


async def test_pipeline_extracts_entities_and_relationships(pipeline):
    """Full pipeline: message -> entities + relationships in graph."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            # Stage 1: Entity extraction
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Nike shoes",
                        "canonical": "nike-shoes",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            # Stage 2: Relationship inference
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Nike shoes",
                        "type": "prefers",
                        "confidence": 0.95,
                        "evidence": "loves Nike shoes",
                        "temporal_marker": "now",
                    },
                ]
            },
            # Stage 3.5: Fact extraction
            {"facts": []},
            # Stage 4: Commitment extraction
            {"commitments": []},
            # Stage 5: Conversation summary
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra loves Nike shoes",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)

    assert response.entities_extracted == 2
    assert response.relationships_inferred == 1

    # Verify entities in store
    entities = await pipeline._store.list_entities("t1", "c1")
    assert len(entities) == 2

    # Verify relationships via entity ID
    kendra_id = "t1:c1:PERSON:kendra"
    rels = await pipeline._store.get_active_relationships(kendra_id)
    assert len(rels) == 1
    assert rels[0].rel_type == RelationshipType.PREFERS
    assert rels[0].extraction_run_id is not None


async def test_pipeline_creates_extraction_run(pipeline):
    """Pipeline should create and persist an ExtractionRun."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {"entities": [{"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0}]},
            {"relationships": []},
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra is here",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
    )
    await pipeline.process_message(request)

    # Verify run exists in store
    runs = list(pipeline._store._runs.values())
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETED
    assert runs[0].prompt_id is not None


async def test_pipeline_extracts_commitments(pipeline):
    """Pipeline should extract commitments and save them to store."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            # Entities
            {"entities": [{"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0}]},
            # Relationships
            {"relationships": []},
            # Facts
            {"facts": []},
            # Commitments
            {
                "commitments": [
                    {
                        "entity_mention": "Kendra",
                        "text": "I will finish the report by Friday",
                        "target_date": "2026-03-27T17:00:00Z",
                        "confidence": 1.0,
                    }
                ]
            },
            # Summary
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="I will finish the report by Friday",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
    )
    await pipeline.process_message(request)

    # Verify commitment in store
    kendra_id = "t1:c1:PERSON:kendra"
    commitments = await pipeline._store.get_commitments("t1", kendra_id)
    assert len(commitments) == 1
    assert commitments[0].text == "I will finish the report by Friday"
    assert commitments[0].status == CommitmentStatus.ACTIVE


async def test_pipeline_deduplicates_messages(pipeline):
    """Processing same message twice should be idempotent."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    }
                ]
            },
            {"relationships": []},
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra is here",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
        message_id="msg-fixed",
    )
    await pipeline.process_message(request)
    response2 = await pipeline.process_message(request)

    # Second call is a dedup no-op
    assert response2.entities_extracted == 0
    assert response2.relationships_inferred == 0


async def test_correction_detection(pipeline):
    """Corrections retract old beliefs via conflict resolution exclusivity."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            # Message 1: Kendra likes Nike
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Nike",
                        "canonical": "nike",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Nike",
                        "type": "prefers",
                        "confidence": 0.9,
                        "evidence": "likes Nike",
                        "temporal_marker": "now",
                    },
                ]
            },
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
            # Message 2: Actually Adidas
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Adidas",
                        "canonical": "adidas",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Adidas",
                        "type": "prefers",
                        "confidence": 0.95,
                        "evidence": "correction to Adidas",
                        "temporal_marker": "now",
                    },
                ]
            },
            {"facts": []},
            {"commitments": []},
            {"opening_state": "correction", "key_shift": "switched to Adidas", "closing_state": "prefers Adidas", "breakthrough": True},
        ]
    )

    msg1 = IngestRequest(
        text="Kendra likes Nike",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
        message_id="msg-1",
    )
    await pipeline.process_message(msg1)

    msg2 = IngestRequest(
        text="Actually, Kendra prefers Adidas",
        speaker="System",
        conversation_id="c1",
        tenant_id="t1",
        message_id="msg-2",
    )
    await pipeline.process_message(msg2)

    # Only Adidas should be active (Nike terminated by exclusivity)
    kendra_id = "t1:c1:PERSON:kendra"
    active_prefs = await pipeline._store.get_active_relationships(
        kendra_id, rel_type=RelationshipType.PREFERS
    )
    assert len(active_prefs) == 1
    assert "adidas" in active_prefs[0].target_id

    # Both Nike (terminated) and Adidas (active) visible in evolution
    all_prefs = await pipeline._store.query_evolution(
        "t1", "kendra", rel_type=RelationshipType.PREFERS
    )
    assert len(all_prefs) >= 2


# ── Open Question #2: Relationship Type Validation ──────────────────────────


async def test_unknown_relationship_type_falls_back_to_relates_to(pipeline):
    """Invalid relationship types fall back to relates_to with original preserved."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Running",
                        "canonical": "running",
                        "type": "TOPIC",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Running",
                        "type": "slightly_enjoys",  # Not a valid RelationshipType
                        "confidence": 0.8,
                        "evidence": "seems to like running",
                        "temporal_marker": "now",
                    },
                ]
            },
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra seems to enjoy running",
        speaker="observer",
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)

    assert response.relationships_inferred == 1

    kendra_id = "t1:c1:PERSON:kendra"
    rels = await pipeline._store.get_active_relationships(kendra_id)
    assert len(rels) == 1
    assert rels[0].rel_type == RelationshipType.RELATES_TO
    assert rels[0].metadata.get("original_type") == "slightly_enjoys"


# ── Open Question #3: Cross-Conversation Entity Resolution ──────────────────


async def test_group_id_used_for_entity_ids(pipeline):
    """Explicit group_id creates entity IDs scoped to the group."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                ]
            },
            {"relationships": []},
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra is here",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
        group_id="shared-group",
    )
    response = await pipeline.process_message(request)
    assert response.entities_extracted == 1

    # Entity ID uses group_id, not conversation_id
    entity = await pipeline._store.get_entity("t1:shared-group:PERSON:kendra")
    assert entity is not None
    assert entity.group_id == "shared-group"


async def test_group_id_defaults_to_conversation_id(pipeline):
    """Without group_id, entities are conversation-scoped."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                ]
            },
            {"relationships": []},
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra is here",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
        # No group_id
    )
    await pipeline.process_message(request)

    # Entity ID uses conversation_id as default group
    entity = await pipeline._store.get_entity("t1:c1:PERSON:kendra")
    assert entity is not None


# ── Open Question #4: Confidence Snapping in Pipeline ────────────────────────


async def test_confidence_is_snapped_in_pipeline(pipeline):
    """Relationship confidence is snapped to discrete levels during inference."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Coffee",
                        "canonical": "coffee",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Coffee",
                        "type": "prefers",
                        "confidence": 0.73,  # Should snap to 0.8
                        "evidence": "seems to like coffee",
                        "temporal_marker": "now",
                    },
                ]
            },
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra seems to like coffee",
        speaker="observer",
        conversation_id="c1",
        tenant_id="t1",
    )
    await pipeline.process_message(request)

    kendra_id = "t1:c1:PERSON:kendra"
    rels = await pipeline._store.get_active_relationships(kendra_id)
    assert len(rels) == 1
    assert rels[0].confidence == 0.8  # Snapped from 0.73


# ── Fact Extraction ──────────────────────────────────────────────────────────


async def test_fact_extraction(pipeline):
    """Test that facts are extracted from messages."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            # Stage 1: Entity extraction
            {"entities": [{"name": "Alice", "canonical": "alice", "type": "PERSON", "confidence": 1.0}]},
            # Stage 2: Relationship inference
            {"relationships": []},
            # Stage 3.5: Fact extraction
            {"facts": [
                {"entity_mention": "Alice", "fact_key": "age", "fact_text": "Alice is 32 years old", "confidence": 0.9},
            ]},
            # Stage 4: Commitment extraction
            {"commitments": []},
            # Stage 5: Conversation summary
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Alice mentioned she is 32 years old",
        speaker="Alice",
        timestamp=datetime.now(UTC),
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)
    assert response.entities_extracted == 1

    facts = await pipeline._store.get_facts("t1", "t1:c1:PERSON:alice")
    assert len(facts) == 1
    assert facts[0].fact_key == "age"
    assert facts[0].fact_text == "Alice is 32 years old"


# ── Edge Cases ───────────────────────────────────────────────────────────────


async def test_empty_extraction_returns_zero(pipeline):
    """LLM returns no entities — skip relationship inference entirely."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {"entities": []},
            # No second item — pipeline must not call LLM again
        ]
    )

    request = IngestRequest(
        text="Hi there",
        speaker="User",
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)

    assert response.entities_extracted == 0
    assert response.relationships_inferred == 0
    # Verify LLM was called exactly once (entity extraction only)
    assert pipeline._llm.complete_json.call_count == 1


async def test_unresolvable_mention_skipped(pipeline):
    """Relationship mentions that don't match any entity are skipped."""
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {
                "entities": [
                    {
                        "name": "Kendra",
                        "canonical": "kendra",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Unknown Entity",
                        "type": "prefers",
                        "confidence": 0.9,
                        "evidence": "test",
                        "temporal_marker": "now",
                    },
                ]
            },
            {"facts": []},
            {"commitments": []},
            {"opening_state": "context", "key_shift": None, "closing_state": "state", "breakthrough": False},
        ]
    )

    request = IngestRequest(
        text="Kendra mentions something",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)

    assert response.entities_extracted == 1
    assert response.relationships_inferred == 0  # Skipped: target not found


# ── Richer LLM Context ──────────────────────────────────────────────────────


async def test_entity_extraction_receives_prior_knowledge_context(pipeline):
    """LLM should receive prior facts and relationships as context, not just entity names."""
    from engram.models.entity import Entity, EntityType
    from engram.models.fact import Fact

    # Pre-populate store with existing entity and fact
    entity = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
        last_mentioned=datetime.now(UTC),
    )
    await pipeline._store.upsert_entity(entity)

    fact = Fact(
        id="t1:fact:old:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="old",
        entity_id="t1:c1:PERSON:alice",
        fact_key="age",
        fact_text="Alice is 32",
        confidence=0.9,
    )
    await pipeline._store.save_fact(fact)

    # Mock LLM — no entities extracted (we just want to verify prompt content)
    pipeline._llm.complete_json = AsyncMock(
        side_effect=[
            {"entities": []},  # entity extraction
        ]
    )

    request = IngestRequest(
        text="Alice said something",
        speaker="Bob",
        timestamp=datetime.now(UTC),
        conversation_id="c1",
        tenant_id="t1",
    )
    await pipeline.process_message(request)

    # Verify the prompt included prior facts context
    call_args = pipeline._llm.complete_json.call_args_list[0]
    prompt_text = call_args[0][0]
    assert "alice" in prompt_text.lower()
    assert "32" in prompt_text  # Prior fact should appear in context
