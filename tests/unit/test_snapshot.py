"""Tests for conversation state snapshots and delta tracking."""

import pytest
from datetime import UTC, datetime

from engram.models.snapshot import ConversationSnapshot, SnapshotDelta, ChangeType
from engram.models.entity import Entity, EntityType
from engram.services.snapshot import SnapshotService
from engram.storage.memory import MemoryStore


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
    assert snap.relationship_count == 2
    assert snap.fact_count == 1
    assert len(snap.entities) == 3


def test_delta_creation():
    delta = SnapshotDelta(
        change_type=ChangeType.ADDED,
        artifact_type="entity",
        artifact_id="t1:c1:PERSON:alice",
        summary="New entity: alice (PERSON)",
    )
    assert delta.change_type == ChangeType.ADDED
    assert delta.artifact_type == "entity"
    assert delta.artifact_id == "t1:c1:PERSON:alice"


def test_change_type_values():
    assert ChangeType.ADDED == "added"
    assert ChangeType.UPDATED == "updated"
    assert ChangeType.SUPERSEDED == "superseded"


def test_snapshot_defaults():
    snap = ConversationSnapshot(
        id="snap-1",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        extraction_run_id="run1",
    )
    assert snap.entity_count == 0
    assert snap.relationship_count == 0
    assert snap.fact_count == 0
    assert snap.entities == []
    assert snap.deltas == []
    assert snap.metadata == {}
    assert snap.created_at is not None


@pytest.fixture
def memory_store():
    return MemoryStore()


@pytest.fixture
def snapshot_service(memory_store):
    return SnapshotService(memory_store)


@pytest.mark.asyncio
async def test_build_snapshot(snapshot_service, memory_store):
    e = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    await memory_store.upsert_entity(e)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
    )
    assert snap.entity_count == 1
    assert "alice" in snap.entities
    assert len(snap.deltas) == 0


@pytest.mark.asyncio
async def test_build_snapshot_with_entity_deltas(snapshot_service, memory_store):
    e = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    await memory_store.upsert_entity(e)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
        new_entities=[e],
    )
    assert len(snap.deltas) == 1
    assert snap.deltas[0].change_type == ChangeType.ADDED
    assert snap.deltas[0].artifact_type == "entity"
    assert "alice" in snap.deltas[0].summary
    assert "PERSON" in snap.deltas[0].summary


@pytest.mark.asyncio
async def test_build_snapshot_with_relationship_deltas(snapshot_service, memory_store):
    from engram.models.relationship import Relationship, RelationshipType

    e1 = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    e2 = Entity(
        id="t1:c1:PERSON:bob",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="bob",
    )
    await memory_store.upsert_entity(e1)
    await memory_store.upsert_entity(e2)

    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id=e1.id,
        target_id=e2.id,
        rel_type=RelationshipType.KNOWS,
        confidence=0.8,
        valid_from=datetime.now(UTC),
    )
    await memory_store.create_relationship(rel)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
        new_relationships=[rel],
    )
    assert snap.relationship_count == 1
    assert len(snap.deltas) == 1
    assert snap.deltas[0].artifact_type == "relationship"
    assert "knows" in snap.deltas[0].summary


@pytest.mark.asyncio
async def test_build_snapshot_with_fact_deltas(snapshot_service, memory_store):
    from engram.models.fact import Fact

    e = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    await memory_store.upsert_entity(e)

    fact = Fact(
        id="t1:fact:msg1:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        entity_id=e.id,
        fact_key="age",
        fact_text="Alice is 32 years old",
        confidence=0.9,
    )
    await memory_store.save_fact(fact)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
        new_facts=[fact],
    )
    assert snap.fact_count == 1
    assert len(snap.deltas) == 1
    assert snap.deltas[0].change_type == ChangeType.ADDED
    assert snap.deltas[0].artifact_type == "fact"
    assert "age" in snap.deltas[0].summary


@pytest.mark.asyncio
async def test_build_snapshot_superseded_fact_delta(snapshot_service, memory_store):
    from engram.models.fact import Fact

    e = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    await memory_store.upsert_entity(e)

    fact = Fact(
        id="t1:fact:msg2:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg2",
        entity_id=e.id,
        fact_key="age",
        fact_text="Alice is 33 years old",
        confidence=0.9,
        supersedes_fact_id="t1:fact:msg1:0",
    )

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg2",
        run_id="run2",
        new_facts=[fact],
    )
    assert len(snap.deltas) == 1
    assert snap.deltas[0].change_type == ChangeType.SUPERSEDED


@pytest.mark.asyncio
async def test_build_snapshot_multiple_deltas(snapshot_service, memory_store):
    from engram.models.fact import Fact
    from engram.models.relationship import Relationship, RelationshipType

    e1 = Entity(
        id="t1:c1:PERSON:alice",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="alice",
    )
    e2 = Entity(
        id="t1:c1:CONCEPT:running",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.CONCEPT,
        canonical_name="running",
    )
    await memory_store.upsert_entity(e1)
    await memory_store.upsert_entity(e2)

    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id=e1.id,
        target_id=e2.id,
        rel_type=RelationshipType.PREFERS,
        confidence=0.8,
        valid_from=datetime.now(UTC),
    )
    await memory_store.create_relationship(rel)

    fact = Fact(
        id="t1:fact:msg1:0",
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        entity_id=e1.id,
        fact_key="hobby",
        fact_text="Alice enjoys running",
        confidence=0.8,
    )
    await memory_store.save_fact(fact)

    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
        new_entities=[e1, e2],
        new_relationships=[rel],
        new_facts=[fact],
    )
    assert snap.entity_count == 2
    assert snap.relationship_count == 1
    assert snap.fact_count == 1
    assert len(snap.deltas) == 4  # 2 entities + 1 relationship + 1 fact
    artifact_types = [d.artifact_type for d in snap.deltas]
    assert artifact_types.count("entity") == 2
    assert artifact_types.count("relationship") == 1
    assert artifact_types.count("fact") == 1


@pytest.mark.asyncio
async def test_snapshot_counts_reflect_existing_store_state(snapshot_service, memory_store):
    """Counts should reflect total store state, not just new items."""
    from engram.models.fact import Fact
    from engram.models.relationship import Relationship, RelationshipType

    e1 = Entity(
        id="t1:c1:PERSON:alice", tenant_id="t1", conversation_id="c1",
        entity_type=EntityType.PERSON, canonical_name="alice",
    )
    e2 = Entity(
        id="t1:c1:PERSON:bob", tenant_id="t1", conversation_id="c1",
        entity_type=EntityType.PERSON, canonical_name="bob",
    )
    await memory_store.upsert_entity(e1)
    await memory_store.upsert_entity(e2)

    # Save a relationship and fact from a "previous" message
    rel = Relationship(
        tenant_id="t1", conversation_id="c1", message_id="old-msg",
        source_id=e1.id, target_id=e2.id,
        rel_type=RelationshipType.KNOWS, confidence=0.8,
        valid_from=datetime.now(UTC),
    )
    await memory_store.create_relationship(rel)

    fact = Fact(
        id="t1:fact:old:0", tenant_id="t1", conversation_id="c1",
        message_id="old-msg", entity_id=e1.id,
        fact_key="age", fact_text="Alice is 32", confidence=0.9,
    )
    await memory_store.save_fact(fact)

    # Build snapshot for a NEW message with no new artifacts
    snap = await snapshot_service.build_snapshot(
        tenant_id="t1", conversation_id="c1",
        message_id="new-msg", run_id="run2",
    )
    # Counts should reflect existing store state, not zero
    assert snap.entity_count == 2
    assert snap.relationship_count == 1
    assert snap.fact_count == 1
    assert len(snap.deltas) == 0  # No new artifacts


@pytest.mark.asyncio
async def test_build_snapshot_empty_conversation(snapshot_service):
    snap = await snapshot_service.build_snapshot(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        run_id="run1",
    )
    assert snap.entity_count == 0
    assert snap.entities == []
    assert snap.deltas == []
    assert snap.id.startswith("snap-")
