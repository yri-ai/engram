"""Tests for conflict resolution service."""

from datetime import UTC, datetime

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore


@pytest.fixture
async def resolver():
    store = MemoryStore()
    await store.initialize()
    return ConflictResolver(store)


@pytest.fixture
async def store_with_entities():
    store = MemoryStore()
    await store.initialize()
    for name, etype in [
        ("kendra", EntityType.PERSON),
        ("nike", EntityType.PREFERENCE),
        ("adidas", EntityType.PREFERENCE),
    ]:
        canonical = Entity.normalize_name(name)
        await store.upsert_entity(
            Entity(
                id=Entity.build_id("t1", etype, canonical),
                tenant_id="t1",
                conversation_id="c1",
                entity_type=etype,
                canonical_name=canonical,
            )
        )
    return store


async def test_prefers_enforces_single_active(store_with_entities):
    """When new "prefers" created, old one should be terminated."""
    store = store_with_entities
    resolver = ConflictResolver(store)
    now = datetime.now(UTC)

    # Create initial preference: Kendra -> Nike
    nike_rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        group_id="c1",
        message_id="msg-1",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS,
        confidence=0.95,
        evidence="I love Nike",
        valid_from=now,
        recorded_from=now,
    )
    await resolver.resolve_and_create(nike_rel)

    # Create conflicting preference: Kendra -> Adidas
    adidas_rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        group_id="c1",
        message_id="msg-2",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:adidas",
        rel_type=RelationshipType.PREFERS,
        confidence=0.9,
        evidence="Switched to Adidas",
        valid_from=now,
        recorded_from=now,
    )
    await resolver.resolve_and_create(adidas_rel)

    # Verify: only Adidas is active
    active = await store.get_active_relationships("t1:c1:PERSON:kendra")
    assert len(active) == 1
    assert active[0].target_id.endswith("adidas")


async def test_knows_allows_multiple_active(store_with_entities):
    """'knows' relationships should allow multiple active."""
    store = store_with_entities
    # Add second person
    await store.upsert_entity(
        Entity(
            id="t1:c1:PERSON:sarah",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="sarah",
        )
    )
    resolver = ConflictResolver(store)
    now = datetime.now(UTC)

    for target, msg_id in [("nike", "msg-1"), ("sarah", "msg-2")]:
        await resolver.resolve_and_create(
            Relationship(
                tenant_id="t1",
                conversation_id="c1",
                message_id=msg_id,
                source_id="t1:c1:PERSON:kendra",
                target_id=f"t1:c1:PERSON:{target}"
                if target == "sarah"
                else f"t1:c1:PREFERENCE:{target}",
                rel_type=RelationshipType.KNOWS,
                confidence=0.9,
                evidence="test",
                valid_from=now,
                recorded_from=now,
            )
        )

    active = await store.get_active_relationships(
        "t1:c1:PERSON:kendra", rel_type=RelationshipType.KNOWS
    )
    assert len(active) == 2
