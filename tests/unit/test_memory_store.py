# tests/unit/test_memory_store.py
"""Tests for in-memory GraphStore implementation.

These tests define the contract that all GraphStore implementations must satisfy.
They cover entity CRUD, relationship lifecycle, and bitemporal queries.
"""

from datetime import UTC, datetime, timedelta

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.storage.memory import MemoryStore


@pytest.fixture
async def store():
    s = MemoryStore()
    await s.initialize()
    yield s
    await s.close()


def _make_entity(name: str = "kendra", entity_type: EntityType = EntityType.PERSON) -> Entity:
    canonical = Entity.normalize_name(name)
    return Entity(
        id=Entity.build_id("t1", entity_type, canonical, group_id="c1"),
        tenant_id="t1",
        conversation_id="c1",
        entity_type=entity_type,
        canonical_name=canonical,
    )


def _make_rel(
    source_name: str = "kendra",
    target_name: str = "nike",
    rel_type: RelationshipType = RelationshipType.PREFERS,
    valid_from: datetime | None = None,
) -> Relationship:
    now = valid_from or datetime.now(UTC)
    return Relationship(
        tenant_id="t1",
        conversation_id="c1",
        group_id="c1",
        message_id="msg-1",
        source_id=Entity.build_id("t1", EntityType.PERSON, source_name, group_id="c1"),
        target_id=Entity.build_id("t1", EntityType.PREFERENCE, target_name, group_id="c1"),
        rel_type=rel_type,
        confidence=0.95,
        evidence="test evidence",
        valid_from=now,
        recorded_from=now,
    )


class TestMemoryStoreLifecycle:
    async def test_health_check(self, store):
        assert await store.health_check() is True

    async def test_initialize_and_close(self):
        s = MemoryStore()
        await s.initialize()
        assert await s.health_check() is True
        await s.close()


class TestMemoryStoreEntities:
    async def test_upsert_creates_entity(self, store):
        entity = _make_entity("kendra")
        result = await store.upsert_entity(entity)
        assert result.id == entity.id

    async def test_upsert_is_idempotent(self, store):
        entity = _make_entity("kendra")
        await store.upsert_entity(entity)
        await store.upsert_entity(entity)
        entities = await store.list_entities("t1", "c1")
        assert len(entities) == 1

    async def test_upsert_merges_aliases(self, store):
        entity1 = _make_entity("kendra")
        entity1.aliases = ["Kendra"]
        await store.upsert_entity(entity1)

        entity2 = _make_entity("kendra")
        entity2.aliases = ["Kendra M", "Kendra"]
        await store.upsert_entity(entity2)

        result = await store.get_entity(entity1.id)
        assert result is not None
        assert "Kendra" in result.aliases
        assert "Kendra M" in result.aliases

    async def test_get_entity_by_id(self, store):
        entity = _make_entity("kendra")
        await store.upsert_entity(entity)
        result = await store.get_entity(entity.id)
        assert result is not None
        assert result.canonical_name == "kendra"

    async def test_get_entity_not_found(self, store):
        result = await store.get_entity("nonexistent")
        assert result is None

    async def test_get_entity_by_name(self, store):
        entity = _make_entity("kendra")
        await store.upsert_entity(entity)
        result = await store.get_entity_by_name("t1", "c1", "kendra")
        assert result is not None
        assert result.id == entity.id

    async def test_get_entity_by_name_not_found(self, store):
        result = await store.get_entity_by_name("t1", "c1", "nonexistent")
        assert result is None

    async def test_list_entities_filtered_by_type(self, store):
        await store.upsert_entity(_make_entity("kendra", EntityType.PERSON))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        persons = await store.list_entities("t1", "c1", entity_type=EntityType.PERSON)
        assert len(persons) == 1
        assert persons[0].canonical_name == "kendra"

    async def test_list_entities_filtered_by_conversation(self, store):
        await store.upsert_entity(_make_entity("kendra", EntityType.PERSON))
        # Create entity in different conversation
        other = Entity(
            id=Entity.build_id("t1", EntityType.PERSON, "bob", group_id="c2"),
            tenant_id="t1",
            conversation_id="c2",
            entity_type=EntityType.PERSON,
            canonical_name="bob",
        )
        await store.upsert_entity(other)

        c1_entities = await store.list_entities("t1", "c1")
        assert len(c1_entities) == 1
        assert c1_entities[0].canonical_name == "kendra"

    async def test_list_entities_pagination(self, store):
        for i in range(5):
            await store.upsert_entity(_make_entity(f"person{i}", EntityType.PERSON))
        page = await store.list_entities("t1", "c1", limit=2, offset=0)
        assert len(page) == 2
        page2 = await store.list_entities("t1", "c1", limit=2, offset=2)
        assert len(page2) == 2
        page3 = await store.list_entities("t1", "c1", limit=2, offset=4)
        assert len(page3) == 1

    async def test_get_recent_entities(self, store):
        now = datetime.now(UTC)
        old_entity = _make_entity("old_person")
        old_entity.last_mentioned = now - timedelta(days=30)
        await store.upsert_entity(old_entity)

        new_entity = _make_entity("new_person")
        new_entity.last_mentioned = now
        await store.upsert_entity(new_entity)

        since = now - timedelta(days=7)
        recent = await store.get_recent_entities("t1", "c1", since)
        assert len(recent) == 1
        assert recent[0].canonical_name == "new_person"


class TestMemoryStoreRelationships:
    async def test_create_and_get_active(self, store):
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await store.create_relationship(rel)
        active = await store.get_active_relationships(rel.source_id)
        assert len(active) == 1
        assert active[0].rel_type == RelationshipType.PREFERS

    async def test_get_active_filtered_by_type(self, store):
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("running", EntityType.TOPIC))

        await store.create_relationship(_make_rel("kendra", "nike", RelationshipType.PREFERS))
        discussed = _make_rel("kendra", "running", RelationshipType.DISCUSSED)
        discussed.target_id = Entity.build_id("t1", EntityType.TOPIC, "running", group_id="c1")
        discussed.message_id = "msg-2"
        await store.create_relationship(discussed)

        prefers_only = await store.get_active_relationships(
            _make_rel().source_id, rel_type=RelationshipType.PREFERS
        )
        assert len(prefers_only) == 1
        assert prefers_only[0].rel_type == RelationshipType.PREFERS

    async def test_terminate_relationship(self, store):
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await store.create_relationship(rel)
        now = datetime.now(UTC)
        count = await store.terminate_relationship(
            source_id=rel.source_id,
            rel_type=RelationshipType.PREFERS,
            tenant_id="t1",
            group_id="c1",
            termination_time=now,
        )
        assert count == 1
        active = await store.get_active_relationships(rel.source_id)
        assert len(active) == 0

    async def test_terminate_with_exclude(self, store):
        """Terminate all PREFERS except a specific target."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        nike_rel = _make_rel("kendra", "nike")
        await store.create_relationship(nike_rel)
        adidas_rel = _make_rel("kendra", "adidas")
        adidas_rel.message_id = "msg-2"
        await store.create_relationship(adidas_rel)

        now = datetime.now(UTC)
        count = await store.terminate_relationship(
            source_id=nike_rel.source_id,
            rel_type=RelationshipType.PREFERS,
            tenant_id="t1",
            group_id="c1",
            termination_time=now,
            exclude_target_id=adidas_rel.target_id,
        )
        assert count == 1  # Only Nike terminated

        active = await store.get_active_relationships(nike_rel.source_id)
        assert len(active) == 1
        assert active[0].target_id == adidas_rel.target_id

    async def test_terminated_not_in_active(self, store):
        """Terminated relationships should not appear in get_active_relationships."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))

        rel = _make_rel()
        rel.valid_to = datetime.now(UTC)
        rel.recorded_to = datetime.now(UTC)
        await store.create_relationship(rel)

        active = await store.get_active_relationships(rel.source_id)
        assert len(active) == 0


class TestMemoryStoreTemporalQueries:
    async def test_world_state_as_of(self, store):
        """Nike was valid W1-W3, Adidas from W3 onward."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # Nike: valid W1-W3
        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1
        nike_rel.recorded_to = w3
        await store.create_relationship(nike_rel)

        # Adidas: valid W3+
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await store.create_relationship(adidas_rel)

        # Query Week 2: Nike was true
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await store.query_world_state_as_of("t1", "kendra", w2)
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")

        # Query Week 4: Adidas is true
        w4 = datetime(2024, 4, 15, tzinfo=UTC)
        results = await store.query_world_state_as_of("t1", "kendra", w4)
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")

    async def test_world_state_with_rel_type_filter(self, store):
        """Temporal query should filter by relationship type."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        rel = _make_rel("kendra", "nike", valid_from=w1)
        await store.create_relationship(rel)

        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.AVOIDS
        )
        assert len(results) == 0

        results = await store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1

    async def test_knowledge_as_of(self, store):
        """Test 'what did we KNOW at time X' (record time queries)."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # Nike: recorded W1-W3, but was actually valid from W0 (backdated)
        w0 = datetime(2024, 1, 1, tzinfo=UTC)
        nike_rel = _make_rel("kendra", "nike", valid_from=w0)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1  # We learned about it in W1
        nike_rel.recorded_to = w3  # Retracted in W3
        await store.create_relationship(nike_rel)

        # Adidas: recorded from W3 onward
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await store.create_relationship(adidas_rel)

        # Query: What did we KNOW in Week 2?
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await store.query_knowledge_as_of("t1", "kendra", w2, rel_type="prefers")
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")  # We believed Nike

        # Query: What did we KNOW in Week 4?
        w4 = datetime(2024, 4, 15, tzinfo=UTC)
        results = await store.query_knowledge_as_of("t1", "kendra", w4, rel_type="prefers")
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")  # We believe Adidas

    async def test_evolution_query(self, store):
        """Get all versions of a relationship for timeline view."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # V1: Nike
        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        nike_rel.recorded_to = w3
        await store.create_relationship(nike_rel)

        # V2: Adidas
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.version = 2
        await store.create_relationship(adidas_rel)

        results = await store.query_evolution("t1", "kendra", rel_type=RelationshipType.PREFERS)
        assert len(results) == 2

    async def test_evolution_filtered_by_target(self, store):
        """Evolution query can filter by target entity."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        await store.create_relationship(nike_rel)

        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        await store.create_relationship(adidas_rel)

        # Only Nike history
        results = await store.query_evolution("t1", "kendra", target_name="nike")
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")
