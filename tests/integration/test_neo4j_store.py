"""Integration tests for Neo4j storage.

These tests mirror the MemoryStore unit tests but run against real Neo4j.
Requires: docker-compose up neo4j -d
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType


def _make_entity(name: str = "kendra", entity_type: EntityType = EntityType.PERSON) -> Entity:
    canonical = Entity.normalize_name(name)
    return Entity(
        id=Entity.build_id("t1", entity_type, canonical, group_id="c1"),
        tenant_id="t1",
        conversation_id="c1",
        group_id="c1",
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


@pytest.mark.integration
class TestNeo4jStoreLifecycle:
    async def test_health_check(self, neo4j_store):
        assert await neo4j_store.health_check() is True

    async def test_health_check_returns_false_on_bad_connection(self, neo4j_settings):
        from engram.storage.neo4j import Neo4jStore

        bad_settings = neo4j_settings.model_copy(update={"neo4j_uri": "bolt://localhost:9999"})
        store = Neo4jStore(bad_settings)
        await store.initialize()
        result = await store.health_check()
        assert result is False
        await store.close()


@pytest.mark.integration
class TestNeo4jStoreSchema:
    async def test_indexes_created(self, neo4j_store):
        """Verify that initialize() created the required indexes."""
        result = await neo4j_store._execute_read("SHOW INDEXES YIELD name RETURN name")
        index_names = [r["name"] for r in result]
        assert "entity_id" in index_names
        assert "entity_type" in index_names
        assert "entity_tenant" in index_names
        assert "entity_conversation" in index_names


@pytest.mark.integration
class TestNeo4jStoreEntities:
    async def test_upsert_and_get_entity(self, neo4j_store):
        entity = _make_entity("kendra")
        await neo4j_store.upsert_entity(entity)
        result = await neo4j_store.get_entity(entity.id)
        assert result is not None
        assert result.canonical_name == "kendra"
        assert result.entity_type == EntityType.PERSON
        assert result.tenant_id == "t1"

    async def test_upsert_is_idempotent(self, neo4j_store):
        entity = _make_entity("kendra")
        await neo4j_store.upsert_entity(entity)
        await neo4j_store.upsert_entity(entity)
        entities = await neo4j_store.list_entities("t1", "c1")
        assert len(entities) == 1

    async def test_upsert_merges_aliases(self, neo4j_store):
        entity1 = _make_entity("kendra")
        entity1.aliases = ["Kendra"]
        await neo4j_store.upsert_entity(entity1)

        entity2 = _make_entity("kendra")
        entity2.aliases = ["Kendra M", "Kendra"]
        await neo4j_store.upsert_entity(entity2)

        result = await neo4j_store.get_entity(entity1.id)
        assert result is not None
        assert "Kendra" in result.aliases
        assert "Kendra M" in result.aliases

    async def test_get_entity_not_found(self, neo4j_store):
        result = await neo4j_store.get_entity("nonexistent")
        assert result is None

    async def test_get_entity_by_name(self, neo4j_store):
        entity = _make_entity("kendra")
        await neo4j_store.upsert_entity(entity)
        result = await neo4j_store.get_entity_by_name("t1", "c1", "kendra")
        assert result is not None
        assert result.id == entity.id

    async def test_get_entity_by_name_not_found(self, neo4j_store):
        result = await neo4j_store.get_entity_by_name("t1", "c1", "nonexistent")
        assert result is None

    async def test_list_entities_filtered_by_type(self, neo4j_store):
        await neo4j_store.upsert_entity(_make_entity("kendra", EntityType.PERSON))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        persons = await neo4j_store.list_entities("t1", "c1", entity_type=EntityType.PERSON)
        assert len(persons) == 1
        assert persons[0].canonical_name == "kendra"

    async def test_list_entities_filtered_by_conversation(self, neo4j_store):
        await neo4j_store.upsert_entity(_make_entity("kendra", EntityType.PERSON))
        other = Entity(
            id=Entity.build_id("t1", EntityType.PERSON, "bob", group_id="c2"),
            tenant_id="t1",
            conversation_id="c2",
            entity_type=EntityType.PERSON,
            canonical_name="bob",
        )
        await neo4j_store.upsert_entity(other)

        c1_entities = await neo4j_store.list_entities("t1", "c1")
        assert len(c1_entities) == 1
        assert c1_entities[0].canonical_name == "kendra"

    async def test_list_entities_pagination(self, neo4j_store):
        for i in range(5):
            await neo4j_store.upsert_entity(_make_entity(f"person{i}", EntityType.PERSON))
        page = await neo4j_store.list_entities("t1", "c1", limit=2, offset=0)
        assert len(page) == 2
        page2 = await neo4j_store.list_entities("t1", "c1", limit=2, offset=2)
        assert len(page2) == 2
        page3 = await neo4j_store.list_entities("t1", "c1", limit=2, offset=4)
        assert len(page3) == 1

    async def test_get_recent_entities(self, neo4j_store):
        from datetime import timedelta

        now = datetime.now(UTC)
        old_entity = _make_entity("old_person")
        old_entity.last_mentioned = now - timedelta(days=30)
        await neo4j_store.upsert_entity(old_entity)

        new_entity = _make_entity("new_person")
        new_entity.last_mentioned = now
        await neo4j_store.upsert_entity(new_entity)

        since = now - timedelta(days=7)
        recent = await neo4j_store.get_recent_entities("t1", "c1", since)
        assert len(recent) == 1
        assert recent[0].canonical_name == "new_person"


@pytest.mark.integration
class TestNeo4jStoreRelationships:
    async def test_create_and_get_active(self, neo4j_store):
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await neo4j_store.create_relationship(rel)
        active = await neo4j_store.get_active_relationships(rel.source_id)
        assert len(active) == 1
        assert active[0].rel_type == RelationshipType.PREFERS

    async def test_get_active_filtered_by_type(self, neo4j_store):
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("running", EntityType.TOPIC))

        await neo4j_store.create_relationship(_make_rel("kendra", "nike", RelationshipType.PREFERS))
        discussed = _make_rel("kendra", "running", RelationshipType.DISCUSSED)
        discussed.target_id = Entity.build_id("t1", EntityType.TOPIC, "running", group_id="c1")
        discussed.message_id = "msg-2"
        await neo4j_store.create_relationship(discussed)

        prefers_only = await neo4j_store.get_active_relationships(
            _make_rel().source_id, rel_type=RelationshipType.PREFERS
        )
        assert len(prefers_only) == 1
        assert prefers_only[0].rel_type == RelationshipType.PREFERS

    async def test_terminate_relationship(self, neo4j_store):
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await neo4j_store.create_relationship(rel)
        now = datetime.now(UTC)
        count = await neo4j_store.terminate_relationship(
            source_id=rel.source_id,
            rel_type=RelationshipType.PREFERS,
            tenant_id="t1",
            group_id="c1",
            termination_time=now,
        )
        assert count == 1
        active = await neo4j_store.get_active_relationships(rel.source_id)
        assert len(active) == 0

    async def test_terminate_with_exclude(self, neo4j_store):
        """Terminate all PREFERS except a specific target."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        nike_rel = _make_rel("kendra", "nike")
        await neo4j_store.create_relationship(nike_rel)
        adidas_rel = _make_rel("kendra", "adidas")
        adidas_rel.message_id = "msg-2"
        await neo4j_store.create_relationship(adidas_rel)

        now = datetime.now(UTC)
        count = await neo4j_store.terminate_relationship(
            source_id=nike_rel.source_id,
            rel_type=RelationshipType.PREFERS,
            tenant_id="t1",
            group_id="c1",
            termination_time=now,
            exclude_target_id=adidas_rel.target_id,
        )
        assert count == 1  # Only Nike terminated

        active = await neo4j_store.get_active_relationships(nike_rel.source_id)
        assert len(active) == 1
        assert active[0].target_id == adidas_rel.target_id

    async def test_terminated_not_in_active(self, neo4j_store):
        """Terminated relationships should not appear in get_active_relationships."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))

        rel = _make_rel()
        rel.valid_to = datetime.now(UTC)
        rel.recorded_to = datetime.now(UTC)
        await neo4j_store.create_relationship(rel)

        active = await neo4j_store.get_active_relationships(rel.source_id)
        assert len(active) == 0


@pytest.mark.integration
class TestNeo4jStoreTemporalQueries:
    async def test_world_state_as_of(self, neo4j_store):
        """Nike was valid W1-W3, Adidas from W3 onward."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # Nike: valid W1-W3
        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1
        nike_rel.recorded_to = w3
        await neo4j_store.create_relationship(nike_rel)

        # Adidas: valid W3+
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await neo4j_store.create_relationship(adidas_rel)

        # Query Week 2: Nike was true
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await neo4j_store.query_world_state_as_of("t1", "kendra", w2)
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")

        # Query Week 4: Adidas is true
        w4 = datetime(2024, 4, 15, tzinfo=UTC)
        results = await neo4j_store.query_world_state_as_of("t1", "kendra", w4)
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")

    async def test_world_state_with_rel_type_filter(self, neo4j_store):
        """Temporal query should filter by relationship type."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        rel = _make_rel("kendra", "nike", valid_from=w1)
        await neo4j_store.create_relationship(rel)

        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await neo4j_store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.AVOIDS
        )
        assert len(results) == 0

        results = await neo4j_store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1

    async def test_knowledge_as_of(self, neo4j_store):
        """Test 'what did we KNOW at time X' (record time queries)."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # Nike: recorded W1-W3, but was actually valid from W0 (backdated)
        w0 = datetime(2024, 1, 1, tzinfo=UTC)
        nike_rel = _make_rel("kendra", "nike", valid_from=w0)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1  # We learned about it in W1
        nike_rel.recorded_to = w3  # Retracted in W3
        await neo4j_store.create_relationship(nike_rel)

        # Adidas: recorded from W3 onward
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await neo4j_store.create_relationship(adidas_rel)

        # Query: What did we KNOW in Week 2?
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await neo4j_store.query_knowledge_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")

        # Query: What did we KNOW in Week 4?
        w4 = datetime(2024, 4, 15, tzinfo=UTC)
        results = await neo4j_store.query_knowledge_as_of(
            "t1", "kendra", w4, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")

    async def test_evolution_query(self, neo4j_store):
        """Get all versions of a relationship for timeline view."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        # V1: Nike
        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        nike_rel.recorded_to = w3
        await neo4j_store.create_relationship(nike_rel)

        # V2: Adidas
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.version = 2
        await neo4j_store.create_relationship(adidas_rel)

        results = await neo4j_store.query_evolution(
            "t1", "kendra", rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 2

    async def test_evolution_filtered_by_target(self, neo4j_store):
        """Evolution query can filter by target entity."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await neo4j_store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=UTC)
        w3 = datetime(2024, 3, 20, tzinfo=UTC)

        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        await neo4j_store.create_relationship(nike_rel)

        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        await neo4j_store.create_relationship(adidas_rel)

        # Only Nike history
        results = await neo4j_store.query_evolution("t1", "kendra", target_name="nike")
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")


@pytest.mark.integration
class TestNeo4jStoreGroupId:
    """Tests for group_id scoping (Open Question #3)."""

    async def test_entities_scoped_by_group_id(self, neo4j_store):
        """Entities with different group_ids should be separate."""
        entity_g1 = Entity(
            id=Entity.build_id("t1", EntityType.PERSON, "kendra", group_id="group1"),
            tenant_id="t1",
            conversation_id="c1",
            group_id="group1",
            entity_type=EntityType.PERSON,
            canonical_name="kendra",
        )
        entity_g2 = Entity(
            id=Entity.build_id("t1", EntityType.PERSON, "kendra", group_id="group2"),
            tenant_id="t1",
            conversation_id="c2",
            group_id="group2",
            entity_type=EntityType.PERSON,
            canonical_name="kendra",
        )
        await neo4j_store.upsert_entity(entity_g1)
        await neo4j_store.upsert_entity(entity_g2)

        # Both should exist independently
        result1 = await neo4j_store.get_entity(entity_g1.id)
        result2 = await neo4j_store.get_entity(entity_g2.id)
        assert result1 is not None
        assert result2 is not None
        assert result1.id != result2.id

    async def test_relationships_include_group_id(self, neo4j_store):
        """Relationships should carry group_id for cross-conversation linking."""
        await neo4j_store.upsert_entity(_make_entity("kendra"))
        await neo4j_store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))

        rel = _make_rel()
        rel.group_id = "c1"
        await neo4j_store.create_relationship(rel)

        active = await neo4j_store.get_active_relationships(rel.source_id)
        assert len(active) == 1
        assert active[0].group_id == "c1"
