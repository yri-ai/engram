# tests/unit/test_models.py
from datetime import UTC, datetime

from engram.models.entity import Entity, EntityType
from engram.models.message import IngestRequest
from engram.models.relationship import Relationship, RelationshipType


def test_entity_creation():
    entity = Entity(
        id="t1:c1:PERSON:kendra",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="kendra",
        aliases=["Kendra", "Kendra M"],
    )
    assert entity.id == "t1:c1:PERSON:kendra"
    assert entity.entity_type == EntityType.PERSON


def test_entity_id_generation_with_group():
    entity_id = Entity.build_id("t1", EntityType.PERSON, "kendra", group_id="c1")
    assert entity_id == "t1:c1:PERSON:kendra"


def test_entity_id_generation_without_group():
    entity_id = Entity.build_id("t1", EntityType.PERSON, "kendra")
    assert entity_id == "t1:conversation:PERSON:kendra"


def test_relationship_creation():
    now = datetime.now(UTC)
    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS,
        confidence=0.95,
        evidence="I love Nike shoes",
        valid_from=now,
        valid_to=None,
        recorded_from=now,
        recorded_to=None,
        version=1,
    )
    assert rel.is_active is True
    assert rel.is_currently_believed is True


def test_relationship_terminated():
    now = datetime.now(UTC)
    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS,
        confidence=0.95,
        evidence="I love Nike shoes",
        valid_from=now,
        valid_to=now,  # Terminated
        recorded_from=now,
        recorded_to=now,  # No longer believed
        version=1,
    )
    assert rel.is_active is False
    assert rel.is_currently_believed is False


def test_ingest_request_validation():
    req = IngestRequest(
        text="Kendra loves Nike running shoes",
        speaker="Kendra",
        timestamp=datetime.now(UTC),
        conversation_id="c1",
    )
    assert req.text == "Kendra loves Nike running shoes"
    assert req.tenant_id == "default"  # Default tenant for MVP
