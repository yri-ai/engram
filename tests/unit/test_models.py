# tests/unit/test_models.py
from datetime import UTC, datetime

from engram.models.entity import Entity, EntityType
from engram.models.fact import Fact, FactStatus
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
