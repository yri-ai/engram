from datetime import UTC, datetime

from engram.models.entity import Entity, EntityType
from engram.models.fact import Fact
from engram.models.forecasting import (
    ForecastQuestion,
    ForecastQuestionType,
    OutcomeBranch,
    ResolutionCriteria,
)
from engram.models.relationship import Relationship, RelationshipType
from engram.services.as_of_evidence import AsOfEvidenceCompiler, MemoryGraphEvidenceAdapter
from engram.storage.memory import MemoryStore

AS_OF = datetime(2026, 1, 15, tzinfo=UTC)
BEFORE = datetime(2026, 1, 1, tzinfo=UTC)
AFTER = datetime(2026, 2, 1, tzinfo=UTC)


async def test_compiler_includes_direct_relationship_and_one_hop_facts_known_as_of():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    acme = _entity("acme", EntityType.CONCEPT)
    await store.upsert_entity(alice)
    await store.upsert_entity(acme)
    await store.save_fact(_fact("f-direct", alice.id, "Alice is evaluating renewal."))
    await store.create_relationship(
        _relationship("r-msg", alice.id, acme.id, "Alice is linked to Acme.")
    )
    await store.save_fact(_fact("f-hop", acme.id, "Acme offered Alice a discount."))

    dossier = await _compile(store, target_id=alice.id)

    assert [item.id for item in dossier.evidence_items] == [
        "fact:f-direct",
        "fact:f-hop",
        f"relationship:r-msg:{alice.id}:relates_to:{acme.id}:1",
    ]
    assert dossier.excluded_counts == {}
    assert dossier.evidence_items[0].supersession_status == "current_as_of"


async def test_adapter_includes_inbound_relationships_and_related_source_facts():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    project = _entity("project", EntityType.TOPIC)
    await store.upsert_entity(alice)
    await store.upsert_entity(project)
    await store.create_relationship(
        _relationship("r-in", project.id, alice.id, "Project depends on Alice.")
    )
    await store.save_fact(_fact("f-source", project.id, "Project has a January deadline."))

    dossier = await _compile(store, target_id=alice.id)

    assert [item.id for item in dossier.evidence_items] == [
        "fact:f-source",
        f"relationship:r-in:{project.id}:relates_to:{alice.id}:1",
    ]


async def test_future_recorded_fact_is_excluded_even_if_valid_before_as_of():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    await store.upsert_entity(alice)
    await store.save_fact(
        _fact(
            "f-future-recorded",
            alice.id,
            "Alice renewed, but this was recorded later.",
            valid_from=BEFORE,
            recorded_from=AFTER,
        )
    )

    dossier = await _compile(store, target_id=alice.id)

    assert dossier.evidence_items == []
    assert dossier.excluded_counts == {"future_record_time": 1}


async def test_source_ingested_at_after_as_of_is_excluded_separately():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    await store.upsert_entity(alice)
    await store.save_fact(
        _fact(
            "f-late-source",
            alice.id,
            "Late-ingested source.",
            metadata={"source_ingested_at": AFTER},
        )
    )

    dossier = await _compile(store, target_id=alice.id)

    assert dossier.evidence_items == []
    assert dossier.excluded_counts == {"source_ingested_after_as_of": 1}


async def test_resolution_evidence_markers_are_excluded():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    await store.upsert_entity(alice)
    await store.save_fact(
        _fact(
            "f-role",
            alice.id,
            "Resolution-only evidence.",
            metadata={"evidence_role": "resolution_evidence"},
        )
    )
    await store.save_fact(
        _fact(
            "f-question",
            alice.id,
            "Question-specific resolution evidence.",
            metadata={"resolution_for_question_id": "q-1"},
        )
    )

    dossier = await _compile(store, target_id=alice.id)

    assert dossier.evidence_items == []
    assert dossier.excluded_counts == {"resolution_evidence": 2}


async def test_post_hoc_derived_evidence_is_excluded():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    await store.upsert_entity(alice)
    await store.save_fact(
        _fact("f-derived", alice.id, "Derived after forecast.", metadata={"derived_after": AFTER})
    )

    dossier = await _compile(store, target_id=alice.id)

    assert dossier.evidence_items == []
    assert dossier.excluded_counts == {"post_hoc_derived": 1}


async def test_supersession_status_depends_on_replacement_record_time():
    store = MemoryStore()
    await store.initialize()
    alice = _entity("alice", EntityType.PERSON)
    await store.upsert_entity(alice)
    await store.save_fact(_fact("f-old", alice.id, "Alice prefers the basic plan."))
    await store.save_fact(
        _fact(
            "f-replacement-known",
            alice.id,
            "Alice prefers the pro plan.",
            recorded_from=BEFORE,
            metadata={"source_ingested_at": BEFORE},
            supersedes_fact_id="f-old",
        )
    )
    await store.save_fact(_fact("f-later-old", alice.id, "Alice may downgrade."))
    await store.save_fact(
        _fact(
            "f-replacement-later",
            alice.id,
            "Alice will not downgrade.",
            recorded_from=AFTER,
            supersedes_fact_id="f-later-old",
        )
    )

    dossier = await _compile(store, target_id=alice.id)
    items = {item.id: item for item in dossier.evidence_items}

    assert items["fact:f-old"].supersession_status == "superseded_before_as_of"
    assert items["fact:f-old"].superseded_by_id == "fact:f-replacement-known"
    assert items["fact:f-later-old"].supersession_status == "current_as_of_later_superseded"
    assert items["fact:f-later-old"].superseded_by_id == "fact:f-replacement-later"
    assert "fact:f-replacement-later" not in items


def _entity(name: str, entity_type: EntityType) -> Entity:
    canonical = Entity.normalize_name(name)
    return Entity(
        id=Entity.build_id("t1", entity_type, canonical, group_id="g1"),
        tenant_id="t1",
        conversation_id="c1",
        group_id="g1",
        entity_type=entity_type,
        canonical_name=canonical,
    )


def _fact(
    fact_id: str,
    entity_id: str,
    text: str,
    *,
    valid_from: datetime = BEFORE,
    recorded_from: datetime = BEFORE,
    supersedes_fact_id: str | None = None,
    metadata: dict | None = None,
) -> Fact:
    return Fact(
        id=fact_id,
        tenant_id="t1",
        conversation_id="c1",
        message_id=f"msg-{fact_id}",
        entity_id=entity_id,
        fact_key="forecast_signal",
        fact_text=text,
        confidence=0.8,
        supersedes_fact_id=supersedes_fact_id,
        valid_from=valid_from,
        recorded_from=recorded_from,
        metadata=metadata or {},
    )


def _relationship(message_id: str, source_id: str, target_id: str, evidence: str) -> Relationship:
    return Relationship(
        tenant_id="t1",
        conversation_id="c1",
        group_id="g1",
        message_id=message_id,
        source_id=source_id,
        target_id=target_id,
        rel_type=RelationshipType.RELATES_TO,
        confidence=0.7,
        evidence=evidence,
        valid_from=BEFORE,
        recorded_from=BEFORE,
    )


async def _compile(store: MemoryStore, *, target_id: str):
    compiler = AsOfEvidenceCompiler(MemoryGraphEvidenceAdapter(store))
    return await compiler.compile(_question(target_id=target_id))


def _question(*, target_id: str) -> ForecastQuestion:
    return ForecastQuestion(
        id="q-1",
        tenant_id="t1",
        title="Will Alice renew?",
        question_type=ForecastQuestionType.BINARY,
        forecast_as_of=AS_OF,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(description="Observed renewal status."),
        branches=[OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")],
        target_id=target_id,
    )
