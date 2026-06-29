"""Leakage-safe as-of evidence compilation for forecast questions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from engram.models.forecasting import EvidenceDossier, EvidenceItem, ForecastQuestion

if TYPE_CHECKING:
    from engram.models.fact import Fact
    from engram.models.relationship import Relationship
    from engram.storage.memory import MemoryStore


@dataclass(frozen=True)
class GraphEvidencePacket:
    """Raw graph records selected for a target as of a forecast time."""

    facts: list[Fact] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    excluded_counts: dict[str, int] = field(default_factory=dict)
    supersession_scan_facts: list[Fact] = field(default_factory=list)


@runtime_checkable
class GraphEvidenceAdapter(Protocol):
    """Data-source seam used by the as-of evidence compiler."""

    async def get_facts_as_of(
        self,
        tenant_id: str,
        entity_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> tuple[list[Fact], dict[str, int]]:
        """Return facts for an entity that were knowable as of ``as_of``."""

    async def get_relationships_as_of(
        self,
        tenant_id: str,
        entity_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> tuple[list[Relationship], dict[str, int]]:
        """Return inbound and outbound relationships knowable as of ``as_of``."""

    async def get_target_evidence_as_of(
        self,
        tenant_id: str,
        target_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> GraphEvidencePacket:
        """Return target facts, target relationships, and one-hop related facts."""


class MemoryGraphEvidenceAdapter:
    """GraphEvidenceAdapter backed by the in-memory graph store."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def get_facts_as_of(
        self,
        tenant_id: str,
        entity_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> tuple[list[Fact], dict[str, int]]:
        included: list[Fact] = []
        excluded_counts: Counter[str] = Counter()
        for fact in self.store._facts.values():
            if fact.tenant_id != tenant_id or fact.entity_id != entity_id:
                continue
            reason = _exclusion_reason(fact, as_of, question_id=question_id)
            if reason is not None:
                excluded_counts[reason] += 1
                continue
            included.append(fact)
        included.sort(key=lambda fact: (fact.recorded_from, fact.id))
        return included, dict(excluded_counts)

    async def get_relationships_as_of(
        self,
        tenant_id: str,
        entity_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> tuple[list[Relationship], dict[str, int]]:
        included: list[Relationship] = []
        excluded_counts: Counter[str] = Counter()
        for relationship in self.store._relationships:
            if relationship.tenant_id != tenant_id:
                continue
            if relationship.source_id != entity_id and relationship.target_id != entity_id:
                continue
            reason = _exclusion_reason(relationship, as_of, question_id=question_id)
            if reason is not None:
                excluded_counts[reason] += 1
                continue
            included.append(relationship)
        included.sort(
            key=lambda relationship: (
                relationship.recorded_from,
                relationship.message_id,
                relationship.source_id,
                str(relationship.rel_type),
                relationship.target_id,
                relationship.version,
            )
        )
        return included, dict(excluded_counts)

    async def get_target_evidence_as_of(
        self,
        tenant_id: str,
        target_id: str,
        as_of: datetime,
        *,
        question_id: str | None = None,
    ) -> GraphEvidencePacket:
        excluded_counts: Counter[str] = Counter()
        direct_facts, fact_exclusions = await self.get_facts_as_of(
            tenant_id,
            target_id,
            as_of,
            question_id=question_id,
        )
        relationships, relationship_exclusions = await self.get_relationships_as_of(
            tenant_id,
            target_id,
            as_of,
            question_id=question_id,
        )
        excluded_counts.update(fact_exclusions)
        excluded_counts.update(relationship_exclusions)

        related_ids = {
            relationship.target_id if relationship.source_id == target_id else relationship.source_id
            for relationship in relationships
        }
        facts_by_id = {fact.id: fact for fact in direct_facts}
        scan_entity_ids = {target_id, *related_ids}
        for related_id in sorted(related_ids):
            related_facts, related_exclusions = await self.get_facts_as_of(
                tenant_id,
                related_id,
                as_of,
                question_id=question_id,
            )
            excluded_counts.update(related_exclusions)
            for fact in related_facts:
                facts_by_id[fact.id] = fact

        facts = sorted(facts_by_id.values(), key=lambda fact: (fact.recorded_from, fact.id))
        return GraphEvidencePacket(
            facts=facts,
            relationships=relationships,
            excluded_counts=dict(excluded_counts),
            supersession_scan_facts=sorted(
                [
                    fact
                    for fact in self.store._facts.values()
                    if fact.tenant_id == tenant_id and fact.entity_id in scan_entity_ids
                ],
                key=lambda fact: (fact.recorded_from, fact.id),
            ),
        )


class AsOfEvidenceCompiler:
    """Compile graph records into an EvidenceDossier for a forecast question."""

    def __init__(self, adapter: GraphEvidenceAdapter) -> None:
        self.adapter = adapter

    async def compile(self, question: ForecastQuestion) -> EvidenceDossier:
        if question.target_id is None:
            return EvidenceDossier(
                id=f"dossier-{question.id}",
                question_id=question.id,
                forecast_as_of=question.forecast_as_of,
                evidence_items=[],
                missing_evidence=["target_id"],
                compiler="as_of_evidence.v1",
            )

        packet = await self.adapter.get_target_evidence_as_of(
            question.tenant_id,
            question.target_id,
            question.forecast_as_of,
            question_id=question.id,
        )
        fact_items = self._compile_fact_items(
            packet.facts,
            question.forecast_as_of,
            packet.supersession_scan_facts or packet.facts,
        )
        relationship_items = [_relationship_to_item(relationship) for relationship in packet.relationships]
        evidence_items = sorted(fact_items + relationship_items, key=lambda item: item.id)

        return EvidenceDossier(
            id=f"dossier-{question.id}",
            question_id=question.id,
            forecast_as_of=question.forecast_as_of,
            evidence_items=evidence_items,
            excluded_counts=packet.excluded_counts,
            compiler="as_of_evidence.v1",
            metadata={"target_id": question.target_id},
        )

    def _compile_fact_items(
        self,
        facts: list[Fact],
        as_of: datetime,
        supersession_scan_facts: list[Fact],
    ) -> list[EvidenceItem]:
        replacement_by_old_id = {
            fact.supersedes_fact_id: fact
            for fact in supersession_scan_facts
            if fact.supersedes_fact_id is not None
        }
        known_replacements_by_old_id = {
            old_id: replacement
            for old_id, replacement in replacement_by_old_id.items()
            if replacement.recorded_from <= as_of
        }
        items: list[EvidenceItem] = []
        for fact in facts:
            item_id = _fact_item_id(fact)
            replacement = replacement_by_old_id.get(fact.id)
            known_replacement = known_replacements_by_old_id.get(fact.id)
            supersession_status = "current_as_of"
            superseded_by_id = None
            if known_replacement is not None:
                supersession_status = "superseded_before_as_of"
                superseded_by_id = _fact_item_id(known_replacement)
            elif replacement is not None:
                supersession_status = "current_as_of_later_superseded"
                superseded_by_id = _fact_item_id(replacement)

            items.append(
                EvidenceItem(
                    id=item_id,
                    text=fact.fact_text,
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    recorded_from=fact.recorded_from,
                    recorded_to=fact.recorded_to,
                    source_id=fact.message_id,
                    source_span=fact.metadata.get("source_span"),
                    supports_branch=list(fact.metadata.get("supports_branch", [])),
                    opposes_branch=list(fact.metadata.get("opposes_branch", [])),
                    supersession_status=supersession_status,
                    supersedes_id=_fact_item_id_from_raw(fact.supersedes_fact_id),
                    superseded_by_id=superseded_by_id,
                    contradicts_ids=[_fact_item_id_from_raw(value) for value in fact.metadata.get("contradicts_ids", [])],
                    confidence=fact.confidence,
                    metadata={
                        "entity_id": fact.entity_id,
                        "fact_key": fact.fact_key,
                        **fact.metadata,
                    },
                )
            )
        return items


def _relationship_to_item(relationship: Relationship) -> EvidenceItem:
    return EvidenceItem(
        id=_relationship_item_id(relationship),
        text=relationship.evidence
        or f"{relationship.source_id} {relationship.rel_type} {relationship.target_id}",
        valid_from=relationship.valid_from,
        valid_to=relationship.valid_to,
        recorded_from=relationship.recorded_from,
        recorded_to=relationship.recorded_to,
        source_id=relationship.message_id,
        source_span=relationship.metadata.get("source_span"),
        supports_branch=list(relationship.metadata.get("supports_branch", [])),
        opposes_branch=list(relationship.metadata.get("opposes_branch", [])),
        supersession_status="current_as_of",
        supersedes_id=None,
        superseded_by_id=None,
        contradicts_ids=list(relationship.metadata.get("contradicts_ids", [])),
        confidence=relationship.confidence,
        metadata={
            "source_entity_id": relationship.source_id,
            "target_entity_id": relationship.target_id,
            "relationship_type": str(relationship.rel_type),
            **relationship.metadata,
        },
    )


def _exclusion_reason(record: Fact | Relationship, as_of: datetime, *, question_id: str | None) -> str | None:
    if record.recorded_from > as_of:
        return "future_record_time"
    if record.recorded_to is not None and record.recorded_to <= as_of:
        return "recorded_ended_before_as_of"
    if record.valid_from > as_of:
        return "future_valid_time"

    metadata = record.metadata
    if _metadata_datetime(metadata.get("source_ingested_at")) is not None:
        source_ingested_at = _metadata_datetime(metadata.get("source_ingested_at"))
        if source_ingested_at is not None and source_ingested_at > as_of:
            return "source_ingested_after_as_of"
    if metadata.get("evidence_role") == "resolution_evidence":
        return "resolution_evidence"
    if question_id is not None and metadata.get("resolution_for_question_id") == question_id:
        return "resolution_evidence"
    derived_after = _metadata_datetime(metadata.get("derived_after"))
    if derived_after is not None and derived_after > as_of:
        return "post_hoc_derived"
    return None


def _metadata_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _fact_item_id(fact: Fact) -> str:
    return f"fact:{fact.id}"


def _fact_item_id_from_raw(fact_id: str | None) -> str | None:
    if fact_id is None:
        return None
    return f"fact:{fact_id}"


def _relationship_item_id(relationship: Relationship) -> str:
    return (
        f"relationship:{relationship.message_id}:{relationship.source_id}:"
        f"{relationship.rel_type}:{relationship.target_id}:{relationship.version}"
    )
