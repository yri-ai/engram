"""Snapshot service -- captures conversation state after each extraction."""

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
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=500,
        )

        # Count actual relationships and facts across all entities
        rel_ids: set[tuple[str, str, str]] = set()
        total_fact_count = 0
        for e in entities:
            rels = await self._store.get_active_relationships(e.id)
            for r in rels:
                rel_ids.add((r.source_id, r.target_id, r.rel_type.value))
            facts = await self._store.get_facts(tenant_id, e.id)
            total_fact_count += len(facts)

        # Build deltas from what was just extracted
        deltas: list[SnapshotDelta] = []
        if new_entities:
            for e in new_entities:
                deltas.append(
                    SnapshotDelta(
                        change_type=ChangeType.ADDED,
                        artifact_type="entity",
                        artifact_id=e.id,
                        summary=f"New entity: {e.canonical_name} ({e.entity_type.value})",
                    )
                )
        if new_relationships:
            for r in new_relationships:
                deltas.append(
                    SnapshotDelta(
                        change_type=ChangeType.ADDED,
                        artifact_type="relationship",
                        artifact_id=f"{r.source_id}->{r.target_id}",
                        summary=f"{r.source_id.split(':')[-1]} {r.rel_type.value} {r.target_id.split(':')[-1]}",
                    )
                )
        if new_facts:
            for f in new_facts:
                change = ChangeType.SUPERSEDED if f.supersedes_fact_id else ChangeType.ADDED
                deltas.append(
                    SnapshotDelta(
                        change_type=change,
                        artifact_type="fact",
                        artifact_id=f.id,
                        summary=f"{f.fact_key}: {f.fact_text}",
                    )
                )

        return ConversationSnapshot(
            id=f"snap-{uuid.uuid4()}",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_id=message_id,
            extraction_run_id=run_id,
            entity_count=len(entities),
            relationship_count=len(rel_ids),
            fact_count=total_fact_count,
            entities=[e.canonical_name for e in entities],
            deltas=deltas,
        )
