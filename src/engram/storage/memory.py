"""In-memory GraphStore implementation for testing.

WARNING: Not production-safe. No persistence, no concurrency safety.
Use Neo4jStore for production workloads.

This serves as the TDD reference implementation -- every test here
validates the contract that Neo4jStore must also satisfy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from engram.models.entity import Entity
from engram.storage.base import GraphStore

if TYPE_CHECKING:
    from datetime import datetime

    from engram.models.entity import EntityType
    from engram.models.relationship import Relationship, RelationshipType

logger = logging.getLogger(__name__)


class MemoryStore(GraphStore):
    """In-memory graph storage using dicts and lists.

    Entity storage: ``dict[str, Entity]`` keyed by entity ID.
    Relationship storage: ``list[Relationship]`` (append-only).
    Temporal queries: list-comprehension filtering on valid_from/to, recorded_from/to.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Nothing to initialize for in-memory store."""

    async def close(self) -> None:
        """Clear all data."""
        self._entities.clear()
        self._relationships.clear()

    async def health_check(self) -> bool:
        """Always healthy."""
        return True

    # --- Entity Operations ---

    async def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update entity. Merges aliases and source_messages on update."""
        existing = self._entities.get(entity.id)
        if existing is not None:
            # Merge: update last_mentioned, accumulate aliases and source_messages
            existing.last_mentioned = entity.last_mentioned
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            existing.source_messages.extend(entity.source_messages)
            return existing
        self._entities[entity.id] = entity
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    async def get_entity_by_name(
        self, tenant_id: str, conversation_id: str, canonical_name: str
    ) -> Entity | None:
        """Find entity by canonical name within a conversation."""
        for entity in self._entities.values():
            if (
                entity.tenant_id == tenant_id
                and entity.conversation_id == conversation_id
                and entity.canonical_name == canonical_name
            ):
                return entity
        return None

    async def list_entities(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filters, pagination via limit/offset."""
        result = [e for e in self._entities.values() if e.tenant_id == tenant_id]
        if conversation_id is not None:
            result = [e for e in result if e.conversation_id == conversation_id]
        if entity_type is not None:
            result = [e for e in result if e.entity_type == entity_type]
        return result[offset : offset + limit]

    # --- Relationship Operations ---

    async def create_relationship(self, rel: Relationship) -> Relationship:
        """Append a new relationship to the store."""
        self._relationships.append(rel)
        return rel

    async def get_active_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        tenant_id: str | None = None,
    ) -> list[Relationship]:
        """Get currently active relationships (valid_to IS NULL AND recorded_to IS NULL)."""
        result: list[Relationship] = []
        for r in self._relationships:
            if r.source_id != entity_id:
                continue
            if not r.is_active or not r.is_currently_believed:
                continue
            if rel_type is not None and r.rel_type != rel_type:
                continue
            if tenant_id is not None and r.tenant_id != tenant_id:
                continue
            result.append(r)
        return result

    async def terminate_relationship(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        conversation_id: str,
        termination_time: datetime,
        exclude_target_id: str | None = None,
    ) -> int:
        """Terminate active relationships matching criteria. Returns count terminated."""
        count = 0
        for r in self._relationships:
            if (
                r.source_id == source_id
                and r.rel_type == rel_type
                and r.tenant_id == tenant_id
                and r.conversation_id == conversation_id
                and r.is_active
                and r.is_currently_believed
                and (exclude_target_id is None or r.target_id != exclude_target_id)
            ):
                r.valid_to = termination_time
                r.recorded_to = termination_time
                count += 1
        return count

    # --- Temporal Queries ---

    async def query_world_state_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What was ACTUALLY TRUE at a point in time (valid time).

        Filter: valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of)
        """
        entity_ids = self._find_entity_ids_by_name(tenant_id, entity_name)
        result: list[Relationship] = []
        for r in self._relationships:
            if r.source_id not in entity_ids:
                continue
            if r.tenant_id != tenant_id:
                continue
            if rel_type is not None and r.rel_type != rel_type:
                continue
            if r.valid_from <= as_of and (r.valid_to is None or r.valid_to > as_of):
                result.append(r)
        return result

    async def query_knowledge_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What did we BELIEVE at a point in time (record time).

        Filter: recorded_from <= as_of AND (recorded_to IS NULL OR recorded_to > as_of)
        """
        entity_ids = self._find_entity_ids_by_name(tenant_id, entity_name)
        result: list[Relationship] = []
        for r in self._relationships:
            if r.source_id not in entity_ids:
                continue
            if r.tenant_id != tenant_id:
                continue
            if rel_type is not None and r.rel_type != rel_type:
                continue
            if r.recorded_from <= as_of and (r.recorded_to is None or r.recorded_to > as_of):
                result.append(r)
        return result

    async def query_evolution(
        self,
        tenant_id: str,
        entity_name: str,
        target_name: str | None = None,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """Get all versions of relationships for timeline view (no temporal filter)."""
        entity_ids = self._find_entity_ids_by_name(tenant_id, entity_name)
        target_ids: set[str] | None = None
        if target_name is not None:
            target_ids = self._find_entity_ids_by_name(tenant_id, target_name)

        result: list[Relationship] = []
        for r in self._relationships:
            if r.source_id not in entity_ids:
                continue
            if r.tenant_id != tenant_id:
                continue
            if target_ids is not None and r.target_id not in target_ids:
                continue
            if rel_type is not None and r.rel_type != rel_type:
                continue
            result.append(r)
        return result

    async def get_recent_entities(
        self,
        tenant_id: str,
        conversation_id: str,
        since: datetime,
        limit: int = 20,
    ) -> list[Entity]:
        """Get recently mentioned entities, ordered by recency."""
        result = [
            e
            for e in self._entities.values()
            if e.tenant_id == tenant_id
            and e.conversation_id == conversation_id
            and e.last_mentioned >= since
        ]
        result.sort(key=lambda e: e.last_mentioned, reverse=True)
        return result[:limit]

    # --- Internal Helpers ---

    def _find_entity_ids_by_name(self, tenant_id: str, name: str) -> set[str]:
        """Find all entity IDs matching a canonical name within a tenant."""
        normalized = Entity.normalize_name(name)
        return {
            e.id
            for e in self._entities.values()
            if e.tenant_id == tenant_id and e.canonical_name == normalized
        }
