"""Abstract storage interface for graph operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from engram.models.entity import Entity, EntityType
    from engram.models.relationship import Relationship, RelationshipType


class GraphStore(ABC):
    """Abstract graph storage interface.

    Implementations: Neo4jStore (production), MemoryStore (testing).
    """

    # --- Lifecycle ---

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema, indexes, constraints."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if storage is healthy."""
        ...

    # --- Entity Operations ---

    @abstractmethod
    async def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update entity. Idempotent via MERGE on entity.id."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    async def get_entity_by_name(
        self, tenant_id: str, conversation_id: str, canonical_name: str
    ) -> Entity | None:
        """Find entity by canonical name within a conversation."""
        ...

    @abstractmethod
    async def list_entities(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with filters."""
        ...

    # --- Relationship Operations ---

    @abstractmethod
    async def create_relationship(self, rel: Relationship) -> Relationship:
        """Create a new relationship."""
        ...

    @abstractmethod
    async def get_active_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        tenant_id: str | None = None,
    ) -> list[Relationship]:
        """Get currently active relationships (valid_to IS NULL AND recorded_to IS NULL)."""
        ...

    @abstractmethod
    async def terminate_relationship(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        conversation_id: str,
        termination_time: datetime,
        exclude_target_id: str | None = None,
    ) -> int:
        """Terminate active relationships. Returns count of terminated."""
        ...

    # --- Temporal Queries ---

    @abstractmethod
    async def query_world_state_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What was ACTUALLY TRUE at a point in time (valid time)."""
        ...

    @abstractmethod
    async def query_knowledge_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What did we BELIEVE at a point in time (record time)."""
        ...

    @abstractmethod
    async def query_evolution(
        self,
        tenant_id: str,
        entity_name: str,
        target_name: str | None = None,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """Get all versions of relationships for timeline view."""
        ...

    @abstractmethod
    async def get_recent_entities(
        self,
        tenant_id: str,
        conversation_id: str,
        since: datetime,
        limit: int = 20,
    ) -> list[Entity]:
        """Get recently mentioned entities (for LLM context building)."""
        ...
