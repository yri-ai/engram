"""Entity model for knowledge graph nodes."""

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    PERSON = "PERSON"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    TOPIC = "TOPIC"


class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str
    tenant_id: str
    conversation_id: str
    group_id: str | None = None
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_mentioned: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_messages: list[str] = Field(default_factory=list)
    extraction_run_id: str | None = None
    metadata: dict = Field(default_factory=dict)

    @staticmethod
    def build_id(
        tenant_id: str,
        entity_type: EntityType,
        canonical_name: str,
        group_id: str | None = None,
    ) -> str:
        """Build deterministic entity ID.

        Args:
            tenant_id: Tenant identifier
            entity_type: Type of entity
            canonical_name: Normalized canonical name
            group_id: Optional group identifier for cross-conversation linking.
                     If None, entities are conversation-scoped (isolated).
                     If provided, entities are shared across all conversations in that group.

        Returns:
            Deterministic entity ID: {tenant}:{scope}:{type}:{canonical}
        """
        scope = group_id if group_id else "conversation"
        return f"{tenant_id}:{scope}:{entity_type}:{canonical_name}"

    @staticmethod
    def normalize_name(text: str) -> str:
        """Deterministic name normalization. Same input -> same output."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", "-", normalized.strip())
