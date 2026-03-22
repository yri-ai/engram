"""Conversation state snapshots and delta tracking.

Inspired by temporal-relationships' BuildForRun/delta tracking pattern.
After each extraction, capture the conversation state and what changed.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChangeType(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    SUPERSEDED = "superseded"


class SnapshotDelta(BaseModel):
    """A single change that occurred during an extraction."""

    change_type: ChangeType
    artifact_type: str  # "entity", "relationship", "fact", "commitment"
    artifact_id: str
    summary: str  # Human-readable description


class ConversationSnapshot(BaseModel):
    """Point-in-time snapshot of a conversation's knowledge state."""

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str

    # Counts
    entity_count: int = 0
    relationship_count: int = 0
    fact_count: int = 0
    commitment_count: int = 0

    # Entity names (lightweight summary, not full objects)
    entities: list[str] = Field(default_factory=list)

    # What changed in this extraction
    deltas: list[SnapshotDelta] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
