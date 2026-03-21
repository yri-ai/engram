"""Relationship model with bitemporal versioning."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RelationshipType(StrEnum):
    PREFERS = "prefers"
    AVOIDS = "avoids"
    KNOWS = "knows"
    DISCUSSED = "discussed"
    MENTIONED_WITH = "mentioned_with"
    HAS_GOAL = "has_goal"
    RELATES_TO = "relates_to"


class Evidence(BaseModel):
    """Structured evidence for a relationship."""

    message_id: str
    text: str  # Exact quote or snippet
    context: str | None = None  # Surrounding text for context
    observed_at: datetime


class Relationship(BaseModel):
    """A bitemporal edge in the knowledge graph."""

    # Scoping
    tenant_id: str
    conversation_id: str
    group_id: str | None = None
    message_id: str
    extraction_run_id: str | None = None

    # Endpoints
    source_id: str
    target_id: str

    # Semantics
    rel_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""  # Legacy simple evidence
    structured_evidence: list[Evidence] = Field(default_factory=list)

    # Bitemporal (4 time columns)
    valid_from: datetime
    valid_to: datetime | None = None  # NULL = still true
    recorded_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recorded_to: datetime | None = None  # NULL = still believed

    # Evolution
    version: int = 1
    supersedes: str | None = None

    metadata: dict = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """True if relationship is currently valid (truth timeline)."""
        return self.valid_to is None

    @property
    def is_currently_believed(self) -> bool:
        """True if we still believe this relationship (knowledge timeline)."""
        return self.recorded_to is None


class ExclusivityPolicy(BaseModel):
    """Defines which relationships are mutually exclusive."""

    exclusivity_scope: tuple[str, ...] | None = None
    max_active: int | None = None
    close_on_new: bool = False
    exclusive_with: list[str] = Field(default_factory=list)
