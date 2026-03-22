"""Model for extracted commitments and future-oriented actions."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CommitmentStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MISSED = "missed"


class Commitment(BaseModel):
    """A commitment or intention expressed by an entity."""

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str | None = None

    entity_id: str  # Who made the commitment
    text: str  # The commitment itself
    status: CommitmentStatus = CommitmentStatus.ACTIVE

    # Temporal
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    target_date: datetime | None = None  # When it should be completed
    completed_at: datetime | None = None

    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, message_id: str, index: int) -> str:
        """Build deterministic commitment ID."""
        return f"{tenant_id}:commitment:{message_id}:{index}"
