"""Standalone knowledge claims with temporal tracking and supersession."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class Fact(BaseModel):
    """A standalone knowledge claim about an entity.

    Unlike relationships (entity->entity edges), facts capture knowledge
    that belongs to a single entity: "Alice is 32", "Bob works at Acme",
    "The project deadline is March 30th".
    """

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str | None = None

    entity_id: str

    fact_key: str  # Semantic key for grouping (e.g., "age", "employer", "location")
    fact_text: str  # Human-readable fact statement
    confidence: float = Field(ge=0.0, le=1.0)

    status: FactStatus = FactStatus.ACTIVE
    supersedes_fact_id: str | None = None

    # Bitemporal
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_to: datetime | None = None
    recorded_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recorded_to: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, message_id: str, index: int) -> str:
        return f"{tenant_id}:fact:{message_id}:{index}"
