"""Model for recurring behavioral patterns identified from history."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Pattern(BaseModel):
    """A recurring habit, preference, or behavioral pattern."""

    id: str
    tenant_id: str
    entity_id: str  # Who this pattern applies to

    text: str  # Description (e.g., "Prefers morning workouts")
    category: str  # e.g., "HABIT", "PREFERENCE", "SOCIAL"
    frequency: str | None = None  # e.g., "Daily", "Weekly"

    # Evidence
    source_run_ids: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)

    # Temporal
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_observed: datetime = Field(default_factory=lambda: datetime.now(UTC))

    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, entity_id: str, category: str, text_slug: str) -> str:
        """Build deterministic pattern ID."""
        import hashlib

        h = hashlib.md5(text_slug.encode()).hexdigest()[:8]
        return f"{tenant_id}:pattern:{entity_id}:{category}:{h}"
