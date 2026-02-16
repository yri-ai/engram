"""Message models for ingestion."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest a conversation message."""

    text: str
    speaker: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    conversation_id: str = "default"
    tenant_id: str = "default"
    group_id: str | None = None
    message_id: str | None = None  # Auto-generated if not provided
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from message ingestion."""

    message_id: str
    entities_extracted: int
    relationships_inferred: int
    conflicts_resolved: int
    processing_time_ms: float
