"""Model for tracking LLM extraction runs and metadata."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionRun(BaseModel):
    """Metadata for a single extraction execution."""

    id: str
    tenant_id: str
    conversation_id: str | None = None
    message_id: str | None = None

    # Context
    prompt_id: str  # Name of the template used
    prompt_sha256: str | None = None  # Version of the prompt
    provider: str
    model: str
    temperature: float = 0.0

    # Status
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_text: str | None = None

    # Performance
    total_tokens: int | None = None
    processing_time_ms: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
