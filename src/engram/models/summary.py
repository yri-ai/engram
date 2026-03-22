"""Conversation summary model — narrative arc per message/session.

Inspired by temporal-relationships' SessionArc artifact type.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """High-level narrative summary of a conversation segment."""

    id: str
    tenant_id: str
    conversation_id: str
    message_id: str
    extraction_run_id: str | None = None

    # Narrative arc (opening -> shift -> closing)
    opening_state: str  # What was the context/state going in
    key_shift: str | None = None  # What changed (if anything)
    closing_state: str  # Where things ended up

    breakthrough: bool = False  # Was there a significant revelation?

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(tenant_id: str, message_id: str) -> str:
        return f"{tenant_id}:summary:{message_id}"
