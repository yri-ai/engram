"""Temporal query models."""

from datetime import datetime

from pydantic import BaseModel


class PointInTimeQuery(BaseModel):
    """Query for a specific point in time."""

    tenant_id: str = "default"
    entity_name: str
    as_of: datetime
    relationship_type: str | None = None
    mode: str = "world_state"  # "world_state" | "knowledge" | "bitemporal"
    knowledge_date: datetime | None = None  # Only for bitemporal mode


class EvolutionQuery(BaseModel):
    """Query for relationship evolution over time."""

    tenant_id: str = "default"
    entity_name: str
    target_name: str | None = None
    relationship_type: str | None = None


class SearchQuery(BaseModel):
    """General search query across the knowledge graph."""

    tenant_id: str = "default"
    query: str
    entity_type: str | None = None
    relationship_type: str | None = None
    limit: int = 10
