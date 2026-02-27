"""API route definitions.

Endpoints from ARCHITECTURE.md section 6.2:
1. POST /messages - Ingest message with extraction pipeline
2. GET  /entities - List entities with filters
3. GET  /entities/{id} - Get entity by ID
4. GET  /entities/{id}/relationships - Get entity relationships
5. POST /entities/{id}/merge - Manual entity merge (Open Question #1)
6. GET  /query/point-in-time - Temporal query (world-state-as-of)
7. GET  /query/evolution - Evolution timeline
8. GET  /search - Entity search
9. GET  /health - Health check
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from engram.api.deps import get_pipeline, get_store
from engram.models.entity import EntityType
from engram.models.message import IngestRequest, IngestResponse
from engram.models.relationship import RelationshipType

if TYPE_CHECKING:
    from engram.api.deps import PipelineDep, StoreDep

logger = logging.getLogger(__name__)

router = APIRouter()


class MergeRequest(BaseModel):
    """Request body for entity merge endpoint."""

    duplicate_id: str


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        dict: Status and version information.
    """
    return {"status": "healthy", "version": "0.1.0"}


@router.post("/messages", response_model=IngestResponse)
async def ingest_message(
    request: IngestRequest,
    pipeline: PipelineDep = Depends(get_pipeline),  # noqa: B008
) -> IngestResponse:
    """Ingest a conversation message.

    Processes message through extraction pipeline:
    1. Check for duplicates (idempotency)
    2. Extract entities
    3. Infer relationships
    4. Resolve conflicts

    Args:
        request: Message ingestion request
        pipeline: Extraction pipeline

    Returns:
        IngestResponse with extraction results
    """
    return await pipeline.process_message(request)


@router.get("/entities")
async def list_entities(
    tenant_id: str = Query("default"),
    conversation_id: str | None = Query(None),
    entity_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> list[dict]:
    """List entities with optional filters.

    Args:
        tenant_id: Tenant identifier
        conversation_id: Optional conversation filter
        entity_type: Optional entity type filter (PERSON, PREFERENCE, etc.)
        limit: Maximum results (1-1000)
        offset: Pagination offset
        store: Graph store

    Returns:
        List of entities matching filters
    """
    # Parse entity type if provided
    parsed_type = None
    if entity_type:
        try:
            parsed_type = EntityType(entity_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid entity_type: {entity_type}")  # noqa: B904

    # Query store
    entities = await store.list_entities(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        entity_type=parsed_type,
        limit=limit,
        offset=offset,
    )

    # Convert to dict for JSON response
    return [
        {
            "id": e.id,
            "tenant_id": e.tenant_id,
            "conversation_id": e.conversation_id,
            "entity_type": e.entity_type.value,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases,
            "created_at": e.created_at.isoformat(),
            "last_mentioned": e.last_mentioned.isoformat(),
        }
        for e in entities
    ]


@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> dict:
    """Get entity by ID.

    Args:
        entity_id: Entity identifier
        store: Graph store

    Returns:
        Entity details

    Raises:
        HTTPException: 404 if entity not found
    """
    entity = await store.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

    return {
        "id": entity.id,
        "tenant_id": entity.tenant_id,
        "conversation_id": entity.conversation_id,
        "entity_type": entity.entity_type.value,
        "canonical_name": entity.canonical_name,
        "aliases": entity.aliases,
        "created_at": entity.created_at.isoformat(),
        "last_mentioned": entity.last_mentioned.isoformat(),
        "source_messages": entity.source_messages,
        "metadata": entity.metadata,
    }


@router.get("/entities/{entity_id}/relationships")
async def get_relationships(
    entity_id: str,
    rel_type: str | None = Query(None),
    tenant_id: str | None = Query(None),
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> list[dict]:
    """Get active relationships for an entity.

    Args:
        entity_id: Entity identifier
        rel_type: Optional relationship type filter
        store: Graph store

    Returns:
        List of active relationships
    """
    # Parse relationship type if provided
    parsed_type = None
    if rel_type:
        try:
            parsed_type = RelationshipType(rel_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid rel_type: {rel_type}")  # noqa: B904

    # Query store
    relationships = await store.get_active_relationships(
        entity_id=entity_id,
        rel_type=parsed_type,
        tenant_id=tenant_id,
    )

    # Convert to dict for JSON response
    return [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "rel_type": r.rel_type.value,
            "confidence": r.confidence,
            "evidence": r.evidence,
            "valid_from": r.valid_from.isoformat(),
            "valid_to": r.valid_to.isoformat() if r.valid_to else None,
            "recorded_from": r.recorded_from.isoformat(),
            "recorded_to": r.recorded_to.isoformat() if r.recorded_to else None,
            "version": r.version,
        }
        for r in relationships
    ]


@router.get("/query/point-in-time")
async def point_in_time_query(
    entity: str,
    as_of: str,
    tenant_id: str = Query("default"),
    rel_type: str | None = Query(None),
    mode: str = Query("world_state"),
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> list[dict] | dict[str, list[dict]]:
    """Query world state at a point in time.

    Supports three temporal modes:
    - world_state: What was actually true (valid_time)
    - knowledge: What we believed (record_time)
    - bitemporal: Both timelines

    Args:
        entity: Entity name to query
        as_of: ISO 8601 timestamp for point-in-time
        tenant_id: Tenant identifier
        rel_type: Optional relationship type filter
        mode: Temporal mode (world_state, knowledge, bitemporal)
        store: Graph store

    Returns:
        List of relationships valid at that point in time
    """
    # Parse timestamp
    try:
        as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {as_of}")  # noqa: B904

    # Parse relationship type if provided
    parsed_type = None
    if rel_type:
        try:
            parsed_type = RelationshipType(rel_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid rel_type: {rel_type}")  # noqa: B904

    # Query based on mode
    if mode == "knowledge":
        relationships = await store.query_knowledge_as_of(
            tenant_id=tenant_id,
            entity_name=entity,
            as_of=as_of_dt,
            rel_type=parsed_type,
        )
    elif mode == "bitemporal":
        world_state = await store.query_world_state_as_of(
            tenant_id=tenant_id,
            entity_name=entity,
            as_of=as_of_dt,
            rel_type=parsed_type,
        )
        knowledge = await store.query_knowledge_as_of(
            tenant_id=tenant_id,
            entity_name=entity,
            as_of=as_of_dt,
            rel_type=parsed_type,
        )
        return {
            "world_state": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "rel_type": r.rel_type.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "valid_from": r.valid_from.isoformat(),
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    "recorded_from": r.recorded_from.isoformat(),
                    "recorded_to": r.recorded_to.isoformat() if r.recorded_to else None,
                    "version": r.version,
                }
                for r in world_state
            ],
            "knowledge": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "rel_type": r.rel_type.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "valid_from": r.valid_from.isoformat(),
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    "recorded_from": r.recorded_from.isoformat(),
                    "recorded_to": r.recorded_to.isoformat() if r.recorded_to else None,
                    "version": r.version,
                }
                for r in knowledge
            ],
        }
    else:  # world_state
        relationships = await store.query_world_state_as_of(
            tenant_id=tenant_id,
            entity_name=entity,
            as_of=as_of_dt,
            rel_type=parsed_type,
        )

    # Convert to dict for JSON response
    return [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "rel_type": r.rel_type.value,
            "confidence": r.confidence,
            "evidence": r.evidence,
            "valid_from": r.valid_from.isoformat(),
            "valid_to": r.valid_to.isoformat() if r.valid_to else None,
            "recorded_from": r.recorded_from.isoformat(),
            "recorded_to": r.recorded_to.isoformat() if r.recorded_to else None,
            "version": r.version,
        }
        for r in relationships
    ]


@router.get("/query/evolution")
async def evolution_query(
    entity: str,
    tenant_id: str = Query("default"),
    target: str | None = Query(None),
    rel_type: str | None = Query(None),
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> list[dict]:
    """Query relationship evolution over time.

    Returns all versions of relationships for an entity,
    showing how they changed over time.

    Args:
        entity: Entity name to query
        tenant_id: Tenant identifier
        target: Optional target entity filter
        rel_type: Optional relationship type filter
        store: Graph store

    Returns:
        List of relationship versions in chronological order
    """
    # Parse relationship type if provided
    parsed_type = None
    if rel_type:
        try:
            parsed_type = RelationshipType(rel_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid rel_type: {rel_type}")  # noqa: B904

    # Query store
    relationships = await store.query_evolution(
        tenant_id=tenant_id,
        entity_name=entity,
        target_name=target,
        rel_type=parsed_type,
    )

    # Convert to dict for JSON response
    return [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "rel_type": r.rel_type.value,
            "confidence": r.confidence,
            "evidence": r.evidence,
            "valid_from": r.valid_from.isoformat(),
            "valid_to": r.valid_to.isoformat() if r.valid_to else None,
            "recorded_from": r.recorded_from.isoformat(),
            "recorded_to": r.recorded_to.isoformat() if r.recorded_to else None,
            "version": r.version,
            "supersedes": r.supersedes,
        }
        for r in relationships
    ]


@router.get("/search")
async def search_entities(
    q: str = Query(""),
    tenant_id: str = Query("default"),
    limit: int = Query(20, ge=1, le=100),
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> list[dict]:
    """Search for entities by name.

    Simple substring search on canonical_name and aliases.
    In production, would use full-text search or embeddings.

    Args:
        q: Search query
        tenant_id: Tenant identifier
        limit: Maximum results
        store: Graph store

    Returns:
        List of matching entities
    """
    # Get all entities for tenant (MVP approach)
    all_entities = await store.list_entities(
        tenant_id=tenant_id,
        limit=1000,  # Get all for search
    )

    # Filter by query (simple substring match)
    if not q:
        results = all_entities[:limit]
    else:
        q_lower = q.lower()
        results = [
            e
            for e in all_entities
            if q_lower in e.canonical_name.lower()
            or any(q_lower in alias.lower() for alias in e.aliases)
        ][:limit]

    # Convert to dict for JSON response
    return [
        {
            "id": e.id,
            "tenant_id": e.tenant_id,
            "conversation_id": e.conversation_id,
            "entity_type": e.entity_type.value,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases,
            "created_at": e.created_at.isoformat(),
            "last_mentioned": e.last_mentioned.isoformat(),
        }
        for e in results
    ]


@router.post("/entities/{entity_id}/merge")
async def merge_entities(
    entity_id: str,
    request: MergeRequest,
    store: StoreDep = Depends(get_store),  # noqa: B008
) -> dict:
    """Manual entity merge endpoint (Open Question #1 - Entity Merging).

    Merges duplicate_id into entity_id:
    1. Transfer all relationships from duplicate to primary
    2. Update relationship endpoints to point to primary
    3. Mark duplicate as merged (set metadata.merged_into = entity_id)
    4. Optionally delete duplicate node

    This is the MVP approach: manual merge via API, no auto-merge.
    v0.2 will add merge candidates queue with similarity scores.

    Args:
        entity_id: Primary entity ID (merge target)
        request: Merge request with duplicate_id
        store: Graph store

    Returns:
        Merge result with statistics

    Raises:
        HTTPException: 400 if same ID, 404 if entities not found
    """
    duplicate_id = request.duplicate_id

    # Validate inputs
    if entity_id == duplicate_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot merge entity with itself",
        )

    # Check both entities exist
    primary = await store.get_entity(entity_id)
    duplicate = await store.get_entity(duplicate_id)

    if not primary:
        raise HTTPException(status_code=404, detail=f"Primary entity not found: {entity_id}")
    if not duplicate:
        raise HTTPException(status_code=404, detail=f"Duplicate entity not found: {duplicate_id}")

    relationships_transferred = await store.merge_entity_into(entity_id, duplicate_id)

    return {
        "primary_id": entity_id,
        "duplicate_id": duplicate_id,
        "relationships_transferred": relationships_transferred,
        "status": "merged",
        "message": f"Merged {duplicate_id} into {entity_id}",
    }
