"""3-stage extraction pipeline: Entity -> Relationship -> Conflict Resolution.

See ARCHITECTURE.md section 2 for full design.

Integrates:
- Open Question #1: Vector-based entity resolution (duplicate detection via embeddings)
- Open Question #2: Relationship type validation (unknown → relates_to fallback)
- Open Question #3: group_id scoping for cross-conversation entity linking
- Open Question #4: Confidence snapping to discrete semantic levels
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from engram.llm.prompts import (
    build_entity_extraction_prompt,
    build_relationship_inference_prompt,
)
from engram.models.entity import Entity, EntityType
from engram.models.message import IngestRequest, IngestResponse
from engram.models.relationship import Relationship, RelationshipType

if TYPE_CHECKING:
    from engram.llm.provider import LLMProvider
    from engram.services.dedup import DedupService
    from engram.services.embeddings import EmbeddingService
    from engram.services.resolution import ConflictResolver
    from engram.storage.base import GraphStore

logger = logging.getLogger(__name__)

# Context lookback for building LLM prompts with recent entities
_CONTEXT_LOOKBACK_DAYS = 7


def snap_confidence(raw: float, levels: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)) -> float:
    """Snap raw LLM confidence to nearest defined level.

    Open Question #4 (Confidence Calibration): LLM self-reported confidence
    is poorly calibrated. Snapping to discrete levels ensures semantic
    consistency: 1.0=direct, 0.8=strong implication, 0.6=weak inference,
    0.4=co-occurrence.

    Args:
        raw: Raw confidence from LLM (0.0-1.0).
        levels: Allowed confidence levels, ordered descending.

    Returns:
        Confidence snapped to nearest level.
    """
    return min(levels, key=lambda level: abs(level - raw))


class ExtractionPipeline:
    """Full 3-stage extraction pipeline.

    Stage 1: Entity extraction (LLM)
    Stage 2: Relationship inference (LLM)
    Stage 3: Conflict resolution (rule-based via ConflictResolver)

    Integrates vector-based entity resolution (Open Question #1) via optional
    EmbeddingService for duplicate detection.
    """

    def __init__(
        self,
        store: GraphStore,
        dedup: DedupService,
        resolver: ConflictResolver,
        llm: LLMProvider,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._store = store
        self._dedup = dedup
        self._resolver = resolver
        self._llm = llm
        self._embedding_service = embedding_service

    async def process_message(self, request: IngestRequest) -> IngestResponse:
        """Process a conversation message through the full extraction pipeline.

        Returns an IngestResponse with counts of entities extracted,
        relationships inferred, and conflicts resolved.
        """
        start = time.monotonic()
        message_id = request.message_id or str(uuid.uuid4())

        # Step 0: Idempotency check — mark before pipeline so concurrent duplicates
        # are rejected atomically. Rolled back if the pipeline fails so retries work.
        is_new = await self._dedup.check_and_mark(message_id)
        if not is_new:
            return IngestResponse(
                message_id=message_id,
                entities_extracted=0,
                relationships_inferred=0,
                conflicts_resolved=0,
                processing_time_ms=0,
            )

        try:
            # Step 1: Entity extraction (LLM)
            entities, raw_entity_items = await self._extract_entities(request, message_id)

            # No entities → nothing to infer, skip Stage 2
            if not entities:
                elapsed_ms = (time.monotonic() - start) * 1000
                return IngestResponse(
                    message_id=message_id,
                    entities_extracted=0,
                    relationships_inferred=0,
                    conflicts_resolved=0,
                    processing_time_ms=round(elapsed_ms, 2),
                )

            # Step 2: Relationship inference (LLM)
            relationships = await self._infer_relationships(
                request, entities, raw_entity_items, message_id
            )

            # Step 3: Conflict resolution + graph writes
            conflicts = 0
            for rel in relationships:
                _, terminated = await self._resolver.resolve_and_create(rel)
                conflicts += terminated

        except Exception:
            await self._dedup.rollback(message_id)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        return IngestResponse(
            message_id=message_id,
            entities_extracted=len(entities),
            relationships_inferred=len(relationships),
            conflicts_resolved=conflicts,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ── Stage 1: Entity Extraction ───────────────────────────────────────

    async def _extract_entities(
        self, request: IngestRequest, message_id: str
    ) -> tuple[list[Entity], list[dict[str, Any]]]:
        """Extract entities from message text via LLM.

        Returns:
            Tuple of (Entity objects upserted to store, raw LLM items for
            relationship inference context).
        """
        context = await self._get_context_entities(request)

        prompt = build_entity_extraction_prompt(
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            context_entities=context,
        )

        result = await self._llm.complete_json(prompt)
        raw_items: list[dict[str, Any]] = result.get("entities", [])

        # Open Question #3: group_id for cross-conversation entity linking
        # Defaults to conversation_id (conversation-scoped isolation)
        group_id = request.group_id or request.conversation_id

        entities: list[Entity] = []
        for item in raw_items:
            try:
                entity_type = EntityType[item["type"]]
            except KeyError:
                logger.warning(
                    "Unknown entity type '%s', skipping entity '%s'",
                    item.get("type"),
                    item.get("name"),
                )
                continue

            # Use LLM canonical directly (it already follows normalize conventions)
            canonical = item["canonical"]
            entity_id = Entity.build_id(
                tenant_id=request.tenant_id,
                entity_type=entity_type,
                canonical_name=canonical,
                group_id=group_id,
            )

            entity = Entity(
                id=entity_id,
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                group_id=group_id,
                entity_type=entity_type,
                canonical_name=canonical,
                aliases=[item["name"]],
                source_messages=[message_id],
                created_at=request.timestamp,  # Fix: Use message timestamp, not now()
                last_mentioned=request.timestamp,  # Fix: Use message timestamp, not now()
            )

            # Open Question #1: Vector-based entity resolution
            # Generate embedding and check for potential duplicates
            if self._embedding_service:
                try:
                    # Embed canonical name + aliases for similarity search
                    text_to_embed = f"{canonical} {' '.join([item['name']])}"
                    embedding = await self._embedding_service.embed(text_to_embed)
                    entity.embedding = embedding
                except Exception as e:
                    logger.debug("Entity embedding generation failed: %s", e)

            # Upsert entity once (after embedding is attached if available)
            # This avoids double-appending source_messages when embeddings are enabled
            await self._store.upsert_entity(entity)
            entities.append(entity)

            if self._embedding_service and entity.embedding:
                try:
                    similar = await self._store.find_similar_entities(
                        embedding=entity.embedding,
                        entity_type=entity_type,
                        limit=3,
                        threshold=0.85,
                        exclude_id=entity.id,
                    )

                    if similar:
                        # Log warning for manual review (no auto-merge in MVP)
                        similar_names = [(e.canonical_name, score) for e, score in similar]
                        logger.warning(
                            "Potential duplicate entity detected: '%s' "
                            "similar to %s. Consider manual merge via POST /entities/%s/merge",
                            entity.canonical_name,
                            similar_names,
                            entity.id,
                        )
                except Exception as e:
                    logger.debug("Entity similarity check failed: %s", e)

        return entities, raw_items

    # ── Stage 2: Relationship Inference ──────────────────────────────────

    async def _infer_relationships(
        self,
        request: IngestRequest,
        entities: list[Entity],
        raw_entity_items: list[dict[str, Any]],
        message_id: str,
    ) -> list[Relationship]:
        """Infer relationships between extracted entities via LLM.

        Integrates:
        - Open Question #2: Unknown type validation → relates_to fallback
        - Open Question #4: Confidence snapping to discrete levels
        """
        # Build entity context for the prompt (use original LLM names)
        entity_dicts = [{"name": item["name"], "type": item["type"]} for item in raw_entity_items]

        # Fetch existing active relationships for context
        existing_rels = await self._get_existing_relationships_context(entities)

        prompt = build_relationship_inference_prompt(
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            entities=entity_dicts,
            existing_relationships=existing_rels,
        )

        result = await self._llm.complete_json(prompt)
        group_id = request.group_id or request.conversation_id

        relationships: list[Relationship] = []
        for item in result.get("relationships", []):
            # Map LLM mention strings to Entity objects
            source = self._find_entity_by_mention(item.get("source_mention", ""), entities)
            target = self._find_entity_by_mention(item.get("target_mention", ""), entities)

            if source is None or target is None:
                logger.warning(
                    "Could not resolve relationship mentions: source='%s' target='%s' — skipping",
                    item.get("source_mention"),
                    item.get("target_mention"),
                )
                continue

            # Open Question #2: Validate relationship type against enum
            metadata: dict[str, Any] = {}
            try:
                rel_type = RelationshipType(item["type"])
            except ValueError:
                logger.warning(
                    "Unknown relationship type '%s', using 'relates_to'",
                    item["type"],
                )
                rel_type = RelationshipType.RELATES_TO
                metadata["original_type"] = item["type"]

            # Open Question #4: Snap confidence to discrete levels
            confidence = snap_confidence(item.get("confidence", 0.8))

            relationship = Relationship(
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                group_id=group_id,
                message_id=message_id,
                source_id=source.id,
                target_id=target.id,
                rel_type=rel_type,
                confidence=confidence,
                evidence=item.get("evidence", ""),
                valid_from=request.timestamp,
                metadata=metadata,
            )
            relationships.append(relationship)

        return relationships

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _find_entity_by_mention(self, mention: str, entities: list[Entity]) -> Entity | None:
        """Map an LLM mention string to an extracted Entity.

        Matches by normalizing the mention and comparing to canonical names
        and aliases of the extracted entities.
        """
        if not mention:
            return None

        normalized = Entity.normalize_name(mention)

        # First pass: match against canonical name
        for entity in entities:
            if entity.canonical_name == normalized:
                return entity

        # Second pass: match against aliases
        for entity in entities:
            for alias in entity.aliases:
                if Entity.normalize_name(alias) == normalized:
                    return entity

        return None

    async def _get_context_entities(self, request: IngestRequest) -> list[dict[str, str]]:
        """Fetch recently mentioned entities in this conversation for LLM context."""
        since = request.timestamp - timedelta(days=_CONTEXT_LOOKBACK_DAYS)
        recent = await self._store.get_recent_entities(
            tenant_id=request.tenant_id,
            conversation_id=request.conversation_id,
            since=since,
            limit=20,
        )
        return [{"name": e.canonical_name, "type": e.entity_type.value} for e in recent]

    async def _get_existing_relationships_context(
        self, entities: list[Entity]
    ) -> list[dict[str, Any]]:
        """Fetch existing active relationships involving these entities for LLM context."""
        existing: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for entity in entities:
            rels = await self._store.get_active_relationships(entity.id)
            for r in rels:
                key = (r.source_id, r.target_id, r.rel_type)
                if key not in seen:
                    seen.add(key)
                    existing.append(
                        {
                            "source": r.source_id.split(":")[-1],
                            "target": r.target_id.split(":")[-1],
                            "type": r.rel_type.value
                            if isinstance(r.rel_type, RelationshipType)
                            else str(r.rel_type),
                            "confidence": r.confidence,
                        }
                    )

        return existing
