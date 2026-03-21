"""5-stage extraction pipeline: Entity -> Relationship -> Conflict Resolution -> Fact -> Commitment.

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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from engram.config import ConfigService, settings
from engram.models.commitment import Commitment, CommitmentStatus
from engram.models.entity import Entity, EntityType
from engram.models.fact import Fact
from engram.models.message import IngestRequest, IngestResponse
from engram.models.relationship import Evidence, Relationship, RelationshipType
from engram.models.run import ExtractionRun, RunStatus

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
    """Full 5-stage extraction pipeline.

    Stage 1: Entity extraction (LLM)
    Stage 2: Relationship inference (LLM)
    Stage 3: Conflict resolution (rule-based via ConflictResolver)
    Stage 4: Fact extraction (LLM)
    Stage 5: Commitment extraction (LLM)

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
        config_service: ConfigService | None = None,
    ) -> None:
        self._store = store
        self._dedup = dedup
        self._resolver = resolver
        self._llm = llm
        self._embedding_service = embedding_service
        self._config_service = config_service or ConfigService()

    async def process_message(self, request: IngestRequest) -> IngestResponse:
        """Process a conversation message through the full extraction pipeline.

        Returns an IngestResponse with counts of entities extracted,
        relationships inferred, and conflicts resolved.
        """
        start_time = time.monotonic()
        message_id = request.message_id or str(uuid.uuid4())

        # Step 0: Idempotency check
        is_new = await self._dedup.check_and_mark(message_id)
        if not is_new:
            return IngestResponse(
                message_id=message_id,
                entities_extracted=0,
                relationships_inferred=0,
                conflicts_resolved=0,
                processing_time_ms=0,
            )

        # Initialize Extraction Run
        run = ExtractionRun(
            id=str(uuid.uuid4()),
            tenant_id=request.tenant_id,
            conversation_id=request.conversation_id,
            message_id=message_id,
            prompt_id="entity_extraction.jinja2 + relationship_inference.jinja2",
            prompt_sha256=self._config_service.get_template_sha256("entity_extraction.jinja2"),
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            status=RunStatus.RUNNING,
        )
        await self._store.save_run(run)

        try:
            # Step 1: Entity extraction (LLM)
            entities, raw_entity_items = await self._extract_entities(request, message_id, run.id)

            # No entities → nothing to infer, skip Stage 2
            if not entities:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                run.status = RunStatus.COMPLETED
                run.completed_at = datetime.now(UTC)
                run.processing_time_ms = elapsed_ms
                await self._store.save_run(run)
                return IngestResponse(
                    message_id=message_id,
                    entities_extracted=0,
                    relationships_inferred=0,
                    conflicts_resolved=0,
                    processing_time_ms=round(elapsed_ms, 2),
                )

            # Step 2: Relationship inference (LLM)
            relationships = await self._infer_relationships(
                request, entities, raw_entity_items, message_id, run.id
            )

            # Step 3: Conflict resolution + graph writes
            conflicts = 0
            for rel in relationships:
                _, terminated = await self._resolver.resolve_and_create(rel)
                conflicts += terminated

            # Step 4: Fact extraction
            facts = await self._extract_facts(request, entities, message_id, run.id)

            # Step 5: Commitment Extraction
            await self._extract_commitments(request, entities, message_id, run.id)

            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(UTC)
        except Exception as e:
            run.status = RunStatus.FAILED
            run.error_text = str(e)
            run.completed_at = datetime.now(UTC)
            await self._store.save_run(run)
            await self._dedup.rollback(message_id)
            raise

        elapsed_ms = (time.monotonic() - start_time) * 1000
        run.processing_time_ms = elapsed_ms
        await self._store.save_run(run)
        return IngestResponse(
            message_id=message_id,
            entities_extracted=len(entities),
            relationships_inferred=len(relationships),
            conflicts_resolved=conflicts,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ── Stage 1: Entity Extraction ───────────────────────────────────────

    async def _extract_entities(
        self, request: IngestRequest, message_id: str, run_id: str
    ) -> tuple[list[Entity], list[dict[str, Any]]]:
        """Extract entities from message text via LLM.

        Returns:
            Tuple of (Entity objects upserted to store, raw LLM items for
            relationship inference context).
        """
        context = await self._get_extraction_context(request)

        prompt = self._config_service.render_prompt(
            "entity_extraction.jinja2",
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            context=context,
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
                extraction_run_id=run_id,
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
        run_id: str,
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

        prompt = self._config_service.render_prompt(
            "relationship_inference.jinja2",
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            entities=entity_dicts,
            existing_relationships=existing_rels,
        )

        result = await self._llm.complete_json(prompt)
        group_id = request.group_id or request.conversation_id

        # Expand mention resolution pool
        context_entities = await self._store.list_entities(
            tenant_id=request.tenant_id,
            conversation_id=request.conversation_id,
            limit=100,
        )
        current_ids = {e.id for e in entities}
        resolution_pool = entities + [e for e in context_entities if e.id not in current_ids]

        relationships: list[Relationship] = []
        for item in result.get("relationships", []):
            # Map LLM mention strings to Entity objects
            source = self._find_entity_by_mention(item.get("source_mention", ""), resolution_pool)
            target = self._find_entity_by_mention(item.get("target_mention", ""), resolution_pool)

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

            # Handle structured evidence
            structured_evidence = []
            if "structured_evidence" in item:
                se = item["structured_evidence"]
                structured_evidence.append(
                    Evidence(
                        message_id=message_id,
                        text=se.get("text", ""),
                        context=se.get("context"),
                        observed_at=request.timestamp,
                    )
                )

            relationship = Relationship(
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                group_id=group_id,
                message_id=message_id,
                extraction_run_id=run_id,
                source_id=source.id,
                target_id=target.id,
                rel_type=rel_type,
                confidence=confidence,
                evidence=item.get("evidence", ""),
                structured_evidence=structured_evidence,
                valid_from=request.timestamp,
                metadata=metadata,
            )
            relationships.append(relationship)

        return relationships

    # ── Stage 3.5: Fact Extraction ─────────────────────────────────────────

    async def _extract_facts(
        self,
        request: IngestRequest,
        entities: list[Entity],
        message_id: str,
        run_id: str,
    ) -> list[Fact]:
        """Extract standalone facts (knowledge claims about single entities) via LLM."""
        entity_dicts = [{"name": e.canonical_name, "type": e.entity_type.value} for e in entities]

        # Gather existing facts for each entity (for context / supersession)
        existing_facts: list[dict[str, Any]] = []
        for entity in entities:
            facts = await self._store.get_facts(request.tenant_id, entity.id)
            for f in facts:
                existing_facts.append(
                    {
                        "entity": entity.canonical_name,
                        "fact_key": f.fact_key,
                        "fact_text": f.fact_text,
                        "confidence": f.confidence,
                    }
                )

        prompt = self._config_service.render_prompt(
            "fact_extraction.jinja2",
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            entities=entity_dicts,
            existing_facts=existing_facts,
        )

        result = await self._llm.complete_json(prompt)
        raw_items: list[dict[str, Any]] = result.get("facts", [])

        created_facts: list[Fact] = []
        for i, item in enumerate(raw_items):
            # Resolve entity mention to Entity object
            entity = self._find_entity_by_mention(item.get("entity_mention", ""), entities)
            if not entity:
                logger.warning(
                    "Could not resolve fact entity mention '%s' — skipping",
                    item.get("entity_mention"),
                )
                continue

            fact = Fact(
                id=Fact.build_id(request.tenant_id, message_id, i),
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                message_id=message_id,
                extraction_run_id=run_id,
                entity_id=entity.id,
                fact_key=item["fact_key"],
                fact_text=item["fact_text"],
                confidence=snap_confidence(item.get("confidence", 0.8)),
                valid_from=request.timestamp,
            )

            # Handle supersession
            supersedes_key = item.get("supersedes_key")
            if supersedes_key:
                existing = await self._store.get_facts(
                    request.tenant_id, entity.id, fact_key=supersedes_key
                )
                if existing:
                    await self._store.supersede_fact(existing[0].id, fact)
                else:
                    await self._store.save_fact(fact)
            else:
                await self._store.save_fact(fact)

            created_facts.append(fact)

        return created_facts

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

    async def _get_extraction_context(self, request: IngestRequest) -> dict[str, Any]:
        """Build rich context for LLM extraction: recent entities + their facts + active relationships.

        Inspired by temporal-relationships' approach of giving the LLM full prior
        knowledge during extraction, not just entity names.
        """
        since = request.timestamp - timedelta(days=_CONTEXT_LOOKBACK_DAYS)
        recent_entities = await self._store.get_recent_entities(
            tenant_id=request.tenant_id,
            conversation_id=request.conversation_id,
            since=since,
            limit=20,
        )

        entity_context: list[dict[str, Any]] = []
        for e in recent_entities:
            entry: dict[str, Any] = {"name": e.canonical_name, "type": e.entity_type.value}

            # Attach known facts
            facts = await self._store.get_facts(request.tenant_id, e.id)
            if facts:
                entry["known_facts"] = [
                    {"key": f.fact_key, "text": f.fact_text, "confidence": f.confidence}
                    for f in facts[:5]  # Limit to avoid token bloat
                ]

            entity_context.append(entry)

        # Active relationships between recent entities
        rel_context: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for e in recent_entities:
            rels = await self._store.get_active_relationships(e.id)
            for r in rels:
                key = (r.source_id, r.target_id, r.rel_type)
                if key not in seen:
                    seen.add(key)
                    rel_context.append({
                        "source": r.source_id.split(":")[-1],
                        "target": r.target_id.split(":")[-1],
                        "type": r.rel_type.value if isinstance(r.rel_type, RelationshipType) else str(r.rel_type),
                        "evidence": r.evidence[:100] if r.evidence else "",
                    })

        return {
            "entities": entity_context,
            "relationships": rel_context[:20],  # Limit to avoid token bloat
        }

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

    async def _extract_commitments(
        self,
        request: IngestRequest,
        entities: list[Entity],
        message_id: str,
        run_id: str,
    ) -> list[Commitment]:
        """Extract commitments/intentions from message via LLM."""
        entity_dicts = [{"name": e.canonical_name, "type": e.entity_type.value} for e in entities]

        prompt = self._config_service.render_prompt(
            "commitment_extraction.jinja2",
            message_text=request.text,
            speaker=request.speaker,
            timestamp=request.timestamp.isoformat(),
            entities=entity_dicts,
        )

        result = await self._llm.complete_json(prompt)
        raw_items: list[dict[str, Any]] = result.get("commitments", [])

        commitments: list[Commitment] = []
        for i, item in enumerate(raw_items):
            # Resolve entity
            entity = self._find_entity_by_mention(item.get("entity_mention", ""), entities)
            if not entity:
                continue

            # Parse target_date if present
            target_date = None
            if item.get("target_date"):
                try:
                    target_date = datetime.fromisoformat(item["target_date"])
                except ValueError:
                    pass

            commitment = Commitment(
                id=Commitment.build_id(request.tenant_id, message_id, i),
                tenant_id=request.tenant_id,
                conversation_id=request.conversation_id,
                message_id=message_id,
                extraction_run_id=run_id,
                entity_id=entity.id,
                text=item["text"],
                status=CommitmentStatus.ACTIVE,
                created_at=request.timestamp,
                target_date=target_date,
                confidence=item.get("confidence", 0.8),
            )
            await self._store.save_commitment(commitment)
            commitments.append(commitment)

        return commitments
