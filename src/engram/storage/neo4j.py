"""Neo4j storage implementation.

Production GraphStore using Neo4j Community Edition with:
- Full Cypher queries for entity/relationship CRUD
- Bitemporal relationship versioning (valid_from/to, recorded_from/to)
- Vector index for entity embeddings (cosine similarity)
- group_id scoping for cross-conversation entity linking (Open Question #3)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.storage.base import GraphStore

if TYPE_CHECKING:
    from engram.config import Settings

logger = logging.getLogger(__name__)


class Neo4jStore(GraphStore):
    """Neo4j-backed graph storage with bitemporal support.

    Uses AsyncGraphDatabase for non-blocking operations.
    All temporal fields stored as ISO 8601 strings for portability.
    """

    def __init__(self, settings: Settings) -> None:
        self._driver: AsyncDriver | None = None
        self._settings = settings

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Connect to Neo4j and create schema indexes."""
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )
        await self._create_indexes()

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            await self._execute_read("RETURN 1 AS ok")
            return True
        except Exception:
            return False

    async def _create_indexes(self) -> None:
        """Create all indexes from ARCHITECTURE.md section 1.3.

        Includes:
        - Entity property indexes (id, type, tenant, conversation, group_id)
        - Relationship temporal indexes (valid_from/to, recorded_from/to)
        - Vector index for entity embeddings (1536 dims, cosine similarity)
        """
        indexes = [
            # Entity indexes
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
            "CREATE INDEX entity_conversation IF NOT EXISTS FOR (e:Entity) ON (e.conversation_id)",
            "CREATE INDEX entity_group IF NOT EXISTS FOR (e:Entity) ON (e.group_id)",
            "CREATE INDEX entity_canonical_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
        ]

        for idx in indexes:
            try:
                await self._execute_write(idx)
            except Exception as e:
                # Log but don't fail — indexes may already exist
                logger.debug("Index creation note: %s", e)

        # Vector index for entity embeddings — separate try/catch as
        # this may fail on older Neo4j versions or if already exists
        try:
            await self._execute_write(
                "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) "
                "OPTIONS {indexConfig: {"
                "`vector.dimensions`: $dimensions, "
                "`vector.similarity_function`: 'cosine'"
                "}}",
                dimensions=self._settings.embedding_dimensions,
            )
        except Exception as e:
            logger.warning("Vector index creation failed (may already exist): %s", e)

    # --- Internal Helpers ---

    async def _execute_read(self, query: str, **params: Any) -> list[dict]:
        """Execute a read query and return list of record dicts."""
        assert self._driver is not None, "Neo4jStore not initialized"
        async with self._driver.session() as session:
            result = await session.run(query, params)
            return [dict(record) async for record in result]

    async def _execute_write(self, query: str, **params: Any) -> list[dict]:
        """Execute a write query and return list of record dicts."""
        assert self._driver is not None, "Neo4jStore not initialized"
        async with self._driver.session() as session:
            result = await session.run(query, params)
            return [dict(record) async for record in result]

    def _node_to_entity(self, node: Any) -> Entity:
        """Convert Neo4j node properties to Entity model."""
        props = dict(node)
        aliases = props.get("aliases", [])
        if isinstance(aliases, str):
            aliases = json.loads(aliases)
        source_messages = props.get("source_messages", [])
        if isinstance(source_messages, str):
            source_messages = json.loads(source_messages)
        metadata = props.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Entity(
            id=props["id"],
            tenant_id=props["tenant_id"],
            conversation_id=props.get("conversation_id", ""),
            group_id=props.get("group_id"),
            entity_type=EntityType(props["type"]),
            canonical_name=props["canonical_name"],
            aliases=aliases,
            embedding=props.get("embedding"),
            created_at=datetime.fromisoformat(props["created_at"]),
            last_mentioned=datetime.fromisoformat(props["last_mentioned"]),
            source_messages=source_messages,
            metadata=metadata,
        )

    def _edge_to_relationship(self, edge: Any, source_id: str, target_id: str) -> Relationship:
        """Convert Neo4j relationship properties to Relationship model."""
        props = dict(edge)
        metadata = props.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Relationship(
            tenant_id=props["tenant_id"],
            conversation_id=props["conversation_id"],
            group_id=props.get("group_id"),
            message_id=props["message_id"],
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType(props["type"]),
            confidence=props["confidence"],
            evidence=props.get("evidence", ""),
            valid_from=datetime.fromisoformat(props["valid_from"]),
            valid_to=(datetime.fromisoformat(props["valid_to"]) if props.get("valid_to") else None),
            recorded_from=datetime.fromisoformat(props["recorded_from"]),
            recorded_to=(
                datetime.fromisoformat(props["recorded_to"]) if props.get("recorded_to") else None
            ),
            version=props.get("version", 1),
            supersedes=props.get("supersedes"),
            metadata=metadata,
        )

    # --- Entity Operations ---

    async def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update entity. Idempotent via MERGE on entity.id.

        ON CREATE: Sets all fields.
        ON MATCH: Updates last_mentioned, merges aliases and source_messages.
        """
        await self._execute_write(
            """
            MERGE (e:Entity {id: $id})
            ON CREATE SET
                e.tenant_id = $tenant_id,
                e.conversation_id = $conversation_id,
                e.group_id = $group_id,
                e.type = $type,
                e.canonical_name = $canonical_name,
                e.aliases = $aliases,
                e.embedding = $embedding,
                e.created_at = $created_at,
                e.last_mentioned = $last_mentioned,
                e.source_messages = $source_messages,
                e.metadata = $metadata
            ON MATCH SET
                e.last_mentioned = $last_mentioned,
                e.aliases = [a IN e.aliases WHERE NOT a IN $aliases] + $aliases,
                e.source_messages = e.source_messages + $source_messages,
                e.group_id = COALESCE($group_id, e.group_id)
            RETURN e
            """,
            id=entity.id,
            tenant_id=entity.tenant_id,
            conversation_id=entity.conversation_id,
            group_id=entity.group_id,
            type=entity.entity_type.value,
            canonical_name=entity.canonical_name,
            aliases=entity.aliases,
            embedding=entity.embedding,
            created_at=entity.created_at.isoformat(),
            last_mentioned=entity.last_mentioned.isoformat(),
            source_messages=entity.source_messages,
            metadata=json.dumps(entity.metadata),
        )
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        results = await self._execute_read(
            """
            MATCH (e:Entity {id: $id})
            RETURN e
            """,
            id=entity_id,
        )
        if not results:
            return None
        return self._node_to_entity(results[0]["e"])

    async def get_entity_by_name(
        self, tenant_id: str, conversation_id: str, canonical_name: str
    ) -> Entity | None:
        """Find entity by canonical name within a conversation."""
        results = await self._execute_read(
            """
            MATCH (e:Entity)
            WHERE e.tenant_id = $tenant_id
              AND e.conversation_id = $conversation_id
              AND e.canonical_name = $canonical_name
            RETURN e
            LIMIT 1
            """,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            canonical_name=canonical_name,
        )
        if not results:
            return None
        return self._node_to_entity(results[0]["e"])

    async def list_entities(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filters, pagination."""
        query = "MATCH (e:Entity) WHERE e.tenant_id = $tenant_id"
        params: dict[str, Any] = {"tenant_id": tenant_id, "limit": limit, "offset": offset}

        if conversation_id:
            query += " AND e.conversation_id = $conversation_id"
            params["conversation_id"] = conversation_id
        if entity_type:
            query += " AND e.type = $type"
            params["type"] = entity_type.value

        query += " RETURN e ORDER BY e.created_at DESC SKIP $offset LIMIT $limit"
        results = await self._execute_read(query, **params)
        return [self._node_to_entity(r["e"]) for r in results]

    # --- Relationship Operations ---

    async def create_relationship(self, rel: Relationship) -> Relationship:
        """Create a new bitemporal relationship between entities.

        Full bitemporal fields: valid_from/to (truth timeline), recorded_from/to (knowledge timeline).
        Uses MERGE on entities to ensure they exist, then CREATE for the relationship.
        """
        await self._execute_write(
            """
            MERGE (a:Entity {id: $source_id})
            MERGE (b:Entity {id: $target_id})
            CREATE (a)-[r:RELATIONSHIP {
                tenant_id: $tenant_id,
                conversation_id: $conversation_id,
                group_id: $group_id,
                message_id: $message_id,
                type: $type,
                valid_from: $valid_from,
                valid_to: $valid_to,
                recorded_from: $recorded_from,
                recorded_to: $recorded_to,
                confidence: $confidence,
                evidence: $evidence,
                version: $version,
                supersedes: $supersedes,
                metadata: $metadata
            }]->(b)
            """,
            source_id=rel.source_id,
            target_id=rel.target_id,
            tenant_id=rel.tenant_id,
            conversation_id=rel.conversation_id,
            group_id=rel.group_id,
            message_id=rel.message_id,
            type=rel.rel_type.value,
            valid_from=rel.valid_from.isoformat(),
            valid_to=rel.valid_to.isoformat() if rel.valid_to else None,
            recorded_from=rel.recorded_from.isoformat(),
            recorded_to=rel.recorded_to.isoformat() if rel.recorded_to else None,
            confidence=rel.confidence,
            evidence=rel.evidence,
            version=rel.version,
            supersedes=rel.supersedes,
            metadata=json.dumps(rel.metadata),
        )
        return rel

    async def get_active_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        tenant_id: str | None = None,
    ) -> list[Relationship]:
        """Get currently active relationships (valid_to IS NULL AND recorded_to IS NULL)."""
        query = """
            MATCH (e:Entity {id: $entity_id})-[r:RELATIONSHIP]->(target:Entity)
            WHERE r.valid_to IS NULL
              AND r.recorded_to IS NULL
        """
        params: dict[str, Any] = {"entity_id": entity_id}

        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value
        if tenant_id:
            query += " AND r.tenant_id = $tenant_id"
            params["tenant_id"] = tenant_id

        query += (
            " RETURN r, e.id AS source_id, target.id AS target_id ORDER BY r.recorded_from DESC"
        )
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["source_id"], r["target_id"]) for r in results]

    async def terminate_relationship(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        group_id: str,
        termination_time: datetime,
        exclude_target_id: str | None = None,
    ) -> int:
        """Terminate active relationships matching criteria. Returns count terminated.

        Sets both valid_to and recorded_to to the termination time (bitemporal close).
        """
        query = """
            MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP]->(target:Entity)
            WHERE r.type = $type
              AND r.tenant_id = $tenant_id
              AND r.group_id = $group_id
              AND r.valid_to IS NULL
              AND r.recorded_to IS NULL
        """
        params: dict[str, Any] = {
            "source_id": source_id,
            "type": rel_type.value,
            "tenant_id": tenant_id,
            "group_id": group_id,
            "termination_time": termination_time.isoformat(),
        }

        if exclude_target_id:
            query += " AND target.id <> $exclude_target_id"
            params["exclude_target_id"] = exclude_target_id

        query += """
            SET r.valid_to = $termination_time,
                r.recorded_to = $termination_time
            RETURN count(r) AS count
        """
        result = await self._execute_write(query, **params)
        return result[0]["count"] if result else 0

    async def get_max_relationship_version(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        group_id: str,
    ) -> int:
        query = """
            MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP]->()
            WHERE r.type = $type
              AND r.tenant_id = $tenant_id
              AND r.group_id = $group_id
            RETURN COALESCE(MAX(r.version), 0) AS max_version
        """
        result = await self._execute_read(
            query,
            source_id=source_id,
            type=rel_type.value,
            tenant_id=tenant_id,
            group_id=group_id,
        )
        return result[0]["max_version"] if result else 0

    # --- Entity Merge ---

    async def merge_entity_into(self, primary_id: str, duplicate_id: str) -> int:
        # Redirect outgoing relationships: (duplicate)->[:R]->(target) → (primary)->[:R]->(target)
        outgoing = await self._execute_write(
            """
            MATCH (dup:Entity {id: $dup_id})-[r:RELATIONSHIP]->(target:Entity)
            MATCH (primary:Entity {id: $primary_id})
            CREATE (primary)-[r2:RELATIONSHIP]->(target)
            SET r2 = properties(r)
            DELETE r
            RETURN count(r2) AS cnt
            """,
            dup_id=duplicate_id,
            primary_id=primary_id,
        )
        out_count = outgoing[0]["cnt"] if outgoing else 0

        # Redirect incoming relationships: (source)->[:R]->(duplicate) → (source)->[:R]->(primary)
        incoming = await self._execute_write(
            """
            MATCH (source:Entity)-[r:RELATIONSHIP]->(dup:Entity {id: $dup_id})
            MATCH (primary:Entity {id: $primary_id})
            CREATE (source)-[r2:RELATIONSHIP]->(primary)
            SET r2 = properties(r)
            DELETE r
            RETURN count(r2) AS cnt
            """,
            dup_id=duplicate_id,
            primary_id=primary_id,
        )
        in_count = incoming[0]["cnt"] if incoming else 0

        # Merge aliases, source_messages; mark duplicate
        await self._execute_write(
            """
            MATCH (primary:Entity {id: $primary_id}), (dup:Entity {id: $dup_id})
            SET primary.aliases = [a IN primary.aliases WHERE NOT a IN dup.aliases] + dup.aliases
                + CASE WHEN NOT dup.canonical_name IN primary.aliases
                       THEN [dup.canonical_name] ELSE [] END,
                primary.source_messages = primary.source_messages + dup.source_messages,
                dup.metadata = '{"merged_into": "' + $primary_id + '"}'
            """,
            primary_id=primary_id,
            dup_id=duplicate_id,
        )

        return out_count + in_count

    # --- Temporal Queries ---

    async def query_world_state_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What was ACTUALLY TRUE at a point in time (valid time).

        Filter: valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of)
        """
        query = """
            MATCH (e:Entity)-[r:RELATIONSHIP]->(target:Entity)
            WHERE e.tenant_id = $tenant_id
              AND toLower(e.canonical_name) = $entity_name
              AND $as_of >= r.valid_from
              AND (r.valid_to IS NULL OR $as_of < r.valid_to)
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "entity_name": entity_name.lower(),
            "as_of": as_of.isoformat(),
        }

        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value

        query += " RETURN e.id AS source_id, r, target.id AS target_id ORDER BY r.confidence DESC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["source_id"], r["target_id"]) for r in results]

    async def query_knowledge_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What did we BELIEVE at a point in time (record time).

        Filter: recorded_from <= as_of AND (recorded_to IS NULL OR recorded_to > as_of)
        """
        query = """
            MATCH (e:Entity)-[r:RELATIONSHIP]->(target:Entity)
            WHERE e.tenant_id = $tenant_id
              AND toLower(e.canonical_name) = $entity_name
              AND $as_of >= r.recorded_from
              AND (r.recorded_to IS NULL OR $as_of < r.recorded_to)
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "entity_name": entity_name.lower(),
            "as_of": as_of.isoformat(),
        }

        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value

        query += " RETURN e.id AS source_id, r, target.id AS target_id ORDER BY r.confidence DESC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["source_id"], r["target_id"]) for r in results]

    async def query_evolution(
        self,
        tenant_id: str,
        entity_name: str,
        target_name: str | None = None,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """Get all versions of relationships for timeline view (no temporal filter)."""
        query = """
            MATCH (e:Entity)-[r:RELATIONSHIP]->(target:Entity)
            WHERE e.tenant_id = $tenant_id
              AND toLower(e.canonical_name) = $entity_name
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "entity_name": entity_name.lower(),
        }

        if target_name:
            query += " AND toLower(target.canonical_name) = $target_name"
            params["target_name"] = target_name.lower()
        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value

        query += " RETURN e.id AS source_id, r, target.id AS target_id ORDER BY r.valid_from ASC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["source_id"], r["target_id"]) for r in results]

    async def get_recent_entities(
        self,
        tenant_id: str,
        conversation_id: str,
        since: datetime,
        limit: int = 20,
    ) -> list[Entity]:
        """Get recently mentioned entities, ordered by recency."""
        results = await self._execute_read(
            """
            MATCH (e:Entity)
            WHERE e.tenant_id = $tenant_id
              AND e.conversation_id = $conversation_id
              AND e.last_mentioned >= $since
            RETURN e
            ORDER BY e.last_mentioned DESC
            LIMIT $limit
            """,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            since=since.isoformat(),
            limit=limit,
        )
        return [self._node_to_entity(r["e"]) for r in results]

    # --- Vector Search ---

    async def find_similar_entities(
        self,
        embedding: list[float],
        entity_type: EntityType,
        limit: int = 5,
        threshold: float = 0.85,
        exclude_id: str | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find similar entities using vector index (cosine similarity).

        Requires entity_embedding vector index to be created.
        Returns list of (entity, similarity_score) tuples.
        Excludes entity with exclude_id if provided (to avoid self-matching).
        """
        try:
            where_clauses = ["node.type = $type", "score > $threshold"]
            if exclude_id:
                where_clauses.append("node.id <> $exclude_id")
            where_clause = " AND ".join(where_clauses)

            results = await self._execute_read(
                f"""
                CALL db.index.vector.queryNodes('entity_embedding', $limit, $embedding)
                YIELD node, score
                WHERE {where_clause}
                RETURN node, score
                ORDER BY score DESC
                """,
                embedding=embedding,
                limit=limit,
                type=entity_type.value,
                threshold=threshold,
                exclude_id=exclude_id,
            )
            return [(self._node_to_entity(r["node"]), r["score"]) for r in results]
        except Exception as e:
            logger.warning("Vector search failed (index may not exist): %s", e)
            return []
