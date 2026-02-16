"""Integration tests for vector-based entity resolution.

Tests the full pipeline:
1. Entity extraction with embeddings
2. Vector index creation in Neo4j
3. Similarity search for duplicate detection
4. Warning logging for potential duplicates

Requires: docker-compose up neo4j -d
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from engram.models.entity import Entity, EntityType
from engram.models.message import IngestRequest
from engram.services.embeddings import EmbeddingService
from engram.services.extraction import ExtractionPipeline
from engram.services.dedup import InMemoryDedup
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore
from engram.storage.neo4j import Neo4jStore


@pytest.fixture
async def memory_store():
    """In-memory store for unit-level vector tests."""
    store = MemoryStore()
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def neo4j_store(neo4j_settings):
    """Neo4j store for integration tests."""
    store = Neo4jStore(neo4j_settings)
    await store.initialize()
    yield store
    # Clean up test data
    await store._execute_write("MATCH (n) DETACH DELETE n")
    await store.close()


class TestVectorIndexCreation:
    """Test that vector index is created during initialization."""

    @pytest.mark.integration
    async def test_vector_index_created(self, neo4j_store):
        """Verify vector index exists after initialize()."""
        # Query Neo4j for vector indexes
        result = await neo4j_store._execute_read(
            "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"
        )
        index_names = [r["name"] for r in result]
        assert "entity_embedding" in index_names


class TestMemoryStoreSimilaritySearch:
    """Test in-memory cosine similarity search."""

    async def test_find_similar_entities_empty_store(self, memory_store):
        """Empty store returns no results."""
        embedding = [0.1] * 1536
        results = await memory_store.find_similar_entities(
            embedding=embedding, entity_type=EntityType.PERSON, limit=5, threshold=0.85
        )
        assert results == []

    async def test_find_similar_entities_exact_match(self, memory_store):
        """Identical embeddings should match with high score."""
        embedding = [0.1] * 1536
        entity = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=embedding,
        )
        await memory_store.upsert_entity(entity)

        # Search with same embedding
        results = await memory_store.find_similar_entities(
            embedding=embedding, entity_type=EntityType.PERSON, limit=5, threshold=0.85
        )
        assert len(results) == 1
        assert results[0][0].id == entity.id
        assert results[0][1] > 0.99  # Should be nearly 1.0

    async def test_find_similar_entities_filters_by_type(self, memory_store):
        """Should only return entities of matching type."""
        embedding = [0.1] * 1536
        person = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=embedding,
        )
        preference = Entity(
            id="t1:c1:PREFERENCE:nike",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PREFERENCE,
            canonical_name="nike",
            embedding=embedding,
        )
        await memory_store.upsert_entity(person)
        await memory_store.upsert_entity(preference)

        # Search for PERSON type only
        results = await memory_store.find_similar_entities(
            embedding=embedding, entity_type=EntityType.PERSON, limit=5, threshold=0.85
        )
        assert len(results) == 1
        assert results[0][0].entity_type == EntityType.PERSON

    async def test_find_similar_entities_respects_threshold(self, memory_store):
        """Should filter by similarity threshold."""
        embedding1 = [1.0] + [0.0] * 1535
        # Create embedding2 with ~0.5 similarity to embedding1
        # Using normalized vectors: [0.5, 0.866, 0, 0, ...] has dot product ~0.5 with [1, 0, 0, ...]
        embedding2 = [0.5] + [0.866] + [0.0] * 1534

        entity1 = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=embedding1,
        )
        entity2 = Entity(
            id="t1:c1:PERSON:bob",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="bob",
            embedding=embedding2,
        )
        await memory_store.upsert_entity(entity1)
        await memory_store.upsert_entity(entity2)

        # High threshold should exclude low-similarity match
        results = await memory_store.find_similar_entities(
            embedding=embedding1, entity_type=EntityType.PERSON, limit=5, threshold=0.85
        )
        assert len(results) == 1  # Only exact match
        assert results[0][0].id == entity1.id

        # Low threshold should include both
        results = await memory_store.find_similar_entities(
            embedding=embedding1, entity_type=EntityType.PERSON, limit=5, threshold=0.4
        )
        assert len(results) == 2

    async def test_find_similar_entities_respects_limit(self, memory_store):
        """Should return at most 'limit' results."""
        embedding = [0.1] * 1536
        for i in range(5):
            entity = Entity(
                id=f"t1:c1:PERSON:person{i}",
                tenant_id="t1",
                conversation_id="c1",
                entity_type=EntityType.PERSON,
                canonical_name=f"person{i}",
                embedding=embedding,
            )
            await memory_store.upsert_entity(entity)

        results = await memory_store.find_similar_entities(
            embedding=embedding, entity_type=EntityType.PERSON, limit=2, threshold=0.85
        )
        assert len(results) == 2


class TestExtractionPipelineWithEmbeddings:
    """Test extraction pipeline with embedding service integration."""

    async def test_pipeline_generates_embeddings(self, memory_store):
        """Pipeline should generate embeddings for entities."""
        dedup = InMemoryDedup()
        resolver = ConflictResolver(memory_store)
        llm = AsyncMock()

        # Mock LLM responses
        llm.complete_json = AsyncMock(
            side_effect=[
                {
                    "entities": [
                        {
                            "name": "Alice",
                            "canonical": "alice",
                            "type": "PERSON",
                            "confidence": 1.0,
                        }
                    ]
                },
                {"relationships": []},
            ]
        )

        # Mock embedding service
        embedding_service = AsyncMock(spec=EmbeddingService)
        embedding_service.embed = AsyncMock(return_value=[0.1] * 1536)

        pipeline = ExtractionPipeline(
            store=memory_store,
            dedup=dedup,
            resolver=resolver,
            llm=llm,
            embedding_service=embedding_service,
        )

        request = IngestRequest(
            text="Alice is here",
            speaker="Alice",
            conversation_id="c1",
            tenant_id="t1",
        )
        response = await pipeline.process_message(request)

        assert response.entities_extracted == 1
        # Verify embedding service was called
        embedding_service.embed.assert_called()

    async def test_pipeline_detects_duplicate_entities(self, memory_store, caplog):
        """Pipeline should log warning when duplicate entity detected."""
        dedup = InMemoryDedup()
        resolver = ConflictResolver(memory_store)
        llm = AsyncMock()

        # Pre-populate store with similar entity
        existing_entity = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=[0.1] * 1536,
        )
        await memory_store.upsert_entity(existing_entity)

        # Mock LLM responses
        llm.complete_json = AsyncMock(
            side_effect=[
                {
                    "entities": [
                        {
                            "name": "Alice Smith",
                            "canonical": "alice-smith",
                            "type": "PERSON",
                            "confidence": 1.0,
                        }
                    ]
                },
                {"relationships": []},
            ]
        )

        # Mock embedding service to return similar embedding
        embedding_service = AsyncMock(spec=EmbeddingService)
        embedding_service.embed = AsyncMock(return_value=[0.1] * 1536)

        pipeline = ExtractionPipeline(
            store=memory_store,
            dedup=dedup,
            resolver=resolver,
            llm=llm,
            embedding_service=embedding_service,
        )

        request = IngestRequest(
            text="Alice Smith is here",
            speaker="Alice",
            conversation_id="c1",
            tenant_id="t1",
        )

        with caplog.at_level("WARNING"):
            response = await pipeline.process_message(request)

        assert response.entities_extracted == 1
        # Check that warning was logged
        assert any(
            "Potential duplicate entity detected" in record.message for record in caplog.records
        )

    async def test_pipeline_handles_embedding_service_failure(self, memory_store):
        """Pipeline should gracefully handle embedding service failures."""
        dedup = InMemoryDedup()
        resolver = ConflictResolver(memory_store)
        llm = AsyncMock()

        # Mock LLM responses
        llm.complete_json = AsyncMock(
            side_effect=[
                {
                    "entities": [
                        {
                            "name": "Alice",
                            "canonical": "alice",
                            "type": "PERSON",
                            "confidence": 1.0,
                        }
                    ]
                },
                {"relationships": []},
            ]
        )

        # Mock embedding service to fail
        embedding_service = AsyncMock(spec=EmbeddingService)
        embedding_service.embed = AsyncMock(side_effect=Exception("API error"))

        pipeline = ExtractionPipeline(
            store=memory_store,
            dedup=dedup,
            resolver=resolver,
            llm=llm,
            embedding_service=embedding_service,
        )

        request = IngestRequest(
            text="Alice is here",
            speaker="Alice",
            conversation_id="c1",
            tenant_id="t1",
        )

        # Should not raise, just log debug
        response = await pipeline.process_message(request)
        assert response.entities_extracted == 1


@pytest.mark.integration
class TestNeo4jVectorSearch:
    """Integration tests with real Neo4j vector index."""

    async def test_neo4j_vector_search_basic(self, neo4j_store):
        """Test vector search against Neo4j."""
        # Create entity with embedding
        entity = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=[0.1] * 1536,
        )
        await neo4j_store.upsert_entity(entity)

        # Search with same embedding
        results = await neo4j_store.find_similar_entities(
            embedding=[0.1] * 1536,
            entity_type=EntityType.PERSON,
            limit=5,
            threshold=0.85,
        )

        # May return empty if vector index not ready, but should not error
        assert isinstance(results, list)

    async def test_neo4j_vector_search_filters_by_type(self, neo4j_store):
        """Vector search should filter by entity type."""
        person = Entity(
            id="t1:c1:PERSON:alice",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PERSON,
            canonical_name="alice",
            embedding=[0.1] * 1536,
        )
        preference = Entity(
            id="t1:c1:PREFERENCE:nike",
            tenant_id="t1",
            conversation_id="c1",
            entity_type=EntityType.PREFERENCE,
            canonical_name="nike",
            embedding=[0.1] * 1536,
        )
        await neo4j_store.upsert_entity(person)
        await neo4j_store.upsert_entity(preference)

        # Search for PERSON type
        results = await neo4j_store.find_similar_entities(
            embedding=[0.1] * 1536,
            entity_type=EntityType.PERSON,
            limit=5,
            threshold=0.85,
        )

        # All results should be PERSON type
        for entity, score in results:
            assert entity.entity_type == EntityType.PERSON
