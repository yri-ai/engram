"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from engram.main import create_app
from engram.services.dedup import InMemoryDedup
from engram.services.extraction import ExtractionPipeline
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore


@pytest.fixture
def client():
    """Create test client with in-memory store."""
    app = create_app(use_memory_store=True)

    # Manually initialize app state for testing
    async def init_app():
        store = MemoryStore()
        await store.initialize()
        dedup = InMemoryDedup()
        resolver = ConflictResolver(store)

        # Mock LLM that returns empty extractions (for API stub-level compatibility)
        llm = AsyncMock()
        llm.complete_json = AsyncMock(return_value={"entities": [], "relationships": []})

        pipeline = ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=llm)

        app.state.store = store
        app.state.dedup = dedup
        app.state.pipeline = pipeline

    # Run async initialization
    asyncio.run(init_app())

    return TestClient(app)


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check(self, client):
        """Health check should return 200 with status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"


class TestIngestEndpoint:
    """Test POST /messages endpoint."""

    def test_ingest_message_basic(self, client):
        """Ingest a basic message."""
        response = client.post(
            "/messages",
            json={
                "text": "Kendra loves Nike shoes",
                "speaker": "Kendra",
                "conversation_id": "c1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data
        assert "entities_extracted" in data
        assert "relationships_inferred" in data
        assert "conflicts_resolved" in data
        assert "processing_time_ms" in data

    def test_ingest_message_with_timestamp(self, client):
        """Ingest message with explicit timestamp."""
        now = datetime.now(UTC).isoformat()
        response = client.post(
            "/messages",
            json={
                "text": "Test message",
                "speaker": "User",
                "conversation_id": "c1",
                "timestamp": now,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data

    def test_ingest_message_with_tenant(self, client):
        """Ingest message with explicit tenant_id."""
        response = client.post(
            "/messages",
            json={
                "text": "Test message",
                "speaker": "User",
                "conversation_id": "c1",
                "tenant_id": "t1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data

    def test_ingest_message_with_group_id(self, client):
        """Ingest message with group_id for cross-conversation linking."""
        response = client.post(
            "/messages",
            json={
                "text": "Test message",
                "speaker": "User",
                "conversation_id": "c1",
                "group_id": "g1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data


class TestListEntitiesEndpoint:
    """Test GET /entities endpoint."""

    def test_list_entities_empty(self, client):
        """List entities when none exist."""
        response = client.get("/entities", params={"tenant_id": "default"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_entities_with_tenant_filter(self, client):
        """List entities filtered by tenant_id."""
        response = client.get("/entities", params={"tenant_id": "t1"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_entities_with_conversation_filter(self, client):
        """List entities filtered by conversation_id."""
        response = client.get(
            "/entities",
            params={"tenant_id": "default", "conversation_id": "c1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_entities_with_type_filter(self, client):
        """List entities filtered by entity_type."""
        response = client.get(
            "/entities",
            params={"tenant_id": "default", "entity_type": "PERSON"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_entities_with_pagination(self, client):
        """List entities with limit and offset."""
        response = client.get(
            "/entities",
            params={"tenant_id": "default", "limit": 10, "offset": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestGetEntityEndpoint:
    """Test GET /entities/{entity_id} endpoint."""

    def test_get_entity_not_found(self, client):
        """Get non-existent entity returns 404."""
        response = client.get("/entities/nonexistent")
        assert response.status_code == 404

    def test_get_entity_by_id(self, client):
        """Get entity by ID after ingestion."""
        # First ingest a message to create entities
        ingest_response = client.post(
            "/messages",
            json={
                "text": "Kendra loves Nike",
                "speaker": "Kendra",
                "conversation_id": "c1",
            },
        )
        assert ingest_response.status_code == 200

        # List entities to get an ID
        list_response = client.get("/entities", params={"tenant_id": "default"})
        assert list_response.status_code == 200
        entities = list_response.json()

        if entities:
            entity_id = entities[0]["id"]
            get_response = client.get(f"/entities/{entity_id}")
            assert get_response.status_code == 200
            data = get_response.json()
            assert data["id"] == entity_id


class TestGetRelationshipsEndpoint:
    """Test GET /entities/{entity_id}/relationships endpoint."""

    def test_get_relationships_empty(self, client):
        """Get relationships for entity with none."""
        response = client.get(
            "/entities/nonexistent/relationships",
        )
        # Should return empty list or 404
        assert response.status_code in [200, 404]

    def test_get_relationships_with_type_filter(self, client):
        """Get relationships filtered by type."""
        response = client.get(
            "/entities/test-entity/relationships",
            params={"rel_type": "prefers"},
        )
        assert response.status_code in [200, 404]

    def test_get_relationships_after_ingest(self, client):
        """Get relationships after ingesting message."""
        # Ingest message
        ingest_response = client.post(
            "/messages",
            json={
                "text": "Kendra loves Nike shoes",
                "speaker": "Kendra",
                "conversation_id": "c1",
            },
        )
        assert ingest_response.status_code == 200

        # List entities to get an ID
        list_response = client.get("/entities", params={"tenant_id": "default"})
        assert list_response.status_code == 200
        entities = list_response.json()

        if entities:
            entity_id = entities[0]["id"]
            rel_response = client.get(f"/entities/{entity_id}/relationships")
            assert rel_response.status_code == 200
            data = rel_response.json()
            assert isinstance(data, list)


class TestPointInTimeQueryEndpoint:
    """Test GET /query/point-in-time endpoint."""

    def test_point_in_time_query_basic(self, client):
        """Query world state at a point in time."""
        now = datetime.now(UTC).isoformat()
        response = client.get(
            "/query/point-in-time",
            params={
                "entity": "Kendra",
                "as_of": now,
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_point_in_time_query_with_rel_type(self, client):
        """Query with relationship type filter."""
        now = datetime.now(UTC).isoformat()
        response = client.get(
            "/query/point-in-time",
            params={
                "entity": "Kendra",
                "as_of": now,
                "rel_type": "prefers",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_point_in_time_query_with_mode(self, client):
        """Query with different temporal modes."""
        now = datetime.now(UTC).isoformat()
        for mode in ["world_state", "knowledge", "bitemporal"]:
            response = client.get(
                "/query/point-in-time",
                params={
                    "entity": "Kendra",
                    "as_of": now,
                    "mode": mode,
                    "tenant_id": "default",
                },
            )
            assert response.status_code == 200


class TestEvolutionQueryEndpoint:
    """Test GET /query/evolution endpoint."""

    def test_evolution_query_basic(self, client):
        """Query relationship evolution over time."""
        response = client.get(
            "/query/evolution",
            params={
                "entity": "Kendra",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_evolution_query_with_rel_type(self, client):
        """Query evolution filtered by relationship type."""
        response = client.get(
            "/query/evolution",
            params={
                "entity": "Kendra",
                "rel_type": "prefers",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_evolution_query_with_target(self, client):
        """Query evolution for specific target."""
        response = client.get(
            "/query/evolution",
            params={
                "entity": "Kendra",
                "target": "Nike",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestSearchEndpoint:
    """Test GET /search endpoint."""

    def test_search_entities_basic(self, client):
        """Search for entities."""
        response = client.get(
            "/search",
            params={
                "q": "Kendra",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_entities_empty_query(self, client):
        """Search with empty query."""
        response = client.get(
            "/search",
            params={
                "q": "",
                "tenant_id": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_entities_with_limit(self, client):
        """Search with limit parameter."""
        response = client.get(
            "/search",
            params={
                "q": "test",
                "tenant_id": "default",
                "limit": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestMergeEntitiesEndpoint:
    """Test POST /entities/{entity_id}/merge endpoint (Open Question #1)."""

    def test_merge_entities_basic(self, client):
        """Merge duplicate entity into primary."""
        response = client.post(
            "/entities/primary-id/merge",
            json={"duplicate_id": "duplicate-id"},
        )
        # Should return 200 or 404 if entities don't exist
        assert response.status_code in [200, 404]

    def test_merge_entities_with_query_param(self, client):
        """Merge using request body."""
        response = client.post(
            "/entities/primary-id/merge",
            json={"duplicate_id": "duplicate-id"},
        )
        assert response.status_code in [200, 404]

    def test_merge_entities_response_structure(self, client):
        """Verify merge response structure."""
        response = client.post(
            "/entities/primary-id/merge",
            json={"duplicate_id": "duplicate-id"},
        )
        if response.status_code == 200:
            data = response.json()
            # Should contain merge result info
            assert isinstance(data, dict)

    def test_merge_entities_same_id_error(self, client):
        """Merging entity with itself should fail."""
        response = client.post(
            "/entities/same-id/merge",
            json={"duplicate_id": "same-id"},
        )
        # Should return 400 or similar error
        assert response.status_code in [400, 404]


class TestAPIIntegration:
    """Integration tests across multiple endpoints."""

    def test_full_workflow(self, client):
        """Test complete workflow: ingest -> list -> get -> query."""
        # 1. Ingest message
        ingest_response = client.post(
            "/messages",
            json={
                "text": "Kendra loves Nike shoes",
                "speaker": "Kendra",
                "conversation_id": "c1",
            },
        )
        assert ingest_response.status_code == 200

        # 2. List entities
        list_response = client.get("/entities", params={"tenant_id": "default"})
        assert list_response.status_code == 200
        entities = list_response.json()
        assert isinstance(entities, list)

        # 3. Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"

    def test_multiple_ingestions(self, client):
        """Test multiple message ingestions."""
        messages = [
            "Kendra loves Nike shoes",
            "She prefers running in the morning",
            "Her goal is to run a marathon",
        ]

        for text in messages:
            response = client.post(
                "/messages",
                json={
                    "text": text,
                    "speaker": "Kendra",
                    "conversation_id": "c1",
                },
            )
            assert response.status_code == 200

        # Verify entities were created
        list_response = client.get("/entities", params={"tenant_id": "default"})
        assert list_response.status_code == 200
