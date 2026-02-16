"""Integration test fixtures for Neo4j.

Requires: docker-compose up neo4j -d
"""

from __future__ import annotations

import pytest

from engram.config import Settings
from engram.storage.neo4j import Neo4jStore


@pytest.fixture(scope="session")
def neo4j_settings():
    """Integration tests use local Docker Neo4j."""
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="sk-test",
        _env_file=None,
    )


@pytest.fixture
async def neo4j_store(neo4j_settings):
    """Provide a clean Neo4jStore for each test."""
    store = Neo4jStore(neo4j_settings)
    await store.initialize()
    yield store
    # Clean up test data
    await store._execute_write("MATCH (n) DETACH DELETE n")
    await store.close()
