"""FastAPI application with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from engram.api.routes import router
from engram.config import Settings
from engram.llm.provider import LLMProvider
from engram.services.dedup import InMemoryDedup, RedisDedup
from engram.services.extraction import ExtractionPipeline
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore
from engram.storage.neo4j import Neo4jStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down services on app startup/shutdown."""
    settings: Settings = app.state.settings
    use_memory_store: bool = app.state.use_memory_store

    # Initialize storage
    if use_memory_store:
        logger.info("Using in-memory store for testing")
        store = MemoryStore()
    else:
        logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}")
        store = Neo4jStore(settings)

    await store.initialize()
    app.state.store = store

    # Initialize deduplication service
    if use_memory_store or not settings.redis_enabled:
        logger.info("Using in-memory deduplication")
        dedup = InMemoryDedup()
    else:
        logger.info(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}")
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                password=settings.redis_password,
                decode_responses=True,
            )
            dedup = RedisDedup(redis_client)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory.")
            dedup = InMemoryDedup()

    app.state.dedup = dedup

    # Initialize LLM provider
    logger.info(f"Initializing LLM provider: {settings.llm_model}")
    llm = LLMProvider(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key or None,
    )

    # Initialize conflict resolver
    resolver = ConflictResolver(store)

    # Initialize extraction pipeline
    pipeline = ExtractionPipeline(
        store=store,
        dedup=dedup,
        resolver=resolver,
        llm=llm,
        embedding_service=None,  # Optional for MVP
    )
    app.state.pipeline = pipeline

    logger.info("Engram services initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down Engram services")
    await store.close()
    if hasattr(dedup, "close"):
        await dedup.close()
    logger.info("Engram services shut down")


def create_app(use_memory_store: bool = False) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        use_memory_store: If True, use in-memory store for testing.
                         If False, use Neo4j (production).

    Returns:
        Configured FastAPI application.
    """
    settings = Settings(_env_file=None) if use_memory_store else Settings()

    app = FastAPI(
        title="Engram",
        description="Temporal knowledge graph engine for AI memory",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store configuration in app state for dependency injection
    app.state.settings = settings
    app.state.use_memory_store = use_memory_store

    # Include API routes
    app.include_router(router)

    return app


# Create default app instance for uvicorn
app = create_app()
