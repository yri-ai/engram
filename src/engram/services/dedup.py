"""Message deduplication service.

Production: Redis SET NX (atomic, with TTL)
Development: In-memory set (not production-safe)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DedupService(ABC):
    """Abstract deduplication interface."""

    @abstractmethod
    async def check_and_mark(self, message_id: str) -> bool:
        """Check if message is new and mark as processed.

        Returns True if message is NEW (first time seen).
        Returns False if message is a DUPLICATE.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class InMemoryDedup(DedupService):
    """In-memory deduplication (for testing and dev without Redis).

    WARNING: Not production-safe. No persistence, no TTL, naive eviction.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._processed: set[str] = set()
        self._max_size = max_size

    async def check_and_mark(self, message_id: str) -> bool:
        if message_id in self._processed:
            return False
        if len(self._processed) >= self._max_size:
            # Naive eviction: clear half
            to_keep = list(self._processed)[self._max_size // 2 :]
            self._processed = set(to_keep)
            logger.warning("InMemoryDedup evicted old entries (not production-safe)")
        self._processed.add(message_id)
        return True

    async def close(self) -> None:
        self._processed.clear()


class RedisDedup(DedupService):
    """Redis-based deduplication (production).

    Uses SET NX with TTL for atomic idempotency checks.
    Default TTL is 7 days (604800 seconds).
    """

    def __init__(self, redis_client: object, ttl_seconds: int = 604800) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def check_and_mark(self, message_id: str) -> bool:
        key = f"engram:processed:{message_id}"
        is_new = await self._redis.set(key, "1", nx=True, ex=self._ttl)  # type: ignore[attr-defined]
        return is_new is not None  # Redis returns None if key already exists

    async def close(self) -> None:
        pass  # Redis client lifecycle managed externally
