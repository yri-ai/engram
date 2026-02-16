"""Tests for deduplication service."""

import asyncio

import pytest

from engram.services.dedup import InMemoryDedup


class TestInMemoryDedup:
    @pytest.fixture
    def dedup(self):
        return InMemoryDedup()

    async def test_first_message_is_new(self, dedup):
        assert await dedup.check_and_mark("msg-1") is True

    async def test_duplicate_message_is_not_new(self, dedup):
        await dedup.check_and_mark("msg-1")
        assert await dedup.check_and_mark("msg-1") is False

    async def test_different_messages_are_independent(self, dedup):
        assert await dedup.check_and_mark("msg-1") is True
        assert await dedup.check_and_mark("msg-2") is True
        assert await dedup.check_and_mark("msg-1") is False

    async def test_eviction_on_max_size(self):
        dedup = InMemoryDedup(max_size=2)
        await dedup.check_and_mark("msg-1")
        await dedup.check_and_mark("msg-2")
        await dedup.check_and_mark("msg-3")  # Triggers eviction
        # Recent message still tracked
        assert await dedup.check_and_mark("msg-3") is False

    async def test_close_clears_state(self, dedup):
        await dedup.check_and_mark("msg-1")
        await dedup.close()
        # After close, internal state is cleared — new instance would see msg-1 as new
        assert len(dedup._processed) == 0

    async def test_concurrent_access(self, dedup):
        """Multiple concurrent check_and_mark calls for the same ID.

        Only one should return True (new), all others should return False (duplicate).
        """
        results = await asyncio.gather(
            dedup.check_and_mark("msg-concurrent"),
            dedup.check_and_mark("msg-concurrent"),
            dedup.check_and_mark("msg-concurrent"),
            dedup.check_and_mark("msg-concurrent"),
            dedup.check_and_mark("msg-concurrent"),
        )
        # Exactly one True (first to mark), rest False
        assert results.count(True) == 1
        assert results.count(False) == 4


class TestRedisDedup:
    """Tests for RedisDedup using a fake Redis client."""

    async def test_first_message_is_new(self):
        """RedisDedup returns True for first-seen message ID."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis, ttl_seconds=604800)
        assert await dedup.check_and_mark("msg-1") is True

    async def test_duplicate_message_is_not_new(self):
        """RedisDedup returns False for already-seen message ID."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis, ttl_seconds=604800)
        await dedup.check_and_mark("msg-1")
        assert await dedup.check_and_mark("msg-1") is False

    async def test_key_format(self):
        """RedisDedup uses correct key prefix."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis, ttl_seconds=604800)
        await dedup.check_and_mark("msg-42")
        assert "engram:processed:msg-42" in fake_redis.store

    async def test_ttl_is_set(self):
        """RedisDedup passes configured TTL to Redis SET."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis, ttl_seconds=3600)
        await dedup.check_and_mark("msg-ttl")
        assert fake_redis.ttls["engram:processed:msg-ttl"] == 3600

    async def test_default_ttl_is_seven_days(self):
        """Default TTL should be 7 days (604800 seconds)."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis)
        assert dedup._ttl == 604800

    async def test_close_is_noop(self):
        """RedisDedup.close() should not raise (lifecycle managed externally)."""
        from engram.services.dedup import RedisDedup

        fake_redis = FakeRedis()
        dedup = RedisDedup(redis_client=fake_redis)
        await dedup.close()  # Should not raise


class FakeRedis:
    """Minimal fake async Redis client for unit testing RedisDedup."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None):
        """Mimics redis.set(key, value, nx=True, ex=ttl)."""
        if nx and key in self.store:
            return None  # Key already exists
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True
