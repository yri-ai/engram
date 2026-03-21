"""E2E test: Ingest coaching demo -> verify temporal queries.

This test validates the FULL pipeline from message ingestion to temporal queries.
Uses MemoryStore (no Neo4j needed) with mocked LLM.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engram.models.entity import EntityType
from engram.models.message import IngestRequest
from engram.models.relationship import RelationshipType
from engram.services.dedup import InMemoryDedup
from engram.services.extraction import ExtractionPipeline
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore

# ── Fixtures ─────────────────────────────────────────────────────────────


_EMPTY_TAIL = [
    {"facts": []},
    {"commitments": []},
    {"opening_state": "", "key_shift": None, "closing_state": "", "breakthrough": False},
]
"""Stages 4-6 (fact extraction, commitment extraction, summary) return empty for e2e mocks."""


def _msg(entities: dict, relationships: dict) -> list[dict]:
    """Build a full 5-response sequence for one message (entity, rel, fact, commitment, summary)."""
    return [entities, relationships, *_EMPTY_TAIL]


def _build_mock_llm_responses() -> list[dict]:
    """Build deterministic LLM responses for each coaching-demo message.

    coaching-demo.json has 6 messages. Each message triggers 5 LLM calls:
    entity extraction, relationship inference, fact extraction, commitment
    extraction, and conversation summary (30 total responses).
    """
    return [
        # ── Message 1: "Hi, I'm Kendra. I've been training for the Boston Marathon."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Boston Marathon",
                        "canonical": "boston-marathon",
                        "type": "EVENT",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Boston Marathon",
                        "type": "has_goal",
                        "confidence": 0.9,
                        "evidence": "training for the Boston Marathon",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
        # ── Message 2: "I love Nike running shoes, they're the best for long distance."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Nike running shoes",
                        "canonical": "nike-running-shoes",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Nike running shoes",
                        "type": "prefers",
                        "confidence": 0.95,
                        "evidence": "loves Nike running shoes, best for long distance",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
        # ── Message 3: "My goal is to finish the Boston Marathon under 4 hours."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Boston Marathon",
                        "canonical": "boston-marathon",
                        "type": "EVENT",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Boston Marathon",
                        "type": "has_goal",
                        "confidence": 1.0,
                        "evidence": "goal to finish under 4 hours",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
        # ── Message 4: "I've been working with Coach Sarah on my training plan."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Coach Sarah",
                        "canonical": "coach-sarah",
                        "type": "PERSON",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Coach Sarah",
                        "type": "knows",
                        "confidence": 0.9,
                        "evidence": "working with Coach Sarah on training plan",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
        # ── Message 5: "Actually, I switched to Adidas. The arch support is much better."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Adidas",
                        "canonical": "adidas",
                        "type": "PREFERENCE",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Adidas",
                        "type": "prefers",
                        "confidence": 0.9,
                        "evidence": "switched to Adidas, better arch support",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
        # ── Message 6: "I completed the Boston Marathon! Finished in 3:45:00."
        *_msg(
            {
                "entities": [
                    {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
                    {
                        "name": "Boston Marathon",
                        "canonical": "boston-marathon",
                        "type": "EVENT",
                        "confidence": 1.0,
                    },
                ]
            },
            {
                "relationships": [
                    {
                        "source_mention": "Kendra",
                        "target_mention": "Boston Marathon",
                        "type": "discussed",
                        "confidence": 1.0,
                        "evidence": "completed the Boston Marathon in 3:45:00",
                        "temporal_marker": "now",
                    },
                ]
            },
        ),
    ]


@pytest.fixture
async def pipeline_and_store():
    """Set up full pipeline with MemoryStore and mocked LLM."""
    store = MemoryStore()
    await store.initialize()
    dedup = InMemoryDedup()
    resolver = ConflictResolver(store)
    llm = AsyncMock()

    # Program LLM responses for all 6 coaching-demo messages
    llm.complete_json = AsyncMock(side_effect=_build_mock_llm_responses())

    pipeline = ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=llm)
    return pipeline, store


def _load_coaching_demo() -> list[IngestRequest]:
    """Load coaching-demo.json and convert to IngestRequest objects."""
    demo_path = Path(__file__).parent.parent.parent / "examples" / "coaching-demo.json"
    with open(demo_path) as f:
        data = json.load(f)

    conversation_id = data["conversation_id"]
    requests = []
    for i, msg in enumerate(data["messages"], start=1):
        requests.append(
            IngestRequest(
                text=msg["text"],
                speaker=msg["speaker"],
                timestamp=datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")),
                conversation_id=conversation_id,
                tenant_id="t1",
                message_id=f"msg-{i}",
            )
        )
    return requests


# ── Test: Full Pipeline Ingestion ────────────────────────────────────────


class TestCoachingDemoIngestion:
    """Verify that all 6 messages from coaching-demo.json can be ingested."""

    async def test_ingest_all_messages(self, pipeline_and_store):
        """All 6 coaching-demo messages should ingest successfully."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        assert len(messages) == 6

        results = []
        for msg in messages:
            result = await pipeline.process_message(msg)
            results.append(result)

        # Every message should produce a non-zero processing time
        for result in results:
            assert result.processing_time_ms >= 0

        # Messages 1-6 should extract entities
        assert results[0].entities_extracted == 2  # Kendra, Boston Marathon
        assert results[1].entities_extracted == 2  # Kendra, Nike running shoes
        assert results[2].entities_extracted == 2  # Kendra, Boston Marathon
        assert results[3].entities_extracted == 2  # Kendra, Coach Sarah
        assert results[4].entities_extracted == 2  # Kendra, Adidas
        assert results[5].entities_extracted == 2  # Kendra, Boston Marathon

    async def test_duplicate_message_rejected(self, pipeline_and_store):
        """Duplicate message IDs should be rejected (idempotency)."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        # Ingest first message
        first_result = await pipeline.process_message(messages[0])
        assert first_result.entities_extracted == 2

        # Re-ingest same message ID → should be deduped
        dup_result = await pipeline.process_message(messages[0])
        assert dup_result.entities_extracted == 0
        assert dup_result.relationships_inferred == 0


# ── Test: Entity Verification After Each Message ─────────────────────────


class TestEntityVerification:
    """Verify correct entities exist after ingesting each message."""

    async def test_entities_after_message_1(self, pipeline_and_store):
        """After message 1: Kendra and Boston Marathon should exist."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        await pipeline.process_message(messages[0])

        entities = await store.list_entities("t1")
        canonical_names = {e.canonical_name for e in entities}
        assert "kendra" in canonical_names
        assert "boston-marathon" in canonical_names

    async def test_entities_after_message_2(self, pipeline_and_store):
        """After message 2: Nike running shoes should also exist."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages[:2]:
            await pipeline.process_message(msg)

        entities = await store.list_entities("t1")
        canonical_names = {e.canonical_name for e in entities}
        assert "kendra" in canonical_names
        assert "boston-marathon" in canonical_names
        assert "nike-running-shoes" in canonical_names

    async def test_entities_after_all_messages(self, pipeline_and_store):
        """After all messages: all entities should exist."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages:
            await pipeline.process_message(msg)

        entities = await store.list_entities("t1")
        canonical_names = {e.canonical_name for e in entities}

        expected = {"kendra", "boston-marathon", "nike-running-shoes", "coach-sarah", "adidas"}
        assert expected.issubset(canonical_names)

    async def test_entity_types_correct(self, pipeline_and_store):
        """Entities should have correct types."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages:
            await pipeline.process_message(msg)

        entities = await store.list_entities("t1")
        type_map = {e.canonical_name: e.entity_type for e in entities}

        assert type_map["kendra"] == EntityType.PERSON
        assert type_map["boston-marathon"] == EntityType.EVENT
        assert type_map["nike-running-shoes"] == EntityType.PREFERENCE
        assert type_map["coach-sarah"] == EntityType.PERSON
        assert type_map["adidas"] == EntityType.PREFERENCE


# ── Test: Relationship Verification ──────────────────────────────────────


class TestRelationshipVerification:
    """Verify relationships are created correctly after each message."""

    async def test_prefers_nike_after_message_2(self, pipeline_and_store):
        """After message 2: Kendra prefers Nike running shoes."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages[:2]:
            await pipeline.process_message(msg)

        # Find Kendra's entity ID
        kendra = await store.get_entity_by_name("t1", "coaching-session-1", "kendra")
        assert kendra is not None

        # Check active prefers relationships
        active_prefs = await store.get_active_relationships(
            kendra.id, rel_type=RelationshipType.PREFERS
        )
        assert len(active_prefs) == 1
        assert "nike-running-shoes" in active_prefs[0].target_id

    async def test_has_goal_boston_marathon(self, pipeline_and_store):
        """After message 1: Kendra has_goal Boston Marathon."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        await pipeline.process_message(messages[0])

        kendra = await store.get_entity_by_name("t1", "coaching-session-1", "kendra")
        assert kendra is not None

        goals = await store.get_active_relationships(kendra.id, rel_type=RelationshipType.HAS_GOAL)
        assert len(goals) == 1
        assert "boston-marathon" in goals[0].target_id

    async def test_knows_coach_sarah(self, pipeline_and_store):
        """After message 4: Kendra knows Coach Sarah."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages[:4]:
            await pipeline.process_message(msg)

        kendra = await store.get_entity_by_name("t1", "coaching-session-1", "kendra")
        assert kendra is not None

        knows_rels = await store.get_active_relationships(
            kendra.id, rel_type=RelationshipType.KNOWS
        )
        assert len(knows_rels) == 1
        assert "coach-sarah" in knows_rels[0].target_id


# ── Test: Correction Detection (Nike → Adidas) ──────────────────────────


class TestCorrectionDetection:
    """Verify that message 5 (Adidas switch) correctly terminates Nike preference."""

    async def test_nike_terminated_adidas_active(self, pipeline_and_store):
        """After message 5: Nike preference terminated, Adidas active."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        # Ingest messages 1-5 (includes the correction)
        for msg in messages[:5]:
            await pipeline.process_message(msg)

        kendra = await store.get_entity_by_name("t1", "coaching-session-1", "kendra")
        assert kendra is not None

        # Active prefers should be Adidas only
        active_prefs = await store.get_active_relationships(
            kendra.id, rel_type=RelationshipType.PREFERS
        )
        assert len(active_prefs) == 1
        assert "adidas" in active_prefs[0].target_id

    async def test_nike_preference_has_valid_to(self, pipeline_and_store):
        """Nike preference should have valid_to set (terminated)."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages[:5]:
            await pipeline.process_message(msg)

        # Query evolution to see all versions
        evolution = await store.query_evolution("t1", "kendra", rel_type=RelationshipType.PREFERS)

        # Should have 2 entries: Nike (terminated) + Adidas (active)
        assert len(evolution) == 2

        # Find Nike relationship
        nike_rels = [r for r in evolution if "nike" in r.target_id]
        assert len(nike_rels) == 1
        assert nike_rels[0].valid_to is not None  # Terminated
        assert nike_rels[0].is_active is False

        # Find Adidas relationship
        adidas_rels = [r for r in evolution if "adidas" in r.target_id]
        assert len(adidas_rels) == 1
        assert adidas_rels[0].valid_to is None  # Active
        assert adidas_rels[0].is_active is True

    async def test_adidas_version_incremented(self, pipeline_and_store):
        """Adidas relationship should have version > 1 (supersedes Nike)."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages[:5]:
            await pipeline.process_message(msg)

        kendra = await store.get_entity_by_name("t1", "coaching-session-1", "kendra")
        assert kendra is not None

        active_prefs = await store.get_active_relationships(
            kendra.id, rel_type=RelationshipType.PREFERS
        )
        assert len(active_prefs) == 1
        # Version should be > 1 since it superseded Nike
        assert active_prefs[0].version == 2


# ── Test: Temporal Queries (As-Of Behavior) ──────────────────────────────


class TestTemporalQueries:
    """Verify point-in-time queries return correct historical state."""

    async def _ingest_all(self, pipeline, messages):
        """Helper to ingest all messages."""
        for msg in messages:
            await pipeline.process_message(msg)

    async def test_world_state_week_2_returns_nike(self, pipeline_and_store):
        """Temporal query at week 2 (Feb 15): should return Nike preference."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        await self._ingest_all(pipeline, messages)

        # Week 2: Feb 15, 2024 — Nike was active (valid_from Jan 15, valid_to Mar 20)
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        results = await store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1
        assert "nike" in results[0].target_id

    async def test_world_state_week_4_returns_adidas(self, pipeline_and_store):
        """Temporal query at week 4 (Apr 15): should return Adidas preference."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        await self._ingest_all(pipeline, messages)

        # Week 4: Apr 15, 2024 — Adidas is active (valid_from Mar 20, valid_to NULL)
        w4 = datetime(2024, 4, 15, tzinfo=UTC)
        results = await store.query_world_state_as_of(
            "t1", "kendra", w4, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 1
        assert "adidas" in results[0].target_id

    async def test_knowledge_as_of_before_recording(self, pipeline_and_store):
        """Knowledge query before any recording: we knew nothing yet.

        recorded_from is always datetime.now() (when pipeline runs), so
        querying at historical dates (before pipeline ran) returns nothing.
        This is correct: we didn't KNOW anything at that point in time.
        """
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        await self._ingest_all(pipeline, messages)

        # Jan 1 2024 is before any message was recorded
        before = datetime(2024, 1, 1, tzinfo=UTC)
        results = await store.query_knowledge_as_of(
            "t1", "kendra", before, rel_type=RelationshipType.PREFERS
        )
        assert len(results) == 0

    async def test_knowledge_as_of_after_recording(self, pipeline_and_store):
        """Knowledge query after recording: we believe Adidas is preferred.

        recorded_from is datetime.now() (when pipeline ran). Querying AFTER
        that time should return what we currently believe: Adidas (active),
        NOT Nike (terminated, recorded_to set).
        """
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        await self._ingest_all(pipeline, messages)

        # Far future: well after all recording happened
        future = datetime(2027, 1, 1, tzinfo=UTC)
        results = await store.query_knowledge_as_of(
            "t1", "kendra", future, rel_type=RelationshipType.PREFERS
        )
        # Only Adidas should be believed (Nike was terminated → recorded_to set)
        assert len(results) == 1
        assert "adidas" in results[0].target_id

    async def test_world_state_before_first_message(self, pipeline_and_store):
        """Query before any messages: should return nothing."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()
        await self._ingest_all(pipeline, messages)

        before = datetime(2024, 1, 1, tzinfo=UTC)
        results = await store.query_world_state_as_of("t1", "kendra", before)
        assert len(results) == 0


# ── Test: Evolution Timeline ─────────────────────────────────────────────


class TestEvolutionTimeline:
    """Verify relationship evolution shows complete history."""

    async def test_preference_evolution_nike_to_adidas(self, pipeline_and_store):
        """Evolution query should show Nike → Adidas preference transition."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages:
            await pipeline.process_message(msg)

        evolution = await store.query_evolution("t1", "kendra", rel_type=RelationshipType.PREFERS)

        # Should have exactly 2 preference relationships
        assert len(evolution) == 2

        # Sort by valid_from for deterministic ordering
        evolution.sort(key=lambda r: r.valid_from)

        # First: Nike (terminated)
        assert "nike" in evolution[0].target_id
        assert evolution[0].valid_to is not None
        assert evolution[0].is_active is False

        # Second: Adidas (active)
        assert "adidas" in evolution[1].target_id
        assert evolution[1].valid_to is None
        assert evolution[1].is_active is True

    async def test_full_evolution_all_types(self, pipeline_and_store):
        """Evolution should include all relationship types."""
        pipeline, store = pipeline_and_store
        messages = _load_coaching_demo()

        for msg in messages:
            await pipeline.process_message(msg)

        all_evolution = await store.query_evolution("t1", "kendra")

        # Should have multiple relationship types
        rel_types = {r.rel_type for r in all_evolution}
        assert RelationshipType.PREFERS in rel_types
        assert RelationshipType.HAS_GOAL in rel_types


# ── Test: Full Coaching Demo Scenario (Integration) ──────────────────────


class TestFullCoachingScenario:
    """End-to-end test of the complete coaching demo scenario."""

    async def test_coaching_demo_preference_evolution(self, pipeline_and_store):
        """Kendra's shoe preference changes from Nike to Adidas.

        This is the canonical E2E test from the plan — validates full pipeline
        from ingestion through temporal queries.
        """
        pipeline, store = pipeline_and_store

        # Ingest key messages (subset matching plan's test)
        messages = [
            IngestRequest(
                text="Hi, I'm Kendra. Training for Boston Marathon.",
                speaker="Kendra",
                conversation_id="c1",
                tenant_id="t1",
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                message_id="msg-1",
            ),
            IngestRequest(
                text="I love Nike running shoes",
                speaker="Kendra",
                conversation_id="c1",
                tenant_id="t1",
                timestamp=datetime(2024, 1, 15, 10, 5, tzinfo=UTC),
                message_id="msg-2",
            ),
            IngestRequest(
                text="Switched to Adidas, better arch support",
                speaker="Kendra",
                conversation_id="c1",
                tenant_id="t1",
                timestamp=datetime(2024, 3, 20, 14, 30, tzinfo=UTC),
                message_id="msg-5",
            ),
        ]

        # Need a separate pipeline with correctly-sized mock responses
        llm = AsyncMock()
        llm.complete_json = AsyncMock(
            side_effect=[
                *_msg(
                    {
                        "entities": [
                            {
                                "name": "Kendra",
                                "canonical": "kendra",
                                "type": "PERSON",
                                "confidence": 1.0,
                            },
                            {
                                "name": "Boston Marathon",
                                "canonical": "boston-marathon",
                                "type": "EVENT",
                                "confidence": 1.0,
                            },
                        ]
                    },
                    {
                        "relationships": [
                            {
                                "source_mention": "Kendra",
                                "target_mention": "Boston Marathon",
                                "type": "has_goal",
                                "confidence": 0.9,
                                "evidence": "training for",
                                "temporal_marker": "now",
                            },
                        ]
                    },
                ),
                *_msg(
                    {
                        "entities": [
                            {
                                "name": "Kendra",
                                "canonical": "kendra",
                                "type": "PERSON",
                                "confidence": 1.0,
                            },
                            {
                                "name": "Nike running shoes",
                                "canonical": "nike-running-shoes",
                                "type": "PREFERENCE",
                                "confidence": 1.0,
                            },
                        ]
                    },
                    {
                        "relationships": [
                            {
                                "source_mention": "Kendra",
                                "target_mention": "Nike running shoes",
                                "type": "prefers",
                                "confidence": 0.95,
                                "evidence": "loves Nike",
                                "temporal_marker": "now",
                            },
                        ]
                    },
                ),
                *_msg(
                    {
                        "entities": [
                            {
                                "name": "Kendra",
                                "canonical": "kendra",
                                "type": "PERSON",
                                "confidence": 1.0,
                            },
                            {
                                "name": "Adidas",
                                "canonical": "adidas",
                                "type": "PREFERENCE",
                                "confidence": 1.0,
                            },
                        ]
                    },
                    {
                        "relationships": [
                            {
                                "source_mention": "Kendra",
                                "target_mention": "Adidas",
                                "type": "prefers",
                                "confidence": 0.9,
                                "evidence": "switched to Adidas",
                                "temporal_marker": "now",
                            },
                        ]
                    },
                ),
            ]
        )

        dedup = InMemoryDedup()
        resolver = ConflictResolver(store)
        test_pipeline = ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=llm)

        for msg in messages:
            await test_pipeline.process_message(msg)

        # QUERY 1: Current preference = Adidas
        kendra = await store.get_entity_by_name("t1", "c1", "kendra")
        assert kendra is not None
        active_prefs = await store.get_active_relationships(
            kendra.id, rel_type=RelationshipType.PREFERS
        )
        assert len(active_prefs) == 1
        assert "adidas" in active_prefs[0].target_id

        # QUERY 2: Week 2 preference = Nike (world state as of Feb 15)
        w2 = datetime(2024, 2, 15, tzinfo=UTC)
        historical = await store.query_world_state_as_of(
            "t1", "kendra", w2, rel_type=RelationshipType.PREFERS
        )
        assert len(historical) == 1
        assert "nike" in historical[0].target_id

        # QUERY 3: Evolution shows Nike -> Adidas
        evolution = await store.query_evolution("t1", "kendra", rel_type=RelationshipType.PREFERS)
        assert len(evolution) == 2  # Nike (terminated) + Adidas (active)
