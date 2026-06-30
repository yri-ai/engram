"""Tests for forecast lifecycle repository persistence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.forecasting import ForecastQuestion, ForecastResolution, ForecastRun
from engram.services.forecast_repository import ForecastRepository
from engram.storage.memory import MemoryStore


@pytest.fixture()
async def store() -> MemoryStore:
    store = MemoryStore()
    await store.initialize()
    yield store
    await store.close()


def _make_deal_entity() -> Entity:
    canonical_name = Entity.normalize_name("Sterling Town Center")
    return Entity(
        id=Entity.build_id("tenant-1", EntityType.CONCEPT, canonical_name, group_id="forecasting"),
        tenant_id="tenant-1",
        conversation_id="forecasting",
        group_id="forecasting",
        entity_type=EntityType.CONCEPT,
        canonical_name=canonical_name,
    )


def _make_question() -> ForecastQuestion:
    return ForecastQuestion(
        id="fq-1",
        tenant_id="tenant-1",
        target_entity_id=_make_deal_entity().id,
        objective="Predict the next deal branch",
        structural_family="real_estate_acquisition",
        forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
        horizon="30d",
        resolution_due_at=datetime(2026, 6, 1, tzinfo=UTC),
        resolution_criteria="Resolve when the deal closes, reprices, or terminates",
        allowed_branch_names=["advance_diligence", "reprice_or_restructure", "terminated_failed"],
    )


def _make_run() -> ForecastRun:
    return ForecastRun(
        id="fr-1",
        question_id="fq-1",
        model_or_engine="branch-forecaster-v1",
        forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
        branch_probabilities={
            "advance_diligence": 0.8,
            "reprice_or_restructure": 0.2,
        },
        top_branch="advance_diligence",
        selected_evidence_ids=["fact-1", "fact-2"],
        rationale="Operating statement and rent roll support continued diligence.",
    )


def _make_resolution() -> ForecastResolution:
    return ForecastResolution(
        question_id="fq-1",
        run_id="fr-1",
        resolved_at=datetime(2026, 6, 15, tzinfo=UTC),
        outcome_branch="reprice_or_restructure",
        resolution_notes="Seller accepted revised price after diligence.",
        resolved_by="analyst@example.com",
        source="ic_memo_2026_06_15",
    )


async def test_repository_saves_and_lists_forecast_questions(store: MemoryStore) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )

    question = _make_question()
    saved = await repository.save_question(question)
    questions = await repository.list_questions(target_entity_id=deal.id)

    assert saved.id == question.id
    assert [item.id for item in questions] == [question.id]
    assert questions[0].allowed_branch_names == question.allowed_branch_names


async def test_repository_saves_and_lists_forecast_runs(store: MemoryStore) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )

    await repository.save_question(_make_question())
    run = _make_run()
    saved = await repository.save_run(target_entity_id=deal.id, run=run)
    runs = await repository.list_runs(target_entity_id=deal.id, question_id="fq-1")

    assert saved.id == run.id
    assert [item.id for item in runs] == [run.id]
    assert runs[0].top_branch == "advance_diligence"


async def test_repository_saves_and_gets_resolution(store: MemoryStore) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )

    await repository.save_question(_make_question())
    resolution = _make_resolution()

    saved = await repository.save_resolution(target_entity_id=deal.id, resolution=resolution)
    loaded = await repository.get_resolution(target_entity_id=deal.id, question_id="fq-1")

    assert saved.question_id == resolution.question_id
    assert loaded is not None
    assert loaded.run_id == "fr-1"
    assert loaded.outcome_branch == "reprice_or_restructure"


async def test_repository_preserves_layer1_run_artifacts(store: MemoryStore) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )

    await repository.save_question(_make_question())
    run = _make_run().model_copy(
        update={
            "metadata": {
                "branch_forecast": {
                    "selected_context": [
                        {"id": "fact-1", "source": "/tmp/rent-roll.xlsx"},
                        {"id": "fact-2", "source": "/tmp/t12.xlsx"},
                    ]
                }
            }
        }
    )

    await repository.save_run(target_entity_id=deal.id, run=run)
    runs = await repository.list_runs(target_entity_id=deal.id, question_id="fq-1")

    assert len(runs) == 1
    assert runs[0].forecast_as_of == datetime(2026, 5, 1, tzinfo=UTC)
    assert runs[0].selected_evidence_ids == ["fact-1", "fact-2"]
    assert (
        runs[0].metadata["branch_forecast"]["selected_context"][0]["source"]
        == "/tmp/rent-roll.xlsx"
    )


async def test_repository_reuses_deterministic_resolution_record_id(store: MemoryStore) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )

    await repository.save_question(_make_question())
    resolution = _make_resolution()

    await repository.save_resolution(target_entity_id=deal.id, resolution=resolution)
    await repository.save_resolution(target_entity_id=deal.id, resolution=resolution)

    facts = await store.get_facts(
        "tenant-1", deal.id, fact_key=ForecastRepository.RESOLUTION_FACT_KEY
    )
    assert len(facts) == 1
