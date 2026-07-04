"""Tests for forecast repositories."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.forecasting import (
    ForecastQuestion,
    ForecastQuestionType,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
    OutcomeBranch,
    QuestionStatus,
    ResolutionCriteria,
)
from engram.services.forecast_repository import ForecastRepository, JsonForecastRepository
from engram.storage.memory import MemoryStore

NOW = datetime(2026, 1, 15, tzinfo=UTC)
LATER = datetime(2026, 2, 15, tzinfo=UTC)


def test_save_load_and_list_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()

    repository.save_question(question)

    assert repository.load_question(question.id) == question
    assert repository.list_questions() == [question]
    assert (tmp_path / "questions" / f"{question.id}.json").exists()


def test_draft_question_can_be_updated_until_active(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question(title="Initial draft")
    repository.save_question(question)

    updated = question.model_copy(update={"title": "Updated draft"})
    repository.save_question(updated)
    assert repository.load_question(question.id).title == "Updated draft"

    active = updated.model_copy(update={"status": QuestionStatus.ACTIVE})
    repository.save_question(active)

    with pytest.raises(ValueError, match="active question changes require a new id"):
        repository.save_question(active.model_copy(update={"title": "Changed active question"}))

    assert repository.load_question(question.id).title == "Updated draft"
    assert repository.load_question(question.id).status == QuestionStatus.ACTIVE


def test_json_ledger_rejects_non_draft_question_without_branches(tmp_path):
    """Graph-shape questions (allowed_branch_names only) cannot go active in the ledger."""
    repository = JsonForecastRepository(tmp_path)
    question = ForecastQuestion(
        id="q-graph-shape",
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria="Milestone recorded.",
        allowed_branch_names=["advance", "reprice"],
        status=QuestionStatus.ACTIVE,
    )

    with pytest.raises(ValueError, match="non-empty 'branches'"):
        repository.save_question(question)


def test_save_load_and_list_run_refuses_duplicate_run_id(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    run = _run(question)

    repository.save_question(question)
    repository.save_run(run)

    assert repository.load_run(run.id) == run
    assert repository.list_runs() == [run]

    with pytest.raises(FileExistsError, match="forecast run already exists"):
        repository.save_run(run)


def test_save_load_and_list_resolution_validates_branch_against_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    resolution = _resolution(question)

    repository.save_resolution(resolution)

    assert repository.load_resolution(resolution.id) == resolution
    assert repository.list_resolutions() == [resolution]

    invalid_resolution = resolution.model_copy(
        update={"id": "resolution-bad", "resolved_branch": "maybe"}
    )
    with pytest.raises(ValueError, match="resolved branch must be one of the question branches"):
        repository.save_resolution(invalid_resolution)


def test_score_persistence(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    score = _score()

    repository.save_score(score)

    assert repository.load_score(score.id) == score
    assert repository.list_scores() == [score]


def _question(title: str = "Will Alice renew by February?") -> ForecastQuestion:
    return ForecastQuestion(
        id="q-binary",
        title=title,
        question_type=ForecastQuestionType.BINARY,
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Contract renewal is explicitly recorded.",
            resolved_by=LATER,
        ),
        branches=[OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")],
    )


def _run(question: ForecastQuestion) -> ForecastRun:
    return ForecastRun(
        id="run-q-binary",
        question_id=question.id,
        dossier_id="dossier-q-binary",
        forecast_as_of=question.forecast_as_of,
        branch_ids=[branch.id for branch in question.branches],
        probabilities={"yes": 0.7, "no": 0.3},
        top_branch="yes",
        protocol="deterministic-baseline",
    )


def _resolution(question: ForecastQuestion) -> ForecastResolution:
    return ForecastResolution(
        id="resolution-q-binary",
        question_id=question.id,
        branch_ids=[branch.id for branch in question.branches],
        resolved_branch="yes",
        resolved_at=LATER,
        evidence_ids=["e-1"],
    )


def _score() -> ForecastScore:
    return ForecastScore(
        id="score-run-q-binary",
        run_id="run-q-binary",
        question_id="q-binary",
        resolved_branch="yes",
        probability_assigned=0.7,
        brier_score=0.18,
        log_score=0.3566749439,
        top_1_correct=True,
    )


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
