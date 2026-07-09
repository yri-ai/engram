"""Tests for forecast repositories."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from engram.models.entity import Entity, EntityType
from engram.models.forecasting import (
    BaselineDecisionRecord,
    BeliefUpdate,
    DecisionRecord,
    EvidenceDossier,
    EvidenceItem,
    ForecastQuestion,
    ForecastQuestionType,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
    OutcomeBranch,
    QuestionStatus,
    ResolutionCriteria,
)
from engram.services.forecast_repository import (
    ForecastRepository,
    JsonForecastRepository,
    SchemaVersionError,
    migrate_json_forecast_ledger,
)
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
    repository.save_dossier(_dossier(question))
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


def test_json_ledger_validates_schema_version_on_read(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    payload = json.loads((tmp_path / "questions" / f"{question.id}.json").read_text())
    payload["schema_version"] = 999
    (tmp_path / "questions" / f"{question.id}.json").write_text(json.dumps(payload))

    with pytest.raises(SchemaVersionError, match="Unsupported schema_version 999"):
        repository.load_question(question.id)


def test_dossier_round_trip_and_immutability(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    dossier = _dossier(_question())

    repository.save_dossier(dossier)

    assert repository.load_dossier(dossier.id) == dossier
    assert repository.list_dossiers() == [dossier]
    with pytest.raises(FileExistsError, match="evidence dossier already exists"):
        repository.save_dossier(dossier)


def test_save_run_rejects_missing_dossier_reference(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)

    with pytest.raises(ValueError, match="dossier_id does not exist"):
        repository.save_run(_run(question))


def test_save_run_rejects_dossier_for_different_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    other_question = _question().model_copy(update={"id": "q-other"})
    repository.save_question(question)
    repository.save_question(other_question)
    repository.save_dossier(_dossier(other_question).model_copy(update={"id": "dossier-q-other"}))

    with pytest.raises(ValueError, match="same question_id"):
        repository.save_run(_run(question).model_copy(update={"dossier_id": "dossier-q-other"}))


def test_save_run_rejects_dossier_with_different_forecast_as_of(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    mismatched_dossier = _dossier(question).model_copy(update={"forecast_as_of": LATER})
    repository.save_dossier(mismatched_dossier)

    with pytest.raises(ValueError, match="matching forecast_as_of"):
        repository.save_run(_run(question))


def test_save_resolution_rejects_second_resolution_for_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_resolution(_resolution(question))

    second = _resolution(question).model_copy(update={"id": "resolution-q-binary-2"})
    with pytest.raises(ValueError, match="question already has a resolution"):
        repository.save_resolution(second)


def test_migrate_json_forecast_ledger_copies_and_verifies(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    repository = JsonForecastRepository(source)
    question = _question()
    dossier = _dossier(question)
    run = _run(question)
    repository.save_question(question)
    repository.save_dossier(dossier)
    repository.save_run(run)
    repository.save_resolution(_resolution(question))
    repository.save_decision(_decision())
    repository.save_baseline_decision(_baseline_decision())
    repository.save_score(_score())

    summary = migrate_json_forecast_ledger(source, target)

    assert summary == {
        "questions": ["q-binary"],
        "dossiers": ["dossier-q-binary"],
        "runs": ["run-q-binary"],
        "resolutions": ["resolution-q-binary"],
        "decisions": ["decision-q-binary"],
        "baseline_decisions": ["baseline-decision-q-binary"],
        "updates": [],
        "scores": ["score-run-q-binary"],
    }
    migrated = JsonForecastRepository(target)
    assert migrated.load_run(run.id) == run
    assert migrated.list_decisions() == [_decision()]
    assert migrated.list_baseline_decisions() == [_baseline_decision()]
    with pytest.raises(ValueError, match="in-place migration"):
        migrate_json_forecast_ledger(source, source)


def test_migrate_json_forecast_ledger_preserves_pre_dossier_runs(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    repository = JsonForecastRepository(source)
    question = _question()
    run = _run(question)
    repository.save_question(question)
    # Simulate a pre-M7 ledger: runs may cite a dossier id before dossiers were
    # ledger-persisted. Migration preserves the artifact; future save_run calls
    # still enforce the new dangling-dossier guard.
    (source / "runs" / f"{run.id}.json").write_text(
        run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    summary = migrate_json_forecast_ledger(source, target)

    assert summary["runs"] == [run.id]
    assert JsonForecastRepository(target).load_run(run.id) == run


def test_forecast_linked_decision_round_trip(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    decision = _decision()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))

    repository.save_decision(decision)

    assert repository.load_decision(decision.decision_id) == decision
    assert repository.list_decisions() == [decision]
    assert (tmp_path / "decisions" / f"{decision.decision_id}.json").exists()


def test_baseline_decision_round_trip(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    decision = _baseline_decision()

    repository.save_baseline_decision(decision)

    assert repository.load_baseline_decision(decision.decision_id) == decision
    assert repository.list_baseline_decisions() == [decision]
    assert (tmp_path / "baseline_decisions" / f"{decision.decision_id}.json").exists()


def test_save_decision_rejects_dangling_primary_run(tmp_path):
    repository = JsonForecastRepository(tmp_path)

    with pytest.raises(ValueError, match="primary_forecast_run_id does not exist"):
        repository.save_decision(_decision())


def test_save_decision_rejects_dangling_supporting_run(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    decision = _decision().model_copy(update={"supporting_forecast_run_ids": ["missing-run"]})

    with pytest.raises(ValueError, match="supporting_forecast_run_id does not exist"):
        repository.save_decision(decision)


def test_save_decision_rejects_primary_run_in_supporting_runs(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    decision = _decision().model_copy(update={"supporting_forecast_run_ids": ["run-q-binary"]})

    with pytest.raises(ValueError, match="primary forecast run must not be supporting"):
        repository.save_decision(decision)


def test_save_decision_rejects_expected_branch_outside_primary_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    decision = _decision().model_copy(update={"expected_outcome_branch": "maybe"})

    with pytest.raises(
        ValueError, match="expected outcome branch must be one of the primary question branches"
    ):
        repository.save_decision(decision)


def test_decision_records_use_exclusive_create(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    decision = _decision()
    baseline_decision = _baseline_decision()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    repository.save_decision(decision)
    repository.save_baseline_decision(baseline_decision)

    with pytest.raises(FileExistsError, match="decision already exists"):
        repository.save_decision(decision)
    with pytest.raises(FileExistsError, match="baseline decision already exists"):
        repository.save_baseline_decision(baseline_decision)


def test_resolve_decision_rejects_realized_branch_outside_primary_question(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    repository.save_decision(_decision())

    with pytest.raises(ValueError, match="realized_outcome_branch"):
        repository.resolve_decision(
            "decision-q-binary",
            realized_outcome_branch="maybe",
            impact_value=None,
            impact_kind=None,
        )


def test_resolve_decision_is_guarded_one_time_update(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    decision = _decision()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    repository.save_decision(decision)

    resolved = repository.resolve_decision(
        decision.decision_id,
        realized_outcome_branch="yes",
        impact_value=1250.0,
        impact_kind="hit",
    )

    assert resolved.realized_outcome_branch == "yes"
    assert resolved.impact_value == 1250.0
    assert resolved.impact_kind == "hit"
    assert repository.load_decision(decision.decision_id) == resolved
    with pytest.raises(ValueError, match="decision is already resolved"):
        repository.resolve_decision(
            decision.decision_id,
            realized_outcome_branch="no",
            impact_value=-1250.0,
            impact_kind="miss",
        )


def test_resolve_baseline_decision_is_guarded_one_time_update(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    decision = _baseline_decision()
    repository.save_baseline_decision(decision)

    resolved = repository.resolve_baseline_decision(
        decision.decision_id,
        realized_outcome_branch="no",
        impact_value=-500.0,
        impact_kind="miss",
    )

    assert resolved.realized_outcome_branch == "no"
    assert resolved.impact_value == -500.0
    assert resolved.impact_kind == "miss"
    assert repository.load_baseline_decision(decision.decision_id) == resolved
    with pytest.raises(ValueError, match="baseline decision is already resolved"):
        repository.resolve_baseline_decision(
            decision.decision_id,
            realized_outcome_branch="yes",
            impact_value=500.0,
            impact_kind="hit",
        )


def test_decision_records_expose_pending_and_resolved_classification_fields(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    forecast_decision = _decision()
    baseline_decision = _baseline_decision()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question))
    repository.save_decision(forecast_decision)
    repository.save_baseline_decision(baseline_decision)

    assert repository.list_decisions()[0].realized_outcome_branch is None
    assert repository.list_baseline_decisions()[0].realized_outcome_branch is None

    repository.resolve_decision(
        forecast_decision.decision_id,
        realized_outcome_branch="yes",
        impact_value=1250.0,
        impact_kind="hit",
    )
    repository.resolve_baseline_decision(
        baseline_decision.decision_id,
        realized_outcome_branch="no",
        impact_value=-500.0,
        impact_kind="miss",
    )

    assert repository.list_decisions()[0].realized_outcome_branch == "yes"
    assert repository.list_baseline_decisions()[0].realized_outcome_branch == "no"


def test_belief_update_round_trip_and_validation(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question()
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    prior = _run(question)
    posterior = prior.model_copy(
        update={"id": "run-q-binary-posterior", "probabilities": {"yes": 0.8, "no": 0.2}}
    )
    repository.save_run(prior)
    repository.save_run(posterior)
    update = BeliefUpdate(
        update_id="update-1",
        prior_run_id=prior.id,
        posterior_run_id=posterior.id,
        trigger_evidence_ids=["e-1"],
        update_at=question.forecast_as_of,
    )

    repository.save_update(update)

    assert repository.load_update(update.update_id) == update
    assert repository.list_updates() == [update]
    with pytest.raises(FileExistsError, match="belief update already exists"):
        repository.save_update(update)


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


def _dossier(question: ForecastQuestion) -> EvidenceDossier:
    return EvidenceDossier(
        id="dossier-q-binary",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=[
            EvidenceItem(
                id="e-1",
                text="Alice requested renewal paperwork.",
                valid_from=question.forecast_as_of,
                recorded_from=question.forecast_as_of,
                source_id="source-1",
                supports_branch=["yes"],
                supersession_status="current_as_of",
            )
        ],
        compiler="json_evidence.v1",
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


def _decision() -> DecisionRecord:
    return DecisionRecord(
        decision_id="decision-q-binary",
        decided_at=NOW,
        decision_type="renewal_offer",
        primary_forecast_run_id="run-q-binary",
        rationale="Offer because renewal is the most likely outcome.",
        expected_outcome_branch="yes",
    )


def _baseline_decision() -> BaselineDecisionRecord:
    return BaselineDecisionRecord(
        decision_id="baseline-decision-q-binary",
        decided_at=NOW,
        decision_type="renewal_offer",
        rationale="Pre-forecast baseline decision.",
        expected_outcome_branch="yes",
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


@pytest.mark.asyncio
async def test_json_graph_parity_for_questions_runs_and_resolutions(
    store: MemoryStore, tmp_path
) -> None:
    deal = _make_deal_entity()
    await store.upsert_entity(deal)
    graph_repository = ForecastRepository(
        store,
        tenant_id="tenant-1",
        conversation_id="forecasting",
        message_id="forecast-message-1",
    )
    json_repository = JsonForecastRepository(tmp_path)

    for index in range(3):
        question = _make_question().model_copy(update={"id": f"fq-{index}"})
        run = _make_run().model_copy(update={"id": f"fr-{index}", "question_id": question.id})
        resolution = _make_resolution().model_copy(
            update={"question_id": question.id, "run_id": run.id}
        )
        await graph_repository.save_question(question)
        await graph_repository.save_run(target_entity_id=deal.id, run=run)
        await graph_repository.save_resolution(target_entity_id=deal.id, resolution=resolution)
        json_repository.save_question(question)
        json_repository.save_run(run)
        json_repository.save_resolution(resolution.model_copy(update={"id": f"res-{index}"}))

    graph_questions = await graph_repository.list_questions(target_entity_id=deal.id)
    graph_runs = []
    for question in graph_questions:
        graph_runs.extend(
            await graph_repository.list_runs(target_entity_id=deal.id, question_id=question.id)
        )
    graph_resolutions = [
        await graph_repository.get_resolution(target_entity_id=deal.id, question_id=question.id)
        for question in graph_questions
    ]

    assert _normalized_models(json_repository.list_questions()) == _normalized_models(
        graph_questions
    )
    assert _normalized_models(json_repository.list_runs()) == _normalized_models(graph_runs)
    assert _normalized_models(
        json_repository.list_resolutions(), exclude={"id"}
    ) == _normalized_models(
        [resolution for resolution in graph_resolutions if resolution is not None], exclude={"id"}
    )


def _normalized_models(models, *, exclude: set[str] | None = None):  # type: ignore[no-untyped-def]
    return sorted(
        json.dumps(model.model_dump(mode="json", exclude=exclude or set()), sort_keys=True)
        for model in models
    )
