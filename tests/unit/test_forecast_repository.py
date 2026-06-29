from datetime import UTC, datetime

import pytest

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
from engram.services.forecast_repository import JsonForecastRepository

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
