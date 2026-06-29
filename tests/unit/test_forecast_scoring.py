from datetime import UTC, datetime

import pytest

from engram.models.forecasting import (
    ForecastQuestion,
    ForecastQuestionType,
    ForecastResolution,
    ForecastRun,
    OutcomeBranch,
    QuestionStatus,
    ResolutionCriteria,
)
from engram.services.forecast_repository import JsonForecastRepository
from engram.services.forecast_scoring import (
    assign_calibration_bucket,
    binary_brier_score,
    build_calibration_report,
    log_score,
    multiclass_brier_score,
    probability_assigned_to_resolved_branch,
    score_forecast_run,
    top_1_accuracy,
    top_k_accuracy,
)

NOW = datetime(2026, 1, 15, tzinfo=UTC)
LATER = datetime(2026, 2, 15, tzinfo=UTC)


def test_binary_brier_scores_resolved_branch_probability():
    assert binary_brier_score(0.7, resolved=True) == pytest.approx(0.09)
    assert binary_brier_score(0.7, resolved=False) == pytest.approx(0.49)


def test_multiclass_brier_scores_full_distribution():
    probabilities = {"a": 0.2, "b": 0.5, "c": 0.3}

    assert multiclass_brier_score(probabilities, "b") == pytest.approx(0.38)


def test_log_score_clips_extreme_probabilities():
    assert log_score(0.8) == pytest.approx(0.2231435513)
    assert log_score(0.0, epsilon=1e-6) == pytest.approx(13.815510558)
    assert log_score(1.0, epsilon=1e-6) == pytest.approx(0.0000010000005)


def test_top_1_and_top_k_accuracy_use_probability_ranking_with_lexicographic_ties():
    run = _run(
        _question(
            "q-multi",
            [
                OutcomeBranch(id="alpha", label="Alpha"),
                OutcomeBranch(id="beta", label="Beta"),
                OutcomeBranch(id="gamma", label="Gamma"),
            ],
        ),
        probabilities={"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
        top_branch="alpha",
    )

    assert top_1_accuracy(run, "alpha") is True
    assert top_1_accuracy(run, "beta") is False
    assert top_k_accuracy(run, "beta", k=2) is True
    assert top_k_accuracy(run, "gamma", k=2) is False


def test_probability_assigned_to_resolved_branch_requires_branch_probability():
    run = _run(_question("q-binary"), probabilities={"yes": 0.65, "no": 0.35}, top_branch="yes")

    assert probability_assigned_to_resolved_branch(run, "yes") == pytest.approx(0.65)

    with pytest.raises(ValueError, match="resolved branch is not present in run probabilities"):
        probability_assigned_to_resolved_branch(run, "maybe")


def test_calibration_bucket_assignment():
    assert assign_calibration_bucket(0.0, bucket_count=10) == 0
    assert assign_calibration_bucket(0.34, bucket_count=10) == 3
    assert assign_calibration_bucket(1.0, bucket_count=10) == 9


def test_score_forecast_run_builds_forecast_score():
    question = _question("q-score")
    run = _run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes")
    resolution = _resolution(question, resolved_branch="yes")

    score = score_forecast_run(run, resolution, top_k=2)

    assert score.id == "score-run-q-score"
    assert score.run_id == run.id
    assert score.question_id == question.id
    assert score.resolved_branch == "yes"
    assert score.probability_assigned == pytest.approx(0.7)
    assert score.brier_score == pytest.approx(0.18)
    assert score.log_score == pytest.approx(0.3566749439)
    assert score.top_1_correct is True
    assert score.top_k_correct is True
    assert score.metadata["calibration_bucket"] == 7


def test_report_builder_scores_repository_runs_and_warns_on_low_sample(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-report")
    run = _run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes")
    resolution = _resolution(question, resolved_branch="yes")
    repository.save_question(question)
    repository.save_run(run)
    repository.save_resolution(resolution)

    report = build_calibration_report(repository, bucket_count=5, low_sample_threshold=2)

    assert report.run_count == 1
    assert report.mean_brier_score == pytest.approx(0.18)
    assert report.mean_log_score == pytest.approx(0.3566749439)
    assert report.low_sample_warning is True
    assert report.metadata["scored_run_ids"] == [run.id]
    assert report.metadata["skipped_run_count"] == 0
    assert repository.list_scores()[0].run_id == run.id
    bucket = next(bucket for bucket in report.buckets if bucket["bucket"] == 3)
    assert bucket["count"] == 1
    assert bucket["observed_frequency"] == pytest.approx(1.0)


def test_report_builder_buckets_top_branch_confidence_for_calibration(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-wrong")
    run = _run(question, probabilities={"yes": 0.8, "no": 0.2}, top_branch="yes")
    resolution = _resolution(question, resolved_branch="no")
    repository.save_question(question)
    repository.save_run(run)
    repository.save_resolution(resolution)

    report = build_calibration_report(repository, bucket_count=5, low_sample_threshold=0)

    bucket = next(bucket for bucket in report.buckets if bucket["bucket"] == 4)
    assert bucket["count"] == 1
    assert bucket["mean_probability"] == pytest.approx(0.8)
    assert bucket["observed_frequency"] == pytest.approx(0.0)


def test_report_builder_can_skip_runs_missing_resolution(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-unresolved")
    repository.save_question(question)
    repository.save_run(_run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes"))

    report = build_calibration_report(repository, skip_missing_resolutions=True)

    assert report.run_count == 0
    assert report.metadata["skipped_run_count"] == 1
    assert report.metadata["skipped_run_ids"] == ["run-q-unresolved"]


def test_report_builder_errors_on_missing_resolution_by_default(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-unresolved")
    repository.save_question(question)
    repository.save_run(_run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes"))

    with pytest.raises(ValueError, match="missing resolution for run run-q-unresolved"):
        build_calibration_report(repository)


def _question(
    question_id: str,
    branches: list[OutcomeBranch] | None = None,
) -> ForecastQuestion:
    branches = branches or [
        OutcomeBranch(id="yes", label="Yes"),
        OutcomeBranch(id="no", label="No"),
    ]
    return ForecastQuestion(
        id=question_id,
        title="Will the event occur?",
        question_type=ForecastQuestionType.BINARY
        if len(branches) == 2
        else ForecastQuestionType.CLOSED_BRANCH,
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Resolution is recorded by the horizon.",
            resolved_by=LATER,
        ),
        branches=branches,
        status=QuestionStatus.ACTIVE,
    )


def _run(
    question: ForecastQuestion,
    probabilities: dict[str, float],
    top_branch: str,
) -> ForecastRun:
    return ForecastRun(
        id=f"run-{question.id}",
        question_id=question.id,
        dossier_id=f"dossier-{question.id}",
        forecast_as_of=question.forecast_as_of,
        branch_ids=[branch.id for branch in question.branches],
        probabilities=probabilities,
        top_branch=top_branch,
        protocol="deterministic-baseline",
    )


def _resolution(question: ForecastQuestion, resolved_branch: str) -> ForecastResolution:
    return ForecastResolution(
        id=f"resolution-{question.id}",
        question_id=question.id,
        branch_ids=[branch.id for branch in question.branches],
        resolved_branch=resolved_branch,
        resolved_at=LATER,
    )
