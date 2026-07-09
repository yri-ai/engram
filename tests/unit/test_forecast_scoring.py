import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engram.models.forecasting import (
    BeliefUpdate,
    EvidenceDossier,
    EvidenceItem,
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
    ForecastScorer,
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
    repository.save_dossier(_dossier(question))
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
    repository.save_dossier(_dossier(question))
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
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes"))

    report = build_calibration_report(repository, skip_missing_resolutions=True)

    assert report.run_count == 0
    assert report.metadata["skipped_run_count"] == 1
    assert report.metadata["skipped_run_ids"] == ["run-q-unresolved"]


def test_report_builder_errors_on_missing_resolution_by_default(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-unresolved")
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(_run(question, probabilities={"yes": 0.7, "no": 0.3}, top_branch="yes"))

    with pytest.raises(ValueError, match="missing resolution for run run-q-unresolved"):
        build_calibration_report(repository)


def test_forecasting_metrics_seam_reuses_canonical_scoring_functions():
    from engram.forecasting.metrics import multiclass_brier_score as seam_brier

    assert seam_brier({"yes": 0.7, "no": 0.3}, "yes") == multiclass_brier_score(
        {"yes": 0.7, "no": 0.3}, "yes"
    )


def test_calibration_report_includes_update_quality(tmp_path):
    repository = JsonForecastRepository(tmp_path)
    question = _question("q-update")
    prior = _run(question, probabilities={"yes": 0.6, "no": 0.4}, top_branch="yes")
    posterior = prior.model_copy(
        update={"id": "run-q-update-posterior", "probabilities": {"yes": 0.8, "no": 0.2}}
    )
    repository.save_question(question)
    repository.save_dossier(_dossier(question))
    repository.save_run(prior)
    repository.save_run(posterior)
    repository.save_resolution(_resolution(question, resolved_branch="yes"))
    repository.save_update(
        BeliefUpdate(
            update_id="update-q",
            prior_run_id=prior.id,
            posterior_run_id=posterior.id,
            trigger_evidence_ids=["e-q-update"],
            update_at=question.forecast_as_of,
        )
    )

    report = build_calibration_report(repository, low_sample_threshold=0)

    assert report.metadata["update_quality"]["update_count"] == 1
    assert report.metadata["update_quality"]["improved_count"] == 1


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


def _dossier(question: ForecastQuestion) -> EvidenceDossier:
    return EvidenceDossier(
        id=f"dossier-{question.id}",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=[
            EvidenceItem(
                id=f"e-{question.id}",
                text="Evidence available as of forecast time.",
                valid_from=question.forecast_as_of,
                recorded_from=question.forecast_as_of,
                source_id="source-1",
                supports_branch=[question.branches[0].id],
                supersession_status="current_as_of",
            )
        ],
        compiler="json_evidence.v1",
    )


def _resolution(question: ForecastQuestion, resolved_branch: str) -> ForecastResolution:
    return ForecastResolution(
        id=f"resolution-{question.id}",
        question_id=question.id,
        branch_ids=[branch.id for branch in question.branches],
        resolved_branch=resolved_branch,
        resolved_at=LATER,
    )


# ForecastScorer tests merged from master.
def _lifecycle_run(
    run_id: str,
    question_id: str,
    probabilities: dict[str, float],
    top_branch: str,
) -> ForecastRun:
    return ForecastRun(
        id=run_id,
        question_id=question_id,
        model_or_engine="branch_forecaster",
        forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
        branch_probabilities=probabilities,
        top_branch=top_branch,
        selected_evidence_ids=["fact-1"],
        rationale="test rationale",
    )


def _lifecycle_resolution(question_id: str, run_id: str, outcome_branch: str) -> ForecastResolution:
    return ForecastResolution(
        question_id=question_id,
        run_id=run_id,
        resolved_at=datetime(2026, 6, 1, tzinfo=UTC),
        outcome_branch=outcome_branch,
        resolved_by="analyst@example.com",
        source="test-source",
    )


def test_forecast_scorer_computes_aggregate_metrics() -> None:
    scorer = ForecastScorer()
    runs = [
        _lifecycle_run("fr-1", "fq-1", {"advance": 0.8, "reprice": 0.2}, "advance"),
        _lifecycle_run("fr-2", "fq-2", {"advance": 0.3, "reprice": 0.7}, "reprice"),
        _lifecycle_run("fr-3", "fq-3", {"advance": 0.6, "reprice": 0.4}, "advance"),
    ]
    resolutions = [
        _lifecycle_resolution("fq-1", "fr-1", "advance"),
        _lifecycle_resolution("fq-2", "fr-2", "advance"),
        _lifecycle_resolution("fq-3", "fr-3", "advance"),
    ]

    report = scorer.score_runs(runs, resolutions, bins=2)

    assert report["aggregate"]["sample_count"] == 3
    assert report["aggregate"]["top_1_accuracy"] == pytest.approx(2 / 3)
    assert report["aggregate"]["brier_score"] == pytest.approx(0.46)
    assert 0.0 <= report["aggregate"]["expected_calibration_error"] <= 1.0


def test_forecast_scorer_emits_per_question_scores() -> None:
    scorer = ForecastScorer()
    report = scorer.score_runs(
        [_lifecycle_run("fr-1", "fq-1", {"advance": 0.8, "reprice": 0.2}, "advance")],
        [_lifecycle_resolution("fq-1", "fr-1", "advance")],
        bins=5,
    )

    assert len(report["per_question"]) == 1
    score = report["per_question"][0]
    assert score["question_id"] == "fq-1"
    assert score["run_id"] == "fr-1"
    assert score["brier_score"] == pytest.approx(0.08)
    assert score["top_1_correct"] is True
    assert score["calibration_bucket"] == "0.8-1.0"
    assert score["expected_calibration_error"] == pytest.approx(0.2)
    assert score["sample_count"] == 1


def test_forecast_scorer_reads_fixture_and_emits_traceability_fields() -> None:
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "forecast_scores.json"
    payload = json.loads(fixture_path.read_text())
    runs = [ForecastRun.model_validate(item) for item in payload["runs"]]
    resolutions = [ForecastResolution.model_validate(item) for item in payload["resolutions"]]

    report = ForecastScorer().score_runs(runs, resolutions, bins=2)

    assert report["aggregate"]["sample_count"] == 3
    assert set(report["by_extraction_variant"]) == {"baseline", "structured_v1"}
    question_score = report["per_question"][0]
    assert question_score["target_entity_id"] == "deal-123"
    assert question_score["outcome_branch"] in {"advance", "reprice"}
    assert question_score["resolution_source"].startswith("memo-")
    assert question_score["extraction_variant"] in {"baseline", "structured_v1"}
