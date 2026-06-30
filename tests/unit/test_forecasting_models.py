import hashlib
import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engram.models.forecasting import (
    CalibrationSummary,
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

NOW = datetime(2026, 1, 15, tzinfo=UTC)
LATER = datetime(2026, 2, 15, tzinfo=UTC)


def test_binary_question_creation():
    question = ForecastQuestion(
        id="q-binary",
        title="Will Alice renew by February?",
        question_type=ForecastQuestionType.BINARY,
        target_id="person:alice",
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Contract renewal is explicitly recorded.",
            resolved_by=LATER,
        ),
        branches=[
            OutcomeBranch(id="yes", label="Yes"),
            OutcomeBranch(id="no", label="No"),
        ],
    )

    assert question.status == QuestionStatus.DRAFT
    assert question.tenant_id == "default"
    assert question.horizon == "30d"
    assert [branch.id for branch in question.branches] == ["yes", "no"]


def test_closed_branch_question_creation_with_priors():
    question = ForecastQuestion(
        id="q-branch",
        title="Which plan will Alice choose?",
        question_type=ForecastQuestionType.CLOSED_BRANCH,
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Selected plan appears in billing records.",
            resolved_by=LATER,
        ),
        branches=[
            OutcomeBranch(id="basic", label="Basic", prior=0.25),
            OutcomeBranch(id="pro", label="Pro", prior=0.75),
        ],
    )

    assert question.branches[0].prior == 0.25
    assert question.question_type == ForecastQuestionType.CLOSED_BRANCH


def test_forecast_as_of_is_required():
    with pytest.raises(ValidationError):
        ForecastQuestion(
            id="q-missing-as-of",
            title="Will this fail?",
            question_type=ForecastQuestionType.BINARY,
            horizon="30d",
            resolution_criteria=ResolutionCriteria(description="Observed later."),
            branches=[OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")],
        )


def test_resolution_criteria_description_is_required():
    with pytest.raises(ValidationError):
        ForecastQuestion(
            id="q-missing-resolution",
            title="Will this fail?",
            question_type=ForecastQuestionType.BINARY,
            forecast_as_of=NOW,
            horizon="30d",
            branches=[OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")],
        )


def test_probability_distribution_must_match_branches_and_sum_to_one():
    question = _question()
    dossier = _dossier(question)

    ForecastRun(
        id="run-ok",
        question_id=question.id,
        dossier_id=dossier.id,
        forecast_as_of=question.forecast_as_of,
        branch_ids=["yes", "no"],
        probabilities={"yes": 0.6, "no": 0.4},
        top_branch="yes",
        protocol="deterministic-baseline",
    )

    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-missing-probability",
            question_id=question.id,
            dossier_id=dossier.id,
            forecast_as_of=question.forecast_as_of,
            branch_ids=["yes", "no"],
            probabilities={"yes": 0.6},
            top_branch="yes",
            protocol="deterministic-baseline",
        )

    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-bad-sum",
            question_id=question.id,
            dossier_id=dossier.id,
            forecast_as_of=question.forecast_as_of,
            branch_ids=["yes", "no"],
            probabilities={"yes": 0.6, "no": 0.6},
            top_branch="yes",
            protocol="deterministic-baseline",
        )


def test_probability_distribution_rejects_negative_or_above_one_values():
    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-bad-probability-range",
            question_id="q-binary",
            dossier_id="dossier-q-binary",
            forecast_as_of=NOW,
            branch_ids=["yes", "no"],
            probabilities={"yes": 1.2, "no": -0.2},
            top_branch="yes",
            protocol="deterministic-baseline",
        )


def test_probability_distribution_rejects_non_finite_values():
    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-nan-probability",
            question_id="q-binary",
            dossier_id="dossier-q-binary",
            forecast_as_of=NOW,
            branch_ids=["yes", "no"],
            probabilities={"yes": math.nan, "no": 1.0},
            top_branch="yes",
            protocol="deterministic-baseline",
        )


def test_top_branch_must_be_in_distribution_and_have_max_probability():
    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-bad-top",
            question_id="q-binary",
            dossier_id="dossier-q-binary",
            forecast_as_of=NOW,
            branch_ids=["yes", "no"],
            probabilities={"yes": 0.6, "no": 0.4},
            top_branch="maybe",
            protocol="deterministic-baseline",
        )

    with pytest.raises(ValidationError):
        ForecastRun(
            id="run-top-not-max",
            question_id="q-binary",
            dossier_id="dossier-q-binary",
            forecast_as_of=NOW,
            branch_ids=["yes", "no"],
            probabilities={"yes": 0.4, "no": 0.6},
            top_branch="yes",
            protocol="deterministic-baseline",
        )


def test_run_immutability_helper_fields():
    run = ForecastRun(
        id="run-immutable",
        question_id="q-binary",
        dossier_id="dossier-q-binary",
        forecast_as_of=NOW,
        branch_ids=["yes", "no"],
        probabilities={"yes": 0.6, "no": 0.4},
        top_branch="yes",
        protocol="deterministic-baseline",
    )

    assert run.is_append_only is True
    assert run.replaces_run_id is None
    assert run.created_at >= NOW

    with pytest.raises(ValidationError):
        run.id = "mutated"


def test_invalid_branch_resolution_is_rejected():
    with pytest.raises(ValidationError):
        ForecastResolution(
            id="resolution-q-binary",
            question_id="q-binary",
            branch_ids=["yes", "no"],
            resolved_branch="maybe",
            resolved_at=LATER,
            evidence_ids=["e-1"],
        )


def test_evidence_item_contains_protocol_required_fields():
    evidence = EvidenceItem(
        id="e-1",
        text="Alice said she plans to renew.",
        valid_from=NOW,
        valid_to=None,
        recorded_from=NOW,
        recorded_to=None,
        source_id="msg-1",
        source_span="0:32",
        supports_branch=["yes"],
        opposes_branch=["no"],
        supersession_status="current_as_of",
        supersedes_id=None,
        superseded_by_id=None,
        contradicts_ids=["e-0"],
        metadata={"speaker": "Alice"},
    )

    assert evidence.source_id == "msg-1"
    assert evidence.supports_branch == ["yes"]


def test_dossier_tracks_exclusions_and_missing_evidence():
    question = _question()
    dossier = EvidenceDossier(
        id="dossier-q-binary",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=[],
        excluded_counts={"future_record_time": 1},
        missing_evidence=["renewal contract"],
    )

    assert dossier.excluded_counts["future_record_time"] == 1
    assert dossier.missing_evidence == ["renewal contract"]


def test_score_and_calibration_contracts():
    score = ForecastScore(
        id="score-run-1",
        run_id="run-1",
        question_id="q-binary",
        resolved_branch="yes",
        probability_assigned=0.6,
        brier_score=0.32,
        log_score=0.5108256238,
        top_1_correct=True,
    )
    summary = CalibrationSummary(
        id="calibration-small",
        run_count=1,
        mean_brier_score=0.32,
        mean_log_score=0.5108256238,
        low_sample_warning=True,
    )

    assert score.probability_assigned == 0.6
    assert summary.low_sample_warning is True


def _question() -> ForecastQuestion:
    return ForecastQuestion(
        id="q-binary",
        title="Will Alice renew by February?",
        question_type=ForecastQuestionType.BINARY,
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Contract renewal is explicitly recorded.",
            resolved_by=LATER,
        ),
        branches=[OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")],
    )


def _dossier(question: ForecastQuestion) -> EvidenceDossier:
    return EvidenceDossier(
        id="dossier-q-binary",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=[],
    )


# Forecast lifecycle model tests merged from master.
def test_forecast_question_round_trips_required_fields() -> None:
    forecast_as_of = datetime(2026, 5, 1, tzinfo=UTC)
    resolution_due_at = datetime(2026, 6, 1, tzinfo=UTC)

    question = ForecastQuestion(
        id="fq-1",
        tenant_id="tenant-1",
        target_entity_id="deal-123",
        objective="Predict the next deal branch",
        structural_family="real_estate_acquisition",
        forecast_as_of=forecast_as_of,
        horizon="30d",
        resolution_due_at=resolution_due_at,
        resolution_criteria="Resolve when the deal closes, reprices, or terminates",
        allowed_branch_names=["advance_diligence", "reprice_or_restructure", "terminated_failed"],
    )

    payload = question.model_dump(mode="json")

    assert payload["forecast_as_of"] == "2026-05-01T00:00:00Z"
    assert payload["allowed_branch_names"] == [
        "advance_diligence",
        "reprice_or_restructure",
        "terminated_failed",
    ]


def test_forecast_run_rejects_probabilities_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        ForecastRun(
            id="fr-1",
            question_id="fq-1",
            model_or_engine="branch-forecaster-v1",
            forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
            branch_probabilities={
                "advance_diligence": 0.7,
                "reprice_or_restructure": 0.2,
            },
            top_branch="advance_diligence",
            selected_evidence_ids=["fact-1"],
            rationale="Enough diligence evidence is present.",
        )


def test_forecast_run_rejects_top_branch_not_in_probabilities() -> None:
    with pytest.raises(ValidationError, match="top_branch"):
        ForecastRun(
            id="fr-2",
            question_id="fq-1",
            model_or_engine="branch-forecaster-v1",
            forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
            branch_probabilities={
                "advance_diligence": 0.8,
                "reprice_or_restructure": 0.2,
            },
            top_branch="terminated_failed",
            selected_evidence_ids=["fact-1"],
            rationale="Top branch must come from probabilities.",
        )


def test_forecast_resolution_and_score_round_trip() -> None:
    resolution = ForecastResolution(
        question_id="fq-1",
        run_id="fr-1",
        resolved_at=datetime(2026, 6, 15, tzinfo=UTC),
        outcome_branch="closed_repriced",
        outcome_probability_target=1.0,
        resolution_notes="Seller accepted revised price after diligence.",
        resolved_by="analyst@example.com",
        source="ic_memo_2026_06_15",
    )
    score = ForecastScore(
        question_id="fq-1",
        run_id="fr-1",
        brier_score=0.18,
        top_1_correct=True,
        calibration_bucket="0.8-0.9",
        expected_calibration_error=0.06,
        sample_count=25,
    )

    assert resolution.model_dump(mode="json")["outcome_branch"] == "closed_repriced"
    assert resolution.model_dump(mode="json")["run_id"] == "fr-1"
    assert score.model_dump(mode="json")["sample_count"] == 25


def test_forecast_ids_are_deterministic() -> None:
    forecast_as_of = datetime(2026, 5, 1, tzinfo=UTC)

    question_id = ForecastQuestion.build_id(
        tenant_id="tenant-1",
        target_entity_id="deal-123",
        objective="Predict the next deal branch",
        forecast_as_of=forecast_as_of,
    )
    run_id = ForecastRun.build_id(
        question_id=question_id,
        model_or_engine="branch-forecaster-v1",
        forecast_as_of=forecast_as_of,
        config={"max_items": 6, "max_tokens": 1200},
    )

    assert question_id == ForecastQuestion.build_id(
        tenant_id="tenant-1",
        target_entity_id="deal-123",
        objective="Predict the next deal branch",
        forecast_as_of=forecast_as_of,
    )
    assert run_id == ForecastRun.build_id(
        question_id=question_id,
        model_or_engine="branch-forecaster-v1",
        forecast_as_of=forecast_as_of,
        config={"max_tokens": 1200, "max_items": 6},
    )
    assert ForecastResolution.build_id(question_id=question_id, run_id=run_id).endswith(
        hashlib.sha256(f"{question_id}|{run_id}".encode()).hexdigest()[:16]
    )
