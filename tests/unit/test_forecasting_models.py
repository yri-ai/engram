"""Tests for forecast lifecycle models."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engram.models.forecasting import (
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
)


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
