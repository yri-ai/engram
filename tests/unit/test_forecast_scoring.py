"""Tests for forecast scoring metrics service."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engram.models.forecasting import ForecastResolution, ForecastRun
from engram.services.forecast_scoring import ForecastScorer


def _run(
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


def _resolution(question_id: str, run_id: str, outcome_branch: str) -> ForecastResolution:
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
        _run("fr-1", "fq-1", {"advance": 0.8, "reprice": 0.2}, "advance"),
        _run("fr-2", "fq-2", {"advance": 0.3, "reprice": 0.7}, "reprice"),
        _run("fr-3", "fq-3", {"advance": 0.6, "reprice": 0.4}, "advance"),
    ]
    resolutions = [
        _resolution("fq-1", "fr-1", "advance"),
        _resolution("fq-2", "fr-2", "advance"),
        _resolution("fq-3", "fr-3", "advance"),
    ]

    report = scorer.score_runs(runs, resolutions, bins=2)

    assert report["aggregate"]["sample_count"] == 3
    assert report["aggregate"]["top_1_accuracy"] == pytest.approx(2 / 3)
    assert report["aggregate"]["brier_score"] == pytest.approx(0.46)
    assert 0.0 <= report["aggregate"]["expected_calibration_error"] <= 1.0


def test_forecast_scorer_emits_per_question_scores() -> None:
    scorer = ForecastScorer()
    report = scorer.score_runs(
        [_run("fr-1", "fq-1", {"advance": 0.8, "reprice": 0.2}, "advance")],
        [_resolution("fq-1", "fr-1", "advance")],
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
