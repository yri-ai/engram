"""Tests for forecast cutoff and leakage control."""

from __future__ import annotations

from datetime import UTC, datetime

from engram.models.branch_forecasting import ContextBudget, EvidenceItem
from engram.services.branch_forecasting import BranchForecaster


def test_forecaster_excludes_future_evidence_when_forecast_as_of_is_set() -> None:
    forecaster = BranchForecaster()
    evidence = [
        EvidenceItem(
            id="om",
            text="Offering memorandum is available for underwriting.",
            event_type="offering_memorandum",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="rent-roll",
            text="Current rent roll is complete and supports diligence.",
            event_type="rent_roll",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="operating-statement",
            text="Trailing operating statement is available.",
            event_type="operating_statement",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="underwriting",
            text="Underwriting model is complete and reconciled.",
            event_type="underwriting_model",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="tax-bill",
            text="Property tax bill was reviewed as baseline tax data.",
            event_type="tax_data",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="future-tax-shock",
            text="A tax increase notice arrived after the forecast date.",
            event_type="tax_increase",
            salience=1.0,
            timestamp="2026-06-15T00:00:00Z",
        ),
    ]

    forecast = forecaster.forecast(
        objective="acquisition diligence risk",
        structural_family="real_estate_acquisition",
        evidence=evidence,
        budget=ContextBudget(max_items=5, max_tokens=80),
        forecast_as_of=datetime(2026, 5, 15, tzinfo=UTC),
    )

    assert forecast.top_branch == "advance_diligence"
    assert all(item.id != "future-tax-shock" for item in forecast.selected_context)


def test_forecaster_without_cutoff_can_use_later_evidence() -> None:
    forecaster = BranchForecaster()
    evidence = [
        EvidenceItem(
            id="om",
            text="Offering memorandum is available for underwriting.",
            event_type="offering_memorandum",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="rent-roll",
            text="Current rent roll is complete and supports diligence.",
            event_type="rent_roll",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="operating-statement",
            text="Trailing operating statement is available.",
            event_type="operating_statement",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="underwriting",
            text="Underwriting model is complete and reconciled.",
            event_type="underwriting_model",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="tax-bill",
            text="Property tax bill was reviewed as baseline tax data.",
            event_type="tax_data",
            salience=0.9,
            timestamp="2026-05-01T00:00:00Z",
        ),
        EvidenceItem(
            id="future-tax-shock",
            text="A tax increase notice arrived after the forecast date.",
            event_type="tax_increase",
            salience=1.0,
            timestamp="2026-06-15T00:00:00Z",
        ),
    ]

    forecast = forecaster.forecast(
        objective="acquisition diligence risk",
        structural_family="real_estate_acquisition",
        evidence=evidence,
        budget=ContextBudget(max_items=5, max_tokens=80),
    )

    assert any(item.id == "future-tax-shock" for item in forecast.selected_context)
