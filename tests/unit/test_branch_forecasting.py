"""Tests for schema-guided branch forecasting."""

from engram.models.branch_forecasting import BranchFeedback, ContextBudget, EvidenceItem
from engram.services.branch_forecasting import (
    DEFAULT_BRANCH_FAMILIES,
    BayesianUpdateShell,
    BranchForecaster,
    ContextBudgetSelector,
    evidence_from_directory,
    evidence_from_records,
    infer_event_type,
)


def _margin_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="e1",
            text="Management reported input cost pressure from freight and commodities.",
            event_type="input_cost_pressure",
            salience=0.95,
            tokens=12,
        ),
        EvidenceItem(
            id="e2",
            text="Demand weakness appeared in the discretionary segment.",
            event_type="demand_weakness",
            salience=0.9,
            tokens=10,
        ),
        EvidenceItem(
            id="e3",
            text="A small marketing launch is related but not causal for margins.",
            event_type="brand_campaign",
            salience=1.0,
            tokens=8,
        ),
        EvidenceItem(
            id="e4",
            text="Pricing power was explicitly limited by competitive pressure.",
            event_type="pricing_power",
            salience=0.2,
            tokens=9,
        ),
    ]


def test_context_budget_selector_prefers_discriminative_evidence() -> None:
    selector = ContextBudgetSelector()
    branches = DEFAULT_BRANCH_FAMILIES["margin_analysis"]

    selected = selector.select(
        _margin_evidence(),
        branches,
        ContextBudget(max_items=2, max_tokens=25),
    )

    assert [item.id for item in selected] == ["e1", "e2"]
    assert sum(item.tokens for item in selected) <= 25


def test_branch_forecaster_ranks_supported_branch_over_distractor() -> None:
    forecaster = BranchForecaster()

    forecast = forecaster.forecast(
        objective="Q4 gross margin risk",
        structural_family="margin_analysis",
        evidence=_margin_evidence(),
        budget=ContextBudget(max_items=3, max_tokens=40),
    )

    assert forecast.top_branch == "margin_compression"
    top = forecast.scores[0]
    assert top.branch == "margin_compression"
    assert set(top.matched_evidence_ids) == {"e1", "e2"}
    assert "negative_mix_shift" in top.missing_precursors
    assert all(item.id != "e3" for item in forecast.selected_context)


def test_forecaster_surfaces_evidence_gaps_for_competing_branches() -> None:
    forecaster = BranchForecaster()

    forecast = forecaster.forecast(
        objective="Q4 gross margin risk",
        structural_family="margin_analysis",
        evidence=[
            EvidenceItem(
                id="cost",
                text="Input cost pressure remains elevated.",
                event_type="input_cost_pressure",
                salience=0.9,
            )
        ],
        budget=ContextBudget(max_items=3, max_tokens=20),
    )

    assert "demand_weakness" in forecast.evidence_gaps
    assert "negative_mix_shift" in forecast.evidence_gaps


def test_bayesian_update_shell_moves_relevance_belief() -> None:
    bayes = BayesianUpdateShell()
    before = bayes.expected_relevance("margin", "margin_compression")

    bayes.update(
        BranchFeedback(
            objective="margin",
            branch="margin_compression",
            useful=True,
            weight=3.0,
        )
    )

    after = bayes.expected_relevance("margin", "margin_compression")
    assert before == 0.5
    assert after > before


def test_forecaster_rejects_unknown_structural_family() -> None:
    forecaster = BranchForecaster()

    try:
        forecaster.forecast("risk", "unknown", [])
    except ValueError as exc:
        assert "unknown structural family" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_evidence_from_records_maps_plain_text_to_event_types() -> None:
    evidence = evidence_from_records(
        [
            {
                "record_id": "claim-1",
                "text": "Freight inflation created cost pressure in the quarter.",
                "confidence": 0.8,
            }
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].id == "claim-1"
    assert evidence[0].event_type == "input_cost_pressure"
    assert evidence[0].salience == 0.8
    assert evidence[0].tokens > 1


def test_infer_event_type_returns_empty_for_unknown_text() -> None:
    assert infer_event_type("The company held an analyst day.") == ""


def test_evidence_from_directory_maps_structured_deal_files(tmp_path) -> None:
    (tmp_path / "Rent Rolls").mkdir()
    (tmp_path / "Operating Statements").mkdir()
    (tmp_path / "Tax Bills").mkdir()
    (tmp_path / "Rent Rolls" / "Pura Vida Rent Roll Summary.xlsx").write_text("")
    (tmp_path / "Operating Statements" / "Pura Vida T12 March 2023.xls").write_text("")
    (tmp_path / "Tax Bills" / "2022 Property Tax Bill.pdf").write_text("")

    evidence = evidence_from_directory(tmp_path)
    event_types = {item.event_type for item in evidence}

    assert "rent_roll" in event_types
    assert "operating_statement" in event_types
    assert "tax_data" in event_types


def test_tax_trim_notice_is_negative_but_tax_bill_is_baseline_data() -> None:
    bill = evidence_from_records([{"id": "bill", "text": "2022 property tax bill"}])[0]
    notice = evidence_from_records([{"id": "trim", "text": "2022 trim notice"}])[0]

    assert bill.event_type == "tax_data"
    assert notice.event_type == "tax_increase"


def test_real_estate_acquisition_family_forecasts_from_structured_files(tmp_path) -> None:
    (tmp_path / "Rent Roll").mkdir()
    (tmp_path / "Financials").mkdir()
    (tmp_path / "Debt Matrix").mkdir()
    (tmp_path / "Rent Roll" / "LWE Rent Roll.xlsx").write_text("")
    (tmp_path / "Financials" / "LWE_T12 Ending Feb 23.xlsx").write_text("")
    (tmp_path / "Legacy West DC underwriting.xlsx").write_text("")
    (tmp_path / "Debt Matrix" / "LWE_Debt Matrix.pdf").write_text("")

    forecast = BranchForecaster().forecast(
        objective="acquisition diligence risk",
        structural_family="real_estate_acquisition",
        evidence=evidence_from_directory(tmp_path),
    )

    assert forecast.top_branch in {"advance_diligence", "reprice_or_restructure"}
    assert forecast.selected_context
