"""Phase 4 deal data, schema, waterfall, and deal engine coverage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.forecasting.deal_engine import predict_deal
from engram.forecasting.deal_extract import approve_spec, emit_review_markdown
from engram.forecasting.waterfall import simulate
from engram.models.deal import DealSpec, Tranche, Trigger, WaterfallStep
from engram.models.relationship import RelationshipType
from engram.services.dbrs_rating_parser import parse_rating_transitions
from engram.services.deal_remit_parser import parse_trigger_states
from engram.services.deal_repository import DealRepository


def _deal_spec(*, verified: bool = True, spec_id: str = "spec-1", supersedes: str | None = None) -> DealSpec:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return DealSpec(
        spec_id=spec_id,
        deal_id="deal-1",
        valid_from=now,
        recorded_from=now,
        supersedes_spec_id=supersedes,
        verified=verified,
        verified_by="tester" if verified else None,
        verified_at=now if verified else None,
        tranches=[Tranche(tranche_id="A", balance=100.0, coupon=0.05, seniority=1)],
        triggers=[Trigger(trigger_id="oc", trigger_type="OC", formula="assets / liabilities", threshold=50.0)],
        waterfall=[WaterfallStep(step_id="p1", tranche_id="A", rule="principal", priority=1)],
    )


def test_deal_spec_restricted_formula_and_repository_as_of_semantics():
    with pytest.raises(ValueError):
        Trigger(trigger_id="bad", trigger_type="OC", formula="__import__('os').system('x')", threshold=1.0)
    repo = DealRepository()
    original = _deal_spec(spec_id="spec-1")
    replacement = _deal_spec(spec_id="spec-2", supersedes="spec-1")
    repo.write(original)
    repo.write(replacement)
    assert repo.latest_as_of("deal-1", "2025-01-02T00:00:00+00:00") == replacement
    assert repo.rollback_manifest() == ["spec-1", "spec-2"]


def test_waterfall_requires_verification_and_conserves_cash(tmp_path):
    unverified = _deal_spec(verified=False)
    with pytest.raises(ValueError):
        simulate(unverified, [25.0])
    spec = approve_spec(unverified, verified_by="leo", verified_at="2025-01-02T00:00:00+00:00")
    review = emit_review_markdown(spec, tmp_path)
    assert review.exists()
    outcome = simulate(spec, [25.0, 25.0, 25.0, 25.0])
    assert sum(sum(values) for values in outcome.tranche_cashflows.values()) + sum(outcome.retained_cash) == 100.0
    prediction = predict_deal(spec, as_of="2025-01-02", n_paths=3, compute_budget=10)
    assert 0.0 <= prediction["p_trigger_breach"] <= 1.0


def test_phase4_parsers_and_relationship_vocabulary():
    ratings = parse_rating_transitions("tranche_id,from,to,date\nA,AAA,AA,2025-01-01", deal_id="deal-1", source_id="src")
    remits = parse_trigger_states("OC,2025-01,pass", deal_id="deal-1", source_id="src")
    assert ratings[0]["rating_to"] == "AA"
    assert remits[0]["passed"] is True
    assert RelationshipType.LOAN_IN_POOL.value == "loan_in_pool"
