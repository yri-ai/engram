"""Tests for public corpus models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.models.corpus import DEFAULT_CORPUS_TAXONOMY, PublicDeal

FIXTURES = Path(__file__).parents[1] / "fixtures" / "corpus"


def test_public_deal_fixture_schema_validation():
    for path in sorted(FIXTURES.glob("*.json")):
        deal = PublicDeal.model_validate_json(path.read_text())
        assert deal.resolved_branch in DEFAULT_CORPUS_TAXONOMY.branches
        assert deal.resolved_at >= max(
            [doc.published_at for doc in deal.evidence_docs] + [milestone.at for milestone in deal.milestones]
        )


def test_public_deal_rejects_invalid_branch():
    payload = json.loads((FIXTURES / "edgar_reit_sample.json").read_text())
    payload["resolved_branch"] = "not_a_branch"

    with pytest.raises(ValueError, match="branch id is not in taxonomy"):
        PublicDeal.model_validate(payload)


def test_public_deal_rejects_resolution_before_evidence():
    payload = json.loads((FIXTURES / "edgar_reit_sample.json").read_text())
    payload["resolved_at"] = "2026-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="resolved_at must be on or after"):
        PublicDeal.model_validate(payload)


def test_public_deal_rejects_retrieval_before_publication():
    payload = json.loads((FIXTURES / "edgar_reit_sample.json").read_text())
    payload["evidence_docs"][0]["retrieved_at"] = "2026-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="retrieved_at must be on or after"):
        PublicDeal.model_validate(payload)
