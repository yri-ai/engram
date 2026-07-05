"""Tests for public corpus acquisition parsers and lifecycle loader."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from engram.services.forecast_audit import build_audit_report
from engram.services.forecast_repository import JsonForecastRepository

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "corpus"


def _load_script(name: str):
    path = ROOT / "scripts" / "data_collection" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_edgar_reit_parser_golden_response():
    module = _load_script("fetch_edgar_reit.py")
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "4", "10-Q"],
                "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003"],
                "filingDate": ["2026-01-10", "2026-01-11", "2026-02-01"],
                "primaryDocument": ["reit8k.htm", "owner.xml", "reit10q.htm"],
            }
        }
    }

    rows = module.parse_edgar_company_submissions(payload)

    assert [row["form"] for row in rows] == ["8-K", "10-Q"]
    assert module.build_archive_url("123", rows[0]["accession"], rows[0]["primary_document"]).endswith(
        "/123/000126000001/reit8k.htm"
    )


def test_courtlistener_parser_golden_response():
    module = _load_script("fetch_courtlistener.py")
    payload = {
        "results": [
            {
                "id": 42,
                "docketNumber": "1:26-cv-1",
                "caseName": "Lender v. Borrower",
                "dateFiled": "2026-03-05",
                "absolute_url": "/docket/42/",
            }
        ]
    }

    rows = module.parse_courtlistener_search(payload)

    assert rows == [
        {
            "id": "42",
            "docket_number": "1:26-cv-1",
            "case_name": "Lender v. Borrower",
            "date_filed": "2026-03-05",
            "absolute_url": "/docket/42/",
        }
    ]


def test_deal_to_question_uses_latest_evidence_as_forecast_cutoff():
    module = _load_script("build_corpus_questions.py")
    fixture = json.loads((FIXTURES / "gnl-rtl-merger-2023.json").read_text(encoding="utf-8"))
    fixture["evidence_docs"].append(
        {
            "doc_id": "later-public-source",
            "url": "https://example.com/later",
            "published_at": "2023-09-13T00:00:00+00:00",
            "retrieved_at": "2026-07-05T00:00:00+00:00",
            "text_ref": "later source",
            "summary": "Later public confirmation source.",
            "role": "forecast_evidence",
        }
    )
    fixture["resolved_at"] = "2026-07-05T00:00:00+00:00"
    deal = module.PublicDeal.model_validate(fixture)

    question = module.deal_to_question(deal)
    dossier = module.deal_to_dossier(deal, question)

    assert question.forecast_as_of.isoformat() == "2023-09-13T00:00:00+00:00"
    assert len(dossier.evidence_items) == 2
    assert dossier.evidence_items[-1].source_id == "later-public-source"


def test_build_corpus_questions_creates_auditable_ledger(tmp_path):
    module = _load_script("build_corpus_questions.py")
    repo = tmp_path / "corpus-ledger"

    summary = module.build_corpus_ledger(FIXTURES, repo)

    repository = JsonForecastRepository(repo)
    expected_count = len(list(FIXTURES.glob("*.json")))
    source_counts: dict[str, int] = {}
    for fixture_path in FIXTURES.glob("*.json"):
        source_kind = json.loads(fixture_path.read_text(encoding="utf-8"))["source_kind"]
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

    assert summary == {
        "deal_count": expected_count,
        "question_count": expected_count,
        "dossier_count": expected_count,
        "run_count": expected_count,
        "resolution_count": expected_count,
    }
    assert source_counts["edgar_reit"] >= 12
    assert source_counts["courtlistener"] >= 8
    assert repository.list_questions()[0].id.startswith("corpus-question:")
    assert all(item.recorded_from <= dossier.forecast_as_of for dossier in repository.list_dossiers() for item in dossier.evidence_items)
    report = build_audit_report(repository, spot_check=2)
    assert report["status"] == "PASS"
    assert report["run_count"] == expected_count
