#!/usr/bin/env python3
"""Fetch public EDGAR REIT filings for corpus curation.

Network fetching is intentionally small and manifest-oriented; tests exercise the
pure parser helpers with checked-in golden samples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path("data/corpus/edgar")
MANIFEST_DIR = Path("data/manifests")
HEADERS = {"User-Agent": "Engram Research engram-research@yri.ai", "Accept": "application/json"}


def parse_edgar_company_submissions(payload: dict[str, Any], *, forms: set[str] | None = None) -> list[dict[str, str]]:
    forms = forms or {"8-K", "10-Q", "10-K"}
    recent = payload.get("filings", {}).get("recent", {})
    rows = []
    for form, accession, filing_date, primary_doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", []),
        strict=False,
    ):
        if form in forms:
            rows.append({
                "form": form,
                "accession": accession,
                "filing_date": filing_date,
                "primary_document": primary_doc,
            })
    return rows


def build_archive_url(cik: str, accession: str, primary_document: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{primary_document}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", action="append", required=True, help="REIT CIK; repeatable")
    parser.add_argument("--max-filings", type=int, default=10)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        for cik in args.cik:
            submissions_response = client.get(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json")
            submissions_response.raise_for_status()
            payload = submissions_response.json()
            for row in parse_edgar_company_submissions(payload)[: args.max_filings]:
                url = build_archive_url(cik, row["accession"], row["primary_document"])
                out = DATA_DIR / f"{cik}_{row['accession'].replace('-', '')}_{row['primary_document']}"
                filing_response = client.get(url)
                filing_response.raise_for_status()
                out.write_text(filing_response.text, encoding="utf-8")
                manifest.append(row | {"cik": cik, "url": url, "path": str(out)})
                time.sleep(0.12)
    (MANIFEST_DIR / "edgar_reit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
