#!/usr/bin/env python3
"""Fetch CourtListener/RECAP docket metadata for corpus curation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path("data/corpus/courtlistener")
MANIFEST_DIR = Path("data/manifests")


def parse_courtlistener_search(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for result in payload.get("results", []):
        rows.append({
            "id": str(result.get("id", "")),
            "docket_number": str(result.get("docketNumber") or result.get("docket_number") or ""),
            "case_name": str(result.get("caseName") or result.get("case_name") or ""),
            "date_filed": str(result.get("dateFiled") or result.get("date_filed") or ""),
            "absolute_url": str(result.get("absolute_url") or result.get("absoluteUrl") or ""),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default='real estate foreclosure restructure')
    parser.add_argument("--max-results", type=int, default=20)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "application/json"}
    token = os.getenv("COURTLISTENER_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        resp = client.get(
            "https://www.courtlistener.com/api/rest/v4/search/",
            params={"q": args.query, "type": "r", "page_size": args.max_results},
        )
        resp.raise_for_status()
        rows = parse_courtlistener_search(resp.json())
    (DATA_DIR / "search_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (MANIFEST_DIR / "courtlistener_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    time.sleep(0.1)


if __name__ == "__main__":
    main()
