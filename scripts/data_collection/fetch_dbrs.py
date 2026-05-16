#!/usr/bin/env python3
"""Fetch DBRS Morningstar CMBS presale and surveillance reports.

DBRS offers public search and report downloads.

Usage:
    python scripts/data_collection/fetch_dbrs.py [--max-reports 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dbrs" / "cmbs"

# DBRS search/download endpoints — these may need updating as the site evolves.
DBRS_BASE = "https://dbrs.morningstar.com"
DBRS_SEARCH = f"{DBRS_BASE}/search"


async def search_cmbs_reports(client: httpx.AsyncClient, max_reports: int = 50) -> list[dict]:
    """Search DBRS for CMBS-related reports."""
    # Note: DBRS's search API/URL structure may change.
    # This provides the framework — the user may need to capture
    # actual API calls from their browser's network tab.
    reports: list[dict] = []

    params = {
        "q": "CMBS",
        "assetClass": "Structured Finance",
        "subAssetClass": "CMBS",
        "documentType": "Presale,Surveillance",
        "limit": min(max_reports, 50),
        "offset": 0,
    }

    try:
        resp = await client.get(DBRS_SEARCH, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("Search failed: %s — try capturing the real search API URL from browser", e)
        return []

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        data = resp.json()
        results = data.get("results", data.get("items", []))
        for item in results[:max_reports]:
            reports.append(
                {
                    "title": item.get("title", "unknown"),
                    "url": item.get("url", item.get("downloadUrl", "")),
                    "date": item.get("date", item.get("publishDate", "")),
                    "type": item.get("documentType", ""),
                }
            )
    else:
        logger.warning(
            "Got non-JSON response from search. "
            "DBRS may require capturing the API URL from browser dev tools."
        )

    return reports


async def download_report(client: httpx.AsyncClient, report: dict) -> bool:
    """Download a single DBRS report PDF."""
    title = report["title"]
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:100]
    date_prefix = report.get("date", "unknown")[:10]
    filename = f"{date_prefix}_{safe_title}.pdf"
    out_path = DATA_DIR / filename

    if out_path.exists():
        logger.info("  Skipping %s (already exists)", filename)
        return False

    url = report.get("url", "")
    if not url:
        logger.warning("  No URL for report: %s", title)
        return False

    if not url.startswith("http"):
        url = f"{DBRS_BASE}{url}"

    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("  Failed %s: %s", filename, e.response.status_code)
        return False

    out_path.write_bytes(resp.content)

    # Save metadata
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {**report, "download_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2,
        )
    )

    size_mb = len(resp.content) / (1024 * 1024)
    logger.info("  Downloaded %s (%.1f MB)", filename, size_mb)
    return True


async def run(max_reports: int = 50) -> None:
    """Main collection loop."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        reports = await search_cmbs_reports(client, max_reports)
        logger.info("Found %d CMBS reports", len(reports))

        total = 0
        for report in reports:
            downloaded = await download_report(client, report)
            if downloaded:
                total += 1

    logger.info("Done. Downloaded %d new reports to %s", total, DATA_DIR)
    if total == 0:
        logger.info(
            "If no reports downloaded, capture API calls from browser dev tools to update "
            "the search URL in this script."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DBRS CMBS reports")
    parser.add_argument(
        "--max-reports", type=int, default=50, help="Max reports to download (default: 50)"
    )
    args = parser.parse_args()
    asyncio.run(run(max_reports=args.max_reports))


if __name__ == "__main__":
    main()
