#!/usr/bin/env python3
"""Fetch CMBS ABS-EE filings from SEC EDGAR.

Downloads EX-102 XML asset data files for Commercial Mortgage-Backed Securities.
No API key needed. Respects SEC rate limit (10 req/sec).

Usage:
    python scripts/data_collection/fetch_edgar_cmbs.py [--max-filings N] [--start-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "edgar" / "cmbs"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

HEADERS = {
    "User-Agent": "Engram Research engram-research@yri.ai",
    "Accept": "application/json",
}

# Minimum interval between requests (100ms = 10 req/sec)
RATE_LIMIT_INTERVAL = 0.12
_last_request_time = 0.0
MAX_RETRIES = 5


async def rate_limited_get(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    global _last_request_time
    for attempt in range(MAX_RETRIES):
        now = time.monotonic()
        wait = RATE_LIMIT_INTERVAL - (now - _last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_time = time.monotonic()

        try:
            resp = await client.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError, httpx.ConnectTimeout) as e:
            if attempt + 1 >= MAX_RETRIES:
                raise
            backoff = 0.5 * (2**attempt)
            logger.warning(
                "Transient network error on %s (attempt %d/%d): %s; retrying in %.1fs",
                url,
                attempt + 1,
                MAX_RETRIES,
                type(e).__name__,
                backoff,
            )
            await asyncio.sleep(backoff)

    raise RuntimeError(f"Exhausted retries for {url}")


async def discover_cmbs_ciks(client: httpx.AsyncClient) -> dict[str, str]:
    """Find CMBS filers by searching EDGAR for 'commercial mortgage trust' ABS-EE filings."""
    ciks: dict[str, str] = {}
    queries = ['"commercial mortgage trust"', '"commercial mortgage pass-through"']

    for query in queries:
        offset = 0
        while True:
            resp = await rate_limited_get(
                client,
                EDGAR_SEARCH,
                params={
                    "q": query,
                    "forms": "ABS-EE",
                    "from": offset,
                    "size": 100,
                    "_source": "ciks,display_names",
                },
            )
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                for cik, name in zip(src.get("ciks", []), src.get("display_names", [])):
                    ciks[str(cik)] = name

            total = data.get("hits", {}).get("total", {})
            total_val = total.get("value", 0) if isinstance(total, dict) else total
            offset += 100
            if offset >= total_val:
                break

            logger.info("Discovered %d CIKs so far (offset %d/%d)", len(ciks), offset, total_val)

    logger.info("Discovered %d unique CMBS CIKs", len(ciks))
    return ciks


async def get_absee_filings(
    client: httpx.AsyncClient, cik: str, start_date: str | None = None
) -> list[dict]:
    """Get ABS-EE filing accession numbers for a CIK."""
    padded = cik.zfill(10)
    resp = await rate_limited_get(client, f"{EDGAR_SUBMISSIONS}/CIK{padded}.json")
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    filings = []
    for form, accession, date in zip(forms, accessions, dates):
        if form not in ("ABS-EE", "ABS-EE/A"):
            continue
        if start_date and date < start_date:
            continue
        filings.append({"accession": accession, "date": date})

    return filings


async def find_ex102_filename(
    client: httpx.AsyncClient, cik: str, accession_nodash: str
) -> str | None:
    """Find the EX-102 XML filename from the filing index."""
    url = f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/index.xml"
    try:
        resp = await rate_limited_get(client, url)
    except httpx.HTTPStatusError:
        return None

    root = ET.fromstring(resp.text)
    # index.xml uses <directory>/<item>/<name> with no namespace
    for item in root.iter("item"):
        name_el = item.find("name")
        if name_el is not None and name_el.text:
            if re.match(r".*102.*\.xml$", name_el.text, re.IGNORECASE):
                return name_el.text

    return None


async def download_ex102(
    client: httpx.AsyncClient,
    cik: str,
    accession: str,
    filing_date: str,
    entity_name: str,
) -> bool:
    """Download an EX-102 XML file for a single filing."""
    accession_nodash = accession.replace("-", "")

    # Check if already downloaded
    deal_dir = DATA_DIR / cik
    deal_dir.mkdir(parents=True, exist_ok=True)
    out_path = deal_dir / f"{accession_nodash}_{filing_date}.xml"
    if out_path.exists():
        return False

    # Find the EX-102 filename
    ex102_name = await find_ex102_filename(client, cik, accession_nodash)
    if not ex102_name:
        logger.warning("No EX-102 found for %s/%s", cik, accession)
        return False

    # Download it
    url = f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/{ex102_name}"
    try:
        resp = await rate_limited_get(client, url)
    except httpx.HTTPStatusError as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False

    out_path.write_text(resp.text)

    # Save metadata
    meta_path = deal_dir / f"{accession_nodash}_{filing_date}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "cik": cik,
                "entity_name": entity_name,
                "accession": accession,
                "filing_date": filing_date,
                "source_url": url,
                "download_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    return True


async def run(max_filings: int = 0, start_date: str | None = None) -> None:
    """Main collection loop."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        # Step 1: Discover CMBS CIKs
        ciks = await discover_cmbs_ciks(client)

        # Save CIK manifest
        manifest_path = DATA_DIR / "cmbs_ciks.json"
        manifest_path.write_text(json.dumps(ciks, indent=2))
        logger.info("Saved CIK manifest to %s", manifest_path)

        # Step 2: For each CIK, get filings and download EX-102s
        total_downloaded = 0
        for i, (cik, name) in enumerate(ciks.items()):
            logger.info("[%d/%d] Processing %s (CIK %s)", i + 1, len(ciks), name, cik)

            filings = await get_absee_filings(client, cik, start_date)
            logger.info("  Found %d ABS-EE filings", len(filings))

            for filing in filings:
                if max_filings and total_downloaded >= max_filings:
                    logger.info("Reached max filings limit (%d)", max_filings)
                    return

                downloaded = await download_ex102(
                    client, cik, filing["accession"], filing["date"], name
                )
                if downloaded:
                    total_downloaded += 1
                    logger.info(
                        "  Downloaded %s (%s) [%d total]",
                        filing["accession"],
                        filing["date"],
                        total_downloaded,
                    )

    logger.info("Done. Downloaded %d new EX-102 files to %s", total_downloaded, DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CMBS ABS-EE data from EDGAR")
    parser.add_argument(
        "--max-filings", type=int, default=0, help="Max filings to download (0=all)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="Only filings after this date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    asyncio.run(run(max_filings=args.max_filings, start_date=args.start_date))


if __name__ == "__main__":
    main()
