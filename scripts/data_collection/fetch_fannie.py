#!/usr/bin/env python3
"""Fetch Fannie/Freddie Enterprise PUDB archives from FHFA (public, no auth).

Usage:
    python scripts/data_collection/fetch_fannie.py [--start-year 2018]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fannie"
INDEX_URL = "https://www.fhfa.gov/data/public-use-database"
DOC_BASE = "https://www.fhfa.gov"


def extract_zip_links(html: str, start_year: int) -> list[str]:
    links = set(re.findall(r"/document/[^\"'\s>]+\.zip", html, flags=re.IGNORECASE))
    filtered: list[str] = []
    for path in sorted(links):
        name = path.rsplit("/", 1)[-1].lower()
        if "pudb" not in name and "enterprise" not in name:
            continue
        year_match = re.search(r"(20\d{2})", name)
        if year_match and int(year_match.group(1)) < start_year:
            continue
        filtered.append(f"{DOC_BASE}{path}")
    return filtered


def output_name_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1]


async def discover_links(client: httpx.AsyncClient, start_year: int) -> list[str]:
    resp = await client.get(INDEX_URL)
    resp.raise_for_status()
    return extract_zip_links(resp.text, start_year)


async def download_url(client: httpx.AsyncClient, url: str) -> bool:
    name = output_name_from_url(url)
    out_path = DATA_DIR / name
    if out_path.exists():
        logger.info("  Skipping %s (already exists)", name)
        return False

    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("  Failed %s: %s", name, e.response.status_code)
        return False

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        logger.warning("  Got HTML for %s (not a downloadable archive)", name)
        return False

    out_path.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    logger.info("  Downloaded %s (%.1f MB)", name, size_mb)
    return True


async def run(start_year: int = 2018) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        links = await discover_links(client, start_year)
        logger.info("Discovered %d FHFA PUDB archives", len(links))

        total = 0
        for url in links:
            if await download_url(client, url):
                total += 1

    logger.info("Done. Downloaded %d new files to %s", total, DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FHFA Enterprise PUDB archives")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    args = parser.parse_args()
    asyncio.run(run(start_year=args.start_year))


if __name__ == "__main__":
    main()
