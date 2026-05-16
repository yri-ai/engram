#!/usr/bin/env python3
"""Fetch Ginnie Mae bulk data (pool-level and loan-level).

Usage:
    python scripts/data_collection/fetch_ginnie.py [--months 12]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ginnie"
BASE_URL = "https://bulk.ginniemae.gov/protectedfiledownload.aspx"
DISCLOSURE_INDEX_URL = "https://bulk.ginniemae.gov/Disclosure/"
PROFILE_REDIRECT_MARKER = "Disclosure Data Download Account Settings"

# Monthly files we want, with {YYYYMM} placeholder
MONTHLY_FILES = [
    # SF Portfolio
    "monthlySFPS_{period}.zip",
    "monthlySFS_{period}.zip",
    # SF Loan Level
    "llmon1_{period}.zip",
    "llmon2_{period}.zip",
    # SF Liquidations
    "llmonliq_{period}.zip",
    # MF Portfolio
    "mfplmon3_{period}.zip",
    # Factors
    "factorA1_{period}.zip",
    "factorA2_{period}.zip",
    # Loan Performance
    "LoanPerf_{period}.zip",
    # CPR
    "CPRmon_{period}.zip",
    # Forbearance
    "plmonforb_{period}.zip",
    "llmonforb_{period}.zip",
    # Payment History
    "llpaymhist_{period}.zip",
]


def parse_available_filenames(html: str) -> set[str]:
    links = re.findall(r"dlfile=data_bulk/([^\"'&\s>]+)", html, flags=re.IGNORECASE)
    return {name for name in links if name}


def extract_period(filename: str) -> str | None:
    match = re.search(r"_(20\d{4})\.(zip|txt|csv)$", filename)
    return match.group(1) if match else None


def select_latest_periods(periods: list[str], months: int) -> set[str]:
    unique_sorted = sorted(set(periods), reverse=True)
    return set(unique_sorted[:months])


def generate_periods(months_back: int) -> list[str]:
    """Generate YYYYMM period strings going back N months from today."""
    periods = []
    now = datetime.now()
    for i in range(months_back):
        dt = now - timedelta(days=30 * i)
        periods.append(dt.strftime("%Y%m"))
    return sorted(set(periods))


async def download_file(
    client: httpx.AsyncClient,
    filename: str,
    out_dir: Path,
) -> tuple[bool, bool]:
    """Download a single file from Ginnie Mae bulk portal."""
    out_path = out_dir / filename
    if out_path.exists():
        if zipfile.is_zipfile(out_path):
            logger.info("  Skipping %s (already exists)", filename)
            return False, False
        logger.warning("  Removing invalid existing file for %s", filename)
        out_path.unlink()

    url = f"{BASE_URL}?dlfile=data_bulk/{filename}"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("  Failed %s: %s", filename, e.response.status_code)
        return False, False

    if "pages/profile.aspx" in str(resp.url).lower() or PROFILE_REDIRECT_MARKER in resp.text:
        logger.warning("  Access redirected to Ginnie profile page for %s", filename)
        return False, True

    # Check if we got an actual file vs an error/redirect page
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type and len(resp.content) < 10000:
        logger.warning(
            "  Got HTML instead of data for %s (likely missing file or bad URL)", filename
        )
        return False, False

    if not zipfile.is_zipfile(io.BytesIO(resp.content)):
        logger.warning("  Response for %s is not a ZIP archive", filename)
        return False, False

    out_path.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    logger.info("  Downloaded %s (%.1f MB)", filename, size_mb)
    return True, False


async def discover_targets(client: httpx.AsyncClient, months: int) -> list[tuple[str, str]]:
    resp = await client.get(DISCLOSURE_INDEX_URL)
    resp.raise_for_status()
    available = parse_available_filenames(resp.text)

    template_prefixes = {
        template.split("{period}")[0] for template in MONTHLY_FILES if "{period}" in template
    }

    candidates: list[tuple[str, str]] = []
    period_pool: list[str] = []
    for filename in available:
        if not any(filename.startswith(prefix) for prefix in template_prefixes):
            continue
        period = extract_period(filename)
        if period is None:
            continue
        period_pool.append(period)
        candidates.append((period, filename))

    chosen_periods = select_latest_periods(period_pool, months)
    selected = sorted([(p, f) for (p, f) in candidates if p in chosen_periods])
    return selected


def resolve_cookie(cli_cookie: str | None, cookie_file: str | None) -> str | None:
    if cli_cookie:
        return cli_cookie.strip()

    if cookie_file:
        path = Path(cookie_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    env_cookie = os.getenv("GINNIE_COOKIE", "").strip()
    return env_cookie or None


async def run(months: int = 12, cookie: str | None = None) -> None:
    """Main collection loop."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Cookie": cookie} if cookie else None
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True, headers=headers) as client:
        targets = await discover_targets(client, months)
        if not targets:
            logger.warning(
                "No matching monthly target files discovered on %s", DISCLOSURE_INDEX_URL
            )
            return

        periods = sorted({period for period, _ in targets})
        logger.info(
            "Discovered %d target files across periods %s to %s",
            len(targets),
            periods[0],
            periods[-1],
        )

        total = 0
        auth_blocked = False
        for period, filename in targets:
            period_dir = DATA_DIR / period
            period_dir.mkdir(parents=True, exist_ok=True)

            downloaded, auth_required = await download_file(client, filename, period_dir)
            if downloaded:
                total += 1
            if auth_required:
                auth_blocked = True
                break

    if auth_blocked:
        logger.error(
            "Ginnie download appears to require a Disclosure Data Download account. "
            "Visit https://www.ginniemae.gov/Pages/profile.aspx?src=%2fdata_and_reports%2fdisclosure_data%2fPages%2fdatadownload_bulk.aspx"
        )

    logger.info("Done. Downloaded %d new files to %s", total, DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Ginnie Mae bulk data")
    parser.add_argument(
        "--months", type=int, default=12, help="Months of history to fetch (default: 12)"
    )
    parser.add_argument(
        "--cookie",
        help="Raw Cookie header value for bulk.ginniemae.gov (optional).",
    )
    parser.add_argument(
        "--cookie-file",
        help="Path to a text file containing raw Cookie header value (optional).",
    )
    args = parser.parse_args()
    cookie = resolve_cookie(args.cookie, args.cookie_file)
    asyncio.run(run(months=args.months, cookie=cookie))


if __name__ == "__main__":
    main()
