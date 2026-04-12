"""Parser for Ginnie Mae payment history (llpaymhist) files.

Each LL record contains a 48-character string encoding monthly delinquency
status for the past 48 months. Character position 0 = most recent month,
position 47 = oldest month. Each digit represents months delinquent:
  0 = current
  1 = 30-day delinquent
  2 = 60-day
  3 = 90-day
  4-8 = 90+ days (increasingly severe)
  9 = 90+ days / seriously delinquent

This parser expands each history string into 48 TrackBEvent objects,
one per month, giving us multi-month transition data from a single file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import IO, Iterator

from engram.models.track_b import DelinquencyBucket, TrackBEvent


@dataclass
class PayHistRecord:
    """Raw payment history record from llpaymhist file."""

    loan_id: str
    pool_id: str
    coupon: str
    history: str  # 48-char delinquency string
    report_period: str  # YYYYMM of the file


def _delinq_char_to_bucket(ch: str) -> DelinquencyBucket:
    """Convert a single history character to a delinquency bucket."""
    if ch == "0":
        return DelinquencyBucket.CURRENT
    if ch == "1":
        return DelinquencyBucket.D30
    if ch == "2":
        return DelinquencyBucket.D60
    if ch == "3":
        return DelinquencyBucket.D90
    # 4-9 are all 90+ days
    return DelinquencyBucket.D90_PLUS


def _month_offset(base_year: int, base_month: int, offset: int) -> date:
    """Subtract offset months from base date, return first-of-month."""
    total = base_year * 12 + (base_month - 1) - offset
    year = total // 12
    month = total % 12 + 1
    return date(year, month, 1)


def parse_payhist_records(
    stream: IO[bytes],
    limit: int | None = None,
) -> Iterator[PayHistRecord]:
    """Parse LL records from a payment history byte stream."""
    count = 0
    report_period = ""

    for raw_line in stream:
        line = raw_line.decode("ascii", errors="replace").rstrip()

        if line.startswith("HH|"):
            fields = line.split("|")
            if len(fields) >= 2:
                report_period = fields[1].strip()
            continue

        if not line.startswith("LL|"):
            continue

        if limit is not None and count >= limit:
            return

        fields = line.split("|")
        if len(fields) < 5:
            continue

        pool_id = fields[1].strip()
        loan_id = fields[2].strip()
        coupon = fields[3].strip()
        history = fields[4].strip()

        if not loan_id or len(history) < 48:
            continue

        yield PayHistRecord(
            loan_id=loan_id,
            pool_id=pool_id,
            coupon=coupon,
            history=history[:48],
            report_period=report_period,
        )
        count += 1


def expand_history_to_events(rec: PayHistRecord) -> list[TrackBEvent]:
    """Expand a payment history record into 48 monthly TrackBEvent objects.

    Returns events sorted chronologically (oldest first).
    The history string index 0 = most recent month, index 47 = oldest.
    """
    year = int(rec.report_period[:4])
    month = int(rec.report_period[4:6])

    events = []
    for i in range(48):
        ch = rec.history[i]
        bucket = _delinq_char_to_bucket(ch)
        as_of = _month_offset(year, month, i)

        events.append(
            TrackBEvent(
                loan_id=rec.loan_id,
                as_of=as_of,
                bucket=bucket,
                current_upb=0.0,  # Not available in payhist
            )
        )

    # Sort chronologically (oldest first)
    events.sort(key=lambda e: e.as_of)
    return events
