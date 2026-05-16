"""Parser for Ginnie Mae LoanPerf pipe-delimited files."""

from __future__ import annotations

from datetime import date
from typing import IO, Iterator

from engram.models.track_b import DelinquencyBucket, TrackBEvent


# LoanPerf pipe-delimited field indices
_IDX_LOAN_SEQ = 6
_IDX_DELINQ_STATUS = 8
_IDX_DELINQ_MONTHS = 9
_IDX_INTEREST_RATE = 13
_IDX_ORIGINAL_UPB = 14
_IDX_CURRENT_UPB = 16
_IDX_CREDIT_SCORE = 26
_IDX_STATE = 34
_IDX_REPORT_PERIOD = 39


def _parse_float(val: str) -> float | None:
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_int(val: str) -> int | None:
    val = val.strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _parse_period(val: str) -> date | None:
    """Parse YYYYMM period string to first-of-month date."""
    val = val.strip()
    if len(val) != 6:
        return None
    try:
        return date(int(val[:4]), int(val[4:6]), 1)
    except ValueError:
        return None


def parse_loanperf_line(line: str) -> TrackBEvent | None:
    """Parse a single LP record line into a TrackBEvent.

    Returns None if the line is not a valid LP record.
    """
    fields = line.split("|")
    if len(fields) < 40 or fields[0] != "LP":
        return None

    loan_id = fields[_IDX_LOAN_SEQ].strip()
    if not loan_id:
        return None

    as_of = _parse_period(fields[_IDX_REPORT_PERIOD])
    if as_of is None:
        return None

    delinq_status = fields[_IDX_DELINQ_STATUS].strip()
    delinq_months = fields[_IDX_DELINQ_MONTHS].strip()
    bucket = DelinquencyBucket.from_raw(delinq_status, delinq_months)

    current_upb = _parse_float(fields[_IDX_CURRENT_UPB])
    if current_upb is None:
        return None

    return TrackBEvent(
        loan_id=loan_id,
        as_of=as_of,
        bucket=bucket,
        current_upb=current_upb,
        interest_rate=_parse_float(fields[_IDX_INTEREST_RATE]),
        original_upb=_parse_float(fields[_IDX_ORIGINAL_UPB]),
        credit_score=_parse_int(fields[_IDX_CREDIT_SCORE]),
        state=fields[_IDX_STATE].strip() or None,
    )


def parse_loanperf_records(
    stream: IO[bytes],
    limit: int | None = None,
) -> Iterator[TrackBEvent]:
    """Parse LoanPerf records from a byte stream.

    Args:
        stream: File-like object producing bytes (e.g., from zipfile.open()).
        limit: Maximum number of records to yield.
    """
    count = 0
    for raw_line in stream:
        if limit is not None and count >= limit:
            return
        line = raw_line.decode("ascii", errors="replace").rstrip()
        if not line.startswith("LP|"):
            continue
        event = parse_loanperf_line(line)
        if event is not None:
            yield event
            count += 1
