"""Tests for Ginnie Mae payment history parser."""

from datetime import date
from io import BytesIO

from engram.models.track_b import DelinquencyBucket
from engram.services.track_b_payhist_parser import (
    parse_payhist_records,
    expand_history_to_events,
    PayHistRecord,
)


SAMPLE_DATA = (
    b"HH|202602|20260309\n"
    b"PH|36202CNP4|002198|M|SF|19960401|    |202602\n"
    b"LL|002198|1023374299|3871|000000000000000000000000111009080706050403020201\n"
    b"LL|002198|1022574978|4094|000000000000000000000000000000000000000000000000\n"
    b"PT|36202CNP4|002198|M|SF|19960401|    |202602|0000002\n"
)


def test_parse_payhist_records():
    stream = BytesIO(SAMPLE_DATA)
    records = list(parse_payhist_records(stream))
    assert len(records) == 2
    assert records[0].loan_id == "1023374299"
    assert len(records[0].history) == 48
    assert records[1].loan_id == "1022574978"


def test_parse_payhist_respects_limit():
    stream = BytesIO(SAMPLE_DATA)
    records = list(parse_payhist_records(stream, limit=1))
    assert len(records) == 1


def test_expand_history_all_current():
    rec = PayHistRecord(
        loan_id="LN001",
        pool_id="002198",
        coupon="4094",
        history="000000000000000000000000000000000000000000000000",
        report_period="202602",
    )
    events = expand_history_to_events(rec)
    assert len(events) == 48
    assert all(e.bucket == DelinquencyBucket.CURRENT for e in events)
    # Sorted oldest first
    assert events[0].as_of == date(2022, 3, 1)
    # Most recent month is Feb 2026
    assert events[-1].as_of == date(2026, 2, 1)


def test_expand_history_with_transitions():
    # Loan goes: current for most months, then 1, 2, 3, 4, 5, ... delinquent
    rec = PayHistRecord(
        loan_id="LN002",
        pool_id="002198",
        coupon="3871",
        history="000000000000000000000000111009080706050403020201",
        report_period="202602",
    )
    events = expand_history_to_events(rec)
    assert len(events) == 48

    # Sorted oldest first — most recent is last
    assert events[-1].bucket == DelinquencyBucket.CURRENT  # Feb 2026 = '0'

    # Transitions exist in the older portion of the history
    d30_events = [e for e in events if e.bucket == DelinquencyBucket.D30]
    assert len(d30_events) > 0


def test_expand_history_correct_dates():
    rec = PayHistRecord(
        loan_id="LN003",
        pool_id="P1",
        coupon="4000",
        history="0" * 48,
        report_period="202602",
    )
    events = expand_history_to_events(rec)
    # Verify chronological order (oldest first after sorting)
    dates = [e.as_of for e in events]
    assert dates == sorted(dates)
    # First event = oldest = 47 months before 2026-02
    assert events[0].as_of == date(2022, 3, 1)
    assert events[-1].as_of == date(2026, 2, 1)


def test_expand_history_d90_plus():
    # '9' in history = 90+ days delinquent
    rec = PayHistRecord(
        loan_id="LN004",
        pool_id="P1",
        coupon="4000",
        history="9" + "0" * 47,
        report_period="202602",
    )
    events = expand_history_to_events(rec)
    assert events[-1].bucket == DelinquencyBucket.D90_PLUS  # most recent = '9'
