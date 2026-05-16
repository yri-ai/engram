"""Tests for Track B dataset builder with leakage-safe splits."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.track_b_dataset import (
    assign_splits,
    build_labeled_rows,
    validate_no_leakage,
    LeakageError,
)


def _make_events() -> list[TrackBEvent]:
    """Create a sequence of events for one loan across 4 months."""
    return [
        TrackBEvent(loan_id="LN1", as_of=date(2025, 10, 1), bucket=DelinquencyBucket.CURRENT, current_upb=100000.0),
        TrackBEvent(loan_id="LN1", as_of=date(2025, 11, 1), bucket=DelinquencyBucket.CURRENT, current_upb=99500.0),
        TrackBEvent(loan_id="LN1", as_of=date(2025, 12, 1), bucket=DelinquencyBucket.D30, current_upb=99000.0),
        TrackBEvent(loan_id="LN1", as_of=date(2026, 1, 1), bucket=DelinquencyBucket.D60, current_upb=98500.0),
        TrackBEvent(loan_id="LN1", as_of=date(2026, 2, 1), bucket=DelinquencyBucket.D90, current_upb=98000.0),
        # Second loan
        TrackBEvent(loan_id="LN2", as_of=date(2025, 11, 1), bucket=DelinquencyBucket.CURRENT, current_upb=200000.0),
        TrackBEvent(loan_id="LN2", as_of=date(2025, 12, 1), bucket=DelinquencyBucket.CURRENT, current_upb=199500.0),
        TrackBEvent(loan_id="LN2", as_of=date(2026, 1, 1), bucket=DelinquencyBucket.CURRENT, current_upb=199000.0),
    ]


def test_build_labeled_rows():
    events = _make_events()
    rows = build_labeled_rows(events)
    # Last event per loan has no next_bucket, so it gets dropped
    # LN1: 4 labeled (Oct→Nov, Nov→Dec, Dec→Jan, Jan→Feb)
    # LN2: 2 labeled (Nov→Dec, Dec→Jan)
    assert len(rows) == 6
    # Check first row
    assert rows[0]["loan_id"] == "LN1"
    assert rows[0]["features"]["bucket"] == "current"
    assert rows[0]["label"]["next_bucket"] == "current"  # Oct→Nov: stayed current


def test_build_labeled_rows_captures_transitions():
    events = _make_events()
    rows = build_labeled_rows(events)
    # LN1 Dec: current → d30
    dec_row = [r for r in rows if r["loan_id"] == "LN1" and r["as_of"] == "2025-12-01"]
    assert len(dec_row) == 1
    assert dec_row[0]["features"]["bucket"] == "d30"
    assert dec_row[0]["label"]["next_bucket"] == "d60"


def test_assign_splits():
    events = _make_events()
    rows = build_labeled_rows(events)
    split_rows = assign_splits(rows, train_end="2025-12-31", eval_end="2026-01-31")
    trains = [r for r in split_rows if r["split"] == "train"]
    evals = [r for r in split_rows if r["split"] == "eval"]
    holdouts = [r for r in split_rows if r["split"] == "holdout"]
    # Train: as_of <= 2025-12-31
    assert all(r["as_of"] <= "2025-12-31" for r in trains)
    # Eval: 2026-01-01 to 2026-01-31
    assert all("2026-01-01" <= r["as_of"] <= "2026-01-31" for r in evals)


def test_validate_no_leakage_passes():
    events = _make_events()
    rows = build_labeled_rows(events)
    split_rows = assign_splits(rows, train_end="2025-12-31", eval_end="2026-01-31")
    # Should not raise
    validate_no_leakage(split_rows)


def test_validate_no_leakage_catches_duplicate_message_id():
    rows = [
        {"message_id": "track-b-LN1-202510", "loan_id": "LN1", "as_of": "2025-10-01", "split": "train"},
        {"message_id": "track-b-LN1-202510", "loan_id": "LN1", "as_of": "2025-10-01", "split": "eval"},
    ]
    try:
        validate_no_leakage(rows)
        assert False, "Should have raised LeakageError"
    except LeakageError:
        pass


def test_validate_no_leakage_catches_same_loan_month_across_splits():
    rows = [
        {"message_id": "track-b-LN1-202510-a", "loan_id": "LN1", "as_of": "2025-10-01", "split": "train"},
        {"message_id": "track-b-LN1-202510-b", "loan_id": "LN1", "as_of": "2025-10-15", "split": "eval"},
    ]
    try:
        validate_no_leakage(rows)
        assert False, "Should have raised LeakageError"
    except LeakageError:
        pass
