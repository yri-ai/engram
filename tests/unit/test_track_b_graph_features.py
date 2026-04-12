"""Tests for Track B graph feature extraction."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.track_b_graph_features import extract_features_from_event_history


def test_extract_features_basic():
    """Features from a single event with no history."""
    events = [
        TrackBEvent(loan_id="LN1", as_of=date(2025, 10, 1), bucket=DelinquencyBucket.CURRENT, current_upb=100000.0,
                    interest_rate=7.0, credit_score=720, state="CA"),
    ]
    features = extract_features_from_event_history(events, as_of=date(2025, 10, 1), loan_id="LN1")
    assert features["message_id"] == "track-b-LN1-202510"
    assert features["bucket"] == "current"
    assert features["current_upb"] == 100000.0
    assert features["interest_rate"] == 7.0
    assert features["credit_score"] == 720
    assert features["state"] == "CA"
    assert features["months_observed"] == 1
    assert features["bucket_changes_3m"] == 0


def test_extract_features_with_history():
    """Features capture delinquency transitions over time."""
    events = [
        TrackBEvent(loan_id="LN1", as_of=date(2025, 8, 1), bucket=DelinquencyBucket.CURRENT, current_upb=100000.0),
        TrackBEvent(loan_id="LN1", as_of=date(2025, 9, 1), bucket=DelinquencyBucket.CURRENT, current_upb=99500.0),
        TrackBEvent(loan_id="LN1", as_of=date(2025, 10, 1), bucket=DelinquencyBucket.D30, current_upb=99000.0),
    ]
    features = extract_features_from_event_history(events, as_of=date(2025, 10, 1), loan_id="LN1")
    assert features["months_observed"] == 3
    assert features["bucket_changes_3m"] == 1  # current → d30
    assert features["prev_bucket"] == "current"
    assert features["upb_change_pct"] is not None


def test_extract_features_no_matching_loan():
    """Returns None when loan not found."""
    events = [
        TrackBEvent(loan_id="LN1", as_of=date(2025, 10, 1), bucket=DelinquencyBucket.CURRENT, current_upb=100000.0),
    ]
    features = extract_features_from_event_history(events, as_of=date(2025, 10, 1), loan_id="LN999")
    assert features is None
