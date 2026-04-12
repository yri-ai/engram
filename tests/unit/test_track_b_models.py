"""Tests for Track B canonical models."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket


def test_delinquency_bucket_from_raw():
    assert DelinquencyBucket.from_raw("F", "1") == DelinquencyBucket.CURRENT
    assert DelinquencyBucket.from_raw("F", "2") == DelinquencyBucket.D30
    assert DelinquencyBucket.from_raw("V", "1") == DelinquencyBucket.CURRENT
    assert DelinquencyBucket.from_raw("V", "2") == DelinquencyBucket.D30
    assert DelinquencyBucket.from_raw("F", "") == DelinquencyBucket.CURRENT
    assert DelinquencyBucket.from_raw("R", "1") == DelinquencyBucket.REO
    assert DelinquencyBucket.from_raw("R", "") == DelinquencyBucket.REO


def test_delinquency_bucket_from_months():
    assert DelinquencyBucket.from_months_delinquent(0) == DelinquencyBucket.CURRENT
    assert DelinquencyBucket.from_months_delinquent(1) == DelinquencyBucket.D30
    assert DelinquencyBucket.from_months_delinquent(2) == DelinquencyBucket.D60
    assert DelinquencyBucket.from_months_delinquent(3) == DelinquencyBucket.D90
    assert DelinquencyBucket.from_months_delinquent(6) == DelinquencyBucket.D90_PLUS


def test_event_build_id():
    e = TrackBEvent(
        loan_id="1023374917",
        as_of=date(2025, 12, 1),
        bucket=DelinquencyBucket.CURRENT,
        current_upb=73000.0,
    )
    assert e.event_id == "1023374917-202512"
    assert e.message_id == "track-b-1023374917-202512"


def test_event_features_dict():
    e = TrackBEvent(
        loan_id="LN123",
        as_of=date(2026, 1, 1),
        bucket=DelinquencyBucket.D30,
        current_upb=250000.0,
        interest_rate=7.5,
        credit_score=720,
        state="CA",
        original_upb=300000.0,
    )
    features = e.features_dict()
    assert features["bucket"] == "d30"
    assert features["current_upb"] == 250000.0
    assert features["interest_rate"] == 7.5
    assert features["credit_score"] == 720
    assert features["state"] == "CA"


def test_event_to_canonical_row():
    e = TrackBEvent(
        loan_id="LN123",
        as_of=date(2026, 1, 1),
        bucket=DelinquencyBucket.D30,
        current_upb=250000.0,
    )
    row = e.to_canonical_row(next_bucket=DelinquencyBucket.D60, split="eval")
    assert row["event_id"] == "LN123-202601"
    assert row["message_id"] == "track-b-LN123-202601"
    assert row["loan_id"] == "LN123"
    assert row["as_of"] == "2026-01-01"
    assert row["split"] == "eval"
    assert row["features"]["bucket"] == "d30"
    assert row["label"]["next_bucket"] == "d60"
    assert row["label"]["horizon_months"] == 1
